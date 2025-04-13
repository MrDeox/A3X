import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Re-use the Fragment class from self_optimizer for metrics and state
# If it becomes more complex, it might need its own definition here.
from a3x.core.self_optimizer import FragmentState

# Placeholder for shared components if needed
# from a3x.core.llm_interface import LLMInterface
# from a3x.core.skills import SkillRegistry

logger = logging.getLogger(__name__)

# <<< NEW: Define FragmentDef dataclass >>>
@dataclass
class FragmentDef:
    name: str
    fragment_class: Type["BaseFragment"] # Forward reference to BaseFragment
    description: str
    category: str = "Execution" # Default category
    skills: List[str] = field(default_factory=list)
    managed_skills: List[str] = field(default_factory=list)


class BaseFragment(ABC):
    """
    Classe base abstrata para todos os Fragments especializados no A³X.
    Define a interface comum para execução, atualização de métricas e otimização.
    """
    FRAGMENT_NAME: str = "BaseFragment"

    def __init__(self, name: str, skills: List[str], prompt_template: str, llm_interface=None, skill_registry=None, config: Optional[Dict] = None):
        fragment_name = name if name else self.FRAGMENT_NAME
        self.state = FragmentState(fragment_name, skills, prompt_template)
        self.config = config or {}
        # Placeholder for dependencies (LLM, Skills) - inject or load dynamically
        # self.llm = llm_interface or LLMInterface()
        # self.skill_registry = skill_registry or SkillRegistry()
        self.optimizer = self._create_optimizer()
        logger.info(f"Fragment '{self.state.name}' initialized with {len(self.state.skills)} skills.")

    def _create_optimizer(self):
        """Instantiates the optimizer for this fragment."""
        pass # Implementado por subclasses

    async def run_and_optimize(self, task_objective: str, context: Optional[Dict] = None) -> Tuple[Dict, List[Dict]]:
        """
        Método wrapper que executa o Fragment, atualiza métricas e aciona o otimizador.
        Retorna o resultado final da execução e a lista completa de resultados intermediários.
        """
        start_time = time.time()
        final_result = {"status": "error", "message": "Execution did not yield a final result.", "type": "error"} # Ensure type exists
        execution_trace: List[Dict] = [] # Store all yielded results

        try:
            async for step_result in self.execute(task_objective, context):
                execution_trace.append(step_result)
                # The last result is assumed to be the final one for status checking
                final_result = step_result

            # Ensure final_result has a status if execute finishes without error but no explicit status
            if "status" not in final_result:
                 final_result["status"] = "completed" # Assume completed if generator finishes
                 logger.warning(f"Fragment '{self.state.name}' execution finished without explicit status. Assuming 'completed'.")


        except Exception as e:
             logger.error(f"Exception during Fragment '{self.state.name}' execution for task '{task_objective[:50]}...': {e}", exc_info=True)
             error_message = f"Unhandled exception: {e}"
             final_result = {"status": "error", "type": "error", "message": error_message}
             # Ensure at least one result exists for metric update and trace
             if not execution_trace:
                 execution_trace.append(final_result)
             else:
                 # Append error to existing trace if possible
                 execution_trace.append(final_result)

        finally:
            # --- Update Metrics and Optimize ---
            end_time = time.time()
            duration = end_time - start_time
            # Make sure 'status' exists before checking
            is_success = final_result.get("status") == "success"

            self.state.update_metrics(success=is_success, execution_time=duration)
            logger.info(f"Fragment '{self.state.name}' metrics updated. {self.state.get_status_summary()}")

            if self.optimizer:
                try:
                    optimized = await self.optimizer.optimize_if_needed()
                    if optimized:
                        logger.info(f"Optimizer ran for Fragment '{self.state.name}'.")
                except Exception as opt_err:
                    logger.error(f"Error during Fragment '{self.state.name}' optimization: {opt_err}", exc_info=True)

        # Return the *last* result yielded by execute() AND the full trace
        return final_result, execution_trace


    def get_name(self) -> str:
        return self.state.name

    def get_skills(self) -> List[str]:
        return self.state.skills

    def get_current_prompt(self) -> str:
        return self.state.current_prompt

    def get_status_summary(self) -> str:
         return self.state.get_status_summary()

    @abstractmethod
    def get_purpose(self) -> str:
         """Returns a one-sentence description of the fragment's main goal."""
         pass

    def get_description_for_routing(self) -> str:
        """Generates a description string suitable for the routing LLM prompt."""
        purpose = self.get_purpose()
        skills = self.get_skills()
        # Format skills nicely, limit if too many
        skills_str = ", ".join(skills[:10]) # Show max 10 skills
        if len(skills) > 10:
            skills_str += ", ..." # Indicate more exist
        return f"- {self.get_name()}: {purpose} Skills: [{skills_str}]"

# <<< NEW: Manager Fragment Base Class >>>
class ManagerFragment(BaseFragment):
    """
    Base class for Manager Fragments.
    Managers coordinate other Fragments or Skills within a specific domain.
    They receive a sub-task from the Orchestrator and decide how to fulfill it
    by delegating to lower-level components.
    """
    async def coordinate_execution(self, sub_task: str, context: Any) -> Dict[str, Any]:
        """
        Core logic for the Manager.
        Analyzes the sub_task and delegates to appropriate skills/fragments.
        This method MUST be implemented by subclasses.

        Args:
            sub_task: The specific objective assigned by the Orchestrator.
            context: The execution context (e.g., history, workspace).

        Returns:
            A dictionary representing the outcome of the coordinated execution.
            Should typically align with the standard fragment result format.
        """
        # Subclasses must implement this logic, e.g., using LLM calls or rules
        # to select and execute the correct skill(s).
        self.logger.warning(f"Manager {self.name}: coordinate_execution not implemented!")
        return {
            "status": "error",
            "action": f"{self.name}_coordination_failed",
            "data": {"message": "Coordination logic not implemented in manager."},
        }

    async def execute_task(self, sub_task: str, context: Any) -> Dict[str, Any]:
        """
        Overrides the BaseFragment execute_task.
        Instead of running a ReAct loop directly, it calls the coordination logic.
        """
        self.logger.info(f"Manager {self.name}: Starting coordination for sub-task: '{sub_task}'")
        try:
            # Delegate the core work to the coordination method
            result = await self.coordinate_execution(sub_task, context)
            self.logger.info(f"Manager {self.name}: Coordination finished. Status: {result.get('status')}")
            return result
        except Exception as e:
            self.logger.exception(f"Manager {self.name}: Error during coordination.")
            return {
                "status": "error",
                "action": f"{self.name}_coordination_error",
                "data": {"message": f"Exception during coordination: {str(e)}"},
            }

# <<< END NEW >>>

# Make sure existing classes are still defined below if any
# ... rest of the file ... 