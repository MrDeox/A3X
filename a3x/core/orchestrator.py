import asyncio
import logging
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from a3x.core.llm_interface import LLMInterface
from a3x.fragments.registry import FragmentRegistry
from a3x.core.tool_registry import ToolRegistry
from a3x.core.tool_executor import ToolExecutor
from a3x.core.executors.action_executor import ActionExecutor
from a3x.core.prompt_builder import build_orchestrator_messages
from a3x.core.planner import json_find_gpt
from a3x.core.chat_monitor import chat_monitor_task
from a3x.core.executors.fragment_executor import (
    FragmentExecutor,
    FragmentExecutionError,
)
from a3x.core.constants import (
    STATUS_SUCCESS,
    STATUS_ERROR,
    STATUS_MAX_ITERATIONS,
    REASON_EXECUTOR_CALL_FAILED,
    REASON_DELEGATION_FAILED,
    REASON_FRAGMENT_FAILED,
    REASON_MAX_STEPS_REACHED,
    REASON_ORCHESTRATION_CRITICAL_ERROR,
    REASON_NO_ALLOWED_SKILLS,
    REASON_UNKNOWN,
    REASON_FRAGMENT_NOT_FOUND,
    REASON_SETUP_ERROR,
    REASON_PROMPT_BUILD_FAILED,
    REASON_LLM_PROCESSING_ERROR,
    REASON_LLM_ERROR,
    REASON_ACTION_EXECUTION_FAILED,
    REASON_CAPABILITY_NOT_FOUND
)

# Type hint for MemoryManager to avoid circular import
from typing import TYPE_CHECKING
from a3x.core.context import FragmentContext, SharedTaskContext

if TYPE_CHECKING:
    from a3x.core.memory.memory_manager import MemoryManager
    # from a3x.core.context import SharedTaskContext # Removed, imported above


# --- Helper Functions (Notifications) ---
async def notify_task_completion(task_id: str, message: str):
    # Placeholder: Implement actual notification logic (e.g., websocket, callback)
    print(f"[Notification] Task {task_id} completed: {message}")


async def notify_task_error(task_id: str, reason: str, details: Any):
    # Placeholder: Implement actual notification logic
    print(f"[Notification] Task {task_id} failed. Reason: {reason}. Details: {details}")


async def notify_task_update(task_id: str, update_message: str):
    # Placeholder: Implement actual notification logic
    print(f"[Notification] Task {task_id} update: {update_message}")


class TaskOrchestrator:
    """Orchestrates the execution of tasks by delegating to Fragments and Tools."""

    # --- Initialization ---
    def __init__(
        self,
        fragment_registry: FragmentRegistry,
        tool_registry: ToolRegistry,
        memory_manager: "MemoryManager",
        llm_interface: LLMInterface,
        workspace_root: Path,
        agent_logger: logging.Logger,
        config: Dict[str, Any],
    ):
        self.fragment_registry = fragment_registry
        self.tool_registry = tool_registry
        self.memory_manager = memory_manager
        self.llm_interface = llm_interface
        self.workspace_root = workspace_root
        self.logger = agent_logger.getChild("Orchestrator")
        self.monitor_task: Optional[asyncio.Task] = None
        self.config = config

        # <<< ADDED: Instantiate ToolExecutor and ActionExecutor >>>
        self.tool_executor = ToolExecutor(
            tool_registry=self.tool_registry
        )
        self.action_executor = ActionExecutor(tool_executor=self.tool_executor)
        self.logger.info("TaskOrchestrator initialized with ToolExecutor and ActionExecutor.")

    # <<< ADDED: Helper to find fragment by capability >>>
    def _find_fragment_by_capability(self, capability: str) -> Optional[str]:
        """Finds the name of the first registered fragment providing the specified capability."""
        self.logger.debug(f"Searching for fragment with capability: '{capability}'")
        for frag_name, frag_def in self.fragment_registry.get_all_definitions().items():
            if capability in frag_def.capabilities:
                self.logger.info(f"Found fragment '{frag_name}' providing capability '{capability}'.")
                return frag_name
        self.logger.warning(f"No registered fragment found providing capability: '{capability}'")
        return None

    async def _get_next_step_delegation(
        self,
        objective: str,
        history: List[Tuple[str, str]],
        shared_task_context: "SharedTaskContext",
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Delegates the next step decision to the OrchestratorFragment."""
        self.logger.info("[_get_next_step_delegation] Delegating decision to OrchestratorFragment...")

        # Import the specific fragment class
        try:
            from a3x.a3net.core.orchestrator_fragment import OrchestratorFragment
        except ImportError:
            self.logger.exception("Failed to import OrchestratorFragment.")
            return None, None, REASON_SETUP_ERROR

        try:
            # 1. Prepare context for the symbolic fragment
            available_fragment_ids = list(self.fragment_registry.get_all_definitions().keys())
            available_skill_names = self.tool_registry.list_tools()
            task_context_dict = await shared_task_context.get_task_data_copy()
            formatted_history = [{"thought_action": h[0], "observation": h[1]} for h in history]

            # 2. Instantiate and execute the OrchestratorFragment
            from a3x.fragments.base import FragmentDef
            orchestrator_fragment_def_internal = FragmentDef(
                name="orchestrator_internal_instance",
                fragment_class=OrchestratorFragment,
                description="Internal Orchestrator Fragment instance for decision making.",
                category="Management"
            )
            orchestrator_ctx = FragmentContext(
                logger=self.logger.getChild("OrchestratorFragment"),
                llm_interface=self.llm_interface,
                tool_registry=self.tool_registry,
                fragment_registry=self.fragment_registry,
                shared_task_context=shared_task_context,
                workspace_root=self.workspace_root,
                memory_manager=self.memory_manager,
                fragment_id="orchestrator_internal_instance",
                fragment_name="orchestrator_internal_instance",
                fragment_class=OrchestratorFragment,
                fragment_def=orchestrator_fragment_def_internal,
                config=self.config
            )
            orchestrator_fragment = OrchestratorFragment(ctx=orchestrator_ctx)

            decision_result = await orchestrator_fragment.execute(
                objective=objective,
                history=formatted_history,
                available_fragments=available_fragment_ids,
                available_skills=available_skill_names,
                task_context=task_context_dict
            )

            self.logger.debug(f"[_get_next_step_delegation] OrchestratorFragment result: {decision_result}")

            # 3. Parse and Validate the Fragment's Response
            if not isinstance(decision_result, dict):
                self.logger.error(f"OrchestratorFragment returned non-dict result: {decision_result}")
                return None, None, REASON_DELEGATION_FAILED

            component_name = decision_result.get("component_name")
            sub_task = decision_result.get("sub_task")
            decision_status = decision_result.get("status", "error")

            if decision_status != STATUS_SUCCESS:
                message = decision_result.get("message", "OrchestratorFragment decision failed.")
                self.logger.error(f"OrchestratorFragment decision failed: {message}")
                return None, None, REASON_DELEGATION_FAILED

            if not component_name or not sub_task:
                self.logger.error(f"OrchestratorFragment response missing 'component_name' or 'sub_task': {decision_result}")
                return None, None, REASON_DELEGATION_FAILED

            # 4. Validate Component Name
            if component_name not in self.fragment_registry.get_all_definitions():
                self.logger.error(f"OrchestratorFragment delegated to unknown fragment: '{component_name}'")
                return None, None, REASON_DELEGATION_FAILED

            self.logger.info(f"[_get_next_step_delegation] OrchestratorFragment decided: Delegate to '{component_name}' for sub-task '{sub_task[:100]}...'")
            return component_name, sub_task, None

        except Exception as e:
            self.logger.exception("[_get_next_step_delegation] Error executing/processing OrchestratorFragment:")
            return None, None, REASON_DELEGATION_FAILED

    async def _execute_fragment_task(
        self,
        component_name: str,
        sub_task: str,
        objective: str,
        shared_task_context: "SharedTaskContext",
        current_fragment_history: List[Tuple[str, str]] = [],
    ) -> Dict[str, Any]:
        """Executes a task by delegating to the FragmentExecutor."""
        log_prefix = f"[Orchestrator Task {shared_task_context.task_id}]"
        self.logger.info(f"{log_prefix} Executing fragment '{component_name}' for sub-task: '{sub_task[:100]}...'")

        fragment_executor = FragmentExecutor(
            fragment_registry=self.fragment_registry,
            tool_registry=self.tool_registry,
            memory_manager=self.memory_manager,
            logger=self.logger.getChild("FragmentExecutor"),
        )

        try:
            allowed_skills = self._get_allowed_skills(component_name, self.tool_registry, self.fragment_registry)
            if not allowed_skills and component_name != "PlannerFragment":
                self.logger.error(f"{log_prefix} No allowed skills found or determined for fragment '{component_name}'. Cannot execute.")
                return {"status": STATUS_ERROR, "reason": REASON_NO_ALLOWED_SKILLS, "message": f"No skills available/allowed for {component_name}"}

            result_dict = await fragment_executor.execute(
                fragment_name=component_name,
                sub_task=sub_task,
                objective=objective,
                shared_context=shared_task_context,
                allowed_skills=allowed_skills,
                fragment_history=current_fragment_history,
            )
            self.logger.info(f"{log_prefix} FragmentExecutor returned for '{component_name}': Status - {result_dict.get('status')}")
            self.logger.debug(f"{log_prefix} FragmentExecutor result dict: {result_dict}")
            return result_dict

        except FragmentExecutionError as fee:
            error_message = str(fee)
            self.logger.error(f"{log_prefix} FragmentExecutionError for '{component_name}': {error_message} (Status: {fee.status}, Reason: {fee.reason})")
            return {"status": fee.status, "reason": fee.reason, "message": error_message}
        except Exception as e:
            self.logger.exception(f"{log_prefix} Unexpected error calling FragmentExecutor for '{component_name}':")
            return {"status": STATUS_ERROR, "reason": REASON_EXECUTOR_CALL_FAILED, "message": f"Orchestrator failed to execute fragment: {e}"}

    async def _invoke_learning_cycle(
        self,
        objective: str,
        main_history: List,
        final_status: str,
        shared_context: "SharedTaskContext",
    ):
        task_id = shared_context.task_id
        log_prefix = f"[Orchestrator Task {task_id}]"
        self.logger.info(f"{log_prefix} Invoking learning cycle for task.")

        # Garantir max_steps definido para robustez em chamadas isoladas/testes
        if not hasattr(self, '_current_max_steps'):
            try:
                from a3x.core.config import ORCHESTRATOR_MAX_STEPS
                max_steps = ORCHESTRATOR_MAX_STEPS
                self.logger.warning(f"{log_prefix} max_steps não definido no contexto do ciclo de aprendizado. Usando default: {max_steps}")
            except ImportError:
                max_steps = 10
                self.logger.warning(f"{log_prefix} max_steps não definido e config não importável. Usando fallback: {max_steps}")
        else:
            max_steps = self._current_max_steps

        try:
            # Example: Create a summary or structured data from the execution
            learning_data = {
                "objective": objective,
                "final_status": final_status,
                "history": main_history,
                "final_answer": shared_context.get_data("final_answer", "N/A"),
                "steps_taken": len(main_history),
            }
            # Chama o método de aprendizado do MemoryManager e captura recomendação
            recommendation = await self.memory_manager.learn_from_task(learning_data)

            if recommendation:
                trigger = recommendation.get('trigger')
                trigger_context = recommendation.get('context', {}) # Contexto da falha/sucesso
                
                self.logger.info(f"{log_prefix} Recomendação do MemoryManager: Trigger '{trigger}'")

                # Mapeamento de triggers para nomes dos fragmentos meta
                trigger_to_fragment = {
                    'reflection_on_failure': 'MetaReflectorFragment',
                    'self_evolution': 'SelfEvolverFragment',
                    'reflection_on_success': 'MetaReflectorFragment', # Pode ser outro fragmento se desejar
                }
                fragment_name = trigger_to_fragment.get(trigger)
                if fragment_name:
                    meta_fragment = self.fragment_registry.get_fragment(fragment_name)
                    if meta_fragment:
                        try:
                            self.logger.info(f"{log_prefix} Executando fragmento meta: {fragment_name}...")
                            # Executar o fragmento, passando trigger_context como kwargs
                            meta_result = await meta_fragment.execute(**trigger_context)
                        except Exception as meta_exc:
                            self.logger.exception(f"{log_prefix} Erro ao executar fragmento meta {fragment_name}: {meta_exc}")
                self.logger.info(f"{log_prefix} Learning cycle completed (com recomendação).")

        except Exception:
            self.logger.exception(
                f"{log_prefix} Error during learning cycle invocation:"
            )

            # --- End of Orchestration Loop --- #

            # --- Handle Max Steps Reached --- #
            if not task_completed_successfully and current_step >= max_steps:
                self.logger.warning(f"{log_prefix} Task reached max steps ({max_steps}) without completion.")
                final_status = STATUS_MAX_ITERATIONS
                final_reason = REASON_MAX_STEPS_REACHED
                final_answer = "Task stopped: Maximum execution steps reached."
                shared_task_context.error_info = {"stage": "max_steps", "details": final_answer}
                shared_task_context.status = "failed_max_steps"
                await notify_task_error(shared_task_context.task_id, final_reason, shared_task_context.error_info)

        except Exception as e:
            self.logger.exception(f"{log_prefix} Critical error during orchestration loop:")
            final_status = STATUS_ERROR; final_reason = REASON_ORCHESTRATION_CRITICAL_ERROR
            final_answer = f"Task failed due to unexpected orchestrator error: {e}"
            if 'shared_task_context' in locals():
                 shared_task_context.error_info = {"stage": "orchestrator_critical", "details": str(e)}; shared_task_context.status = "failed_critical"
                 await notify_task_error(shared_task_context.task_id, final_reason, shared_task_context.error_info)
            else: await notify_task_error(task_id, final_reason, {"details": str(e)})

        finally:
            # --- Clean up --- #
            self.logger.info(f"{log_prefix} Orchestration loop finished. Final Status: {final_status}")
            if self.monitor_task and not self.monitor_task.done():
                self.monitor_task.cancel()
                await asyncio.sleep(0) # Allow cancellation to propagate
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    self.logger.info(f"{log_prefix} Chat monitor task successfully cancelled.")
                except Exception as monitor_cancel_err:
                    self.logger.error(f"{log_prefix} Error waiting for monitor task cancellation: {monitor_cancel_err}")
            else:
                self.logger.info(f"{log_prefix} Monitor task was already done or not started.")

            # Ensure shared_task_context and main_history are defined for this scope
            local_shared_task_context = locals().get('shared_task_context', None)
            local_main_history = locals().get('main_history', [])

            # --- Invoke Learning Cycle --- #
            if local_shared_task_context is not None:
                await self._invoke_learning_cycle(objective, local_main_history, final_status, local_shared_task_context)
            else:
                self.logger.warning(f"[Orchestrator] Skipping learning cycle: shared_task_context is None.")

            # --- Notify Task Completion/Error --- #
            if local_shared_task_context is not None:
                if final_status == STATUS_SUCCESS:
                    await notify_task_completion(local_shared_task_context.task_id, final_answer)
                elif final_status != "stopped_by_user" and getattr(local_shared_task_context, 'status', None) not in ["failed", "failed_critical", "failed_max_steps"]:
                    await notify_task_error(local_shared_task_context.task_id, final_reason if 'final_reason' in locals() else REASON_UNKNOWN, {"details": final_answer})

        # --- Return Final Result --- #
        safe_final_answer = locals().get('final_answer', None)
        safe_final_reason = locals().get('final_reason', None)
        safe_shared_task_context = locals().get('shared_task_context', None)
        safe_orchestration_history = locals().get('orchestration_history', [])
        safe_task_id = locals().get('task_id', None)
        return {
            "status": final_status,
            "answer": safe_final_answer,
            "task_id": safe_task_id,
            "history": safe_orchestration_history,
            "final_context": safe_shared_task_context.to_dict() if safe_shared_task_context else None
        }

    async def shutdown(self):
        # Cancel any potentially running monitor tasks if the orchestrator is shut down externally
        if (
            hasattr(self, "monitor_task")
            and self.monitor_task
            and not self.monitor_task.done()
        ):
            self.logger.info(
                "[Orchestrator] Shutdown requested, cancelling active monitor task."
            )
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                self.logger.info(
                    "[Orchestrator] Monitor task successfully cancelled during shutdown."
                )
            except Exception as e:
                self.logger.error(
                    f"[Orchestrator] Error waiting for monitor task cancellation during shutdown: {e}"
                )
        self.logger.info("[Orchestrator] Shutdown complete.")

    def _get_allowed_skills(
        self,
        fragment_name: str,
        tool_registry: ToolRegistry,
        fragment_registry: FragmentRegistry,
    ) -> List[str]:
        """Determines the list of allowed skills for a given fragment."""
        fragment_def = fragment_registry.get_fragment_definition(fragment_name)
        if fragment_def and fragment_def.managed_skills:
            self.logger.debug(f"Fragment '{fragment_name}' manages specific skills: {fragment_def.managed_skills}")
            allowed_skills = []
            all_registered_tools = tool_registry.list_tools()
            for skill_name in fragment_def.managed_skills:
                if skill_name in all_registered_tools:
                    allowed_skills.append(skill_name)
                else:
                    self.logger.warning(f"Managed skill '{skill_name}' for fragment '{fragment_name}' not found in ToolRegistry.")
            if not allowed_skills:
                self.logger.warning(f"Fragment '{fragment_name}' defined managed skills, but none were found in registry. Allowing all skills as fallback.")
                return list(all_registered_tools)
            return allowed_skills
        else:
            allowed_skills = list(tool_registry.list_tools())
            self.logger.debug(f"Fragment '{fragment_name}' does not manage specific skills or def not found. Allowing all {len(allowed_skills)} registered skills.")
            return allowed_skills
