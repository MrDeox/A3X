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
                    fragment_class = self.fragment_registry.get_fragment_class(fragment_name)
                    if fragment_class:
                        try:
                            fragment_def = self.fragment_registry.get_fragment_def(fragment_name)
                            from a3x.fragments.base import FragmentContext # Import local para evitar ciclos
                            meta_context = FragmentContext(
                                logger=self.logger,
                                llm_interface=self.llm_interface,
                                tool_registry=self.tool_registry,
                                fragment_registry=self.fragment_registry,
                                shared_task_context=shared_context,
                                workspace_root=self.workspace_root,
                                memory_manager=self.memory_manager,
                                heuristic_store=getattr(shared_context, 'heuristic_store', None),
                                fragment_id=f"meta_{trigger}_{shared_context.task_id}",
                                fragment_name=fragment_name,
                                fragment_class=fragment_class,
                                fragment_def=fragment_def,
                                config=self.fragment_registry.get_fragment_config(fragment_name)
                            )
                            # Instanciar o fragmento meta
                            fragment_instance = fragment_class(fragment_def=fragment_def, tool_registry=self.tool_registry)
                            print(f"[DEBUG orchestrator] fragment_instance type: {type(fragment_instance)} (class: {fragment_class})")
                            fragment_instance.set_context(meta_context)
                            self.logger.info(f"{log_prefix} Executando fragmento meta: {fragment_name}...")
                            # Executar o fragmento, passando trigger_context como kwargs
                            meta_result = await fragment_instance.execute(**trigger_context)
                            self.logger.info(f"{log_prefix} Fragmento meta {fragment_name} concluído. Resultado: {meta_result.get('status', 'N/A')}")
                        except Exception as meta_exc:
                            self.logger.error(f"{log_prefix} Erro ao instanciar ou executar fragmento meta {fragment_name}: {meta_exc}", exc_info=True)
                            if hasattr(self, 'exception_policy'):
                                self.exception_policy.handle(meta_exc, context=f"Erro no fragmento meta {fragment_name}")
                    else:
                        self.logger.error(f"{log_prefix} Fragmento '{fragment_name}' não encontrado no registro para o trigger '{trigger}'.")
                else:
                    self.logger.warning(f"{log_prefix} Trigger desconhecido recebido do MemoryManager: '{trigger}'")
            else:
                self.logger.info(f"{log_prefix} Nenhuma recomendação de aprendizado/evolução recebida do MemoryManager.")

            # O log "Learning cycle completed successfully." pode ser movido ou removido dependendo da lógica acima
            self.logger.info(f"{log_prefix} Learning cycle completed (com recomendação).")

        except Exception:
            self.logger.exception(
                f"{log_prefix} Error during learning cycle invocation:"
            )
            # Decide if this error should impact the overall task status ou apenas ser logado


        # --- Initialize Task State --- #
        shared_task_context = SharedTaskContext(task_id=task_id, initial_objective=objective)
        shared_task_context.status = "starting"
        orchestration_history: List[Tuple[str, str]] = []
        current_step = 0
        final_status = "in_progress"
        final_answer = "Task did not complete."
        task_completed_successfully = False

        # --- Start Chat Monitor --- #
        self.monitor_task = asyncio.create_task(
            chat_monitor_task(
                task_id=task_id, shared_task_context=shared_task_context,
                llm_interface=self.llm_interface, tool_registry=self.tool_registry,
                fragment_registry=self.fragment_registry, memory_manager=self.memory_manager,
                workspace_root=self.workspace_root
            )
        )
        self.logger.info(f"{log_prefix} Chat monitor started.")

        # --- Resolve max_steps --- #
        if max_steps is None:
            try:
                from a3x.core.config import ORCHESTRATOR_MAX_STEPS
                max_steps = ORCHESTRATOR_MAX_STEPS
            except ImportError:
                max_steps = 10
            self.logger.info(f"{log_prefix} Using max_steps: {max_steps}")

        # --- Orchestration Loop --- #
        try:
            while current_step < max_steps and not task_completed_successfully:
                current_step += 1
                self.logger.info(f"{log_prefix} --- Step {current_step}/{max_steps} --- ")
                shared_task_context.status = f"running_step_{current_step}"
                await notify_task_update(shared_task_context.task_id, f"Starting step {current_step}")

                # --- Check Chat Monitor for Interrupts --- #
                try:
                    monitor_message_entry = shared_task_context.internal_chat_queue.get_nowait()
                    if isinstance(monitor_message_entry, dict) and monitor_message_entry.get("type") == "CONTROL" and monitor_message_entry.get("content", {}).get("command") == "STOP":
                        self.logger.warning(f"{log_prefix} Received STOP signal from monitor. Halting task.")
                        final_status = "stopped_by_user"
                        final_answer = "Task stopped by user interaction."
                        await notify_task_error(shared_task_context.task_id, "user_interrupt", {"details": final_answer})
                        break
                    else:
                        self.logger.debug(f"{log_prefix} Received non-control message from queue: {monitor_message_entry}")
                except asyncio.QueueEmpty:
                    pass
                
                # <<< Reset loop-specific state before decision >>>
                component_name = None
                sub_task = None
                delegation_reason = None
                action_was_executed = False 
                pending_request_handled = False

                # --- Check for Pending Request FIRST --- 
                pending_request_obj = shared_task_context.pending_request
                if pending_request_obj:
                    self.logger.info(f"{log_prefix} Handling Pending Request for capability: '{pending_request_obj.capability_needed}'")
                    shared_task_context.pending_request = None # Clear the request
                    pending_request_handled = True

                    # Attempt to find fragment directly by capability
                    found_fragment_name = self._find_fragment_by_capability(pending_request_obj.capability_needed)
                    
                    if found_fragment_name:
                        component_name = found_fragment_name
                        sub_task = pending_request_obj.details or f"Fulfill request for capability '{pending_request_obj.capability_needed}' from {pending_request_obj.requested_by or 'unknown'}."
                        self.logger.info(f"{log_prefix} Delegating directly to '{component_name}' based on capability '{pending_request_obj.capability_needed}'.")
                    else:
                        # Fallback: No fragment found with capability, try asking OrchestratorFragment
                        self.logger.warning(f"{log_prefix} No fragment found for capability '{pending_request_obj.capability_needed}'. Falling back to OrchestratorFragment delegation.")
                        (
                            component_name,
                            sub_task,
                            delegation_reason,
                        ) = await self._get_next_step_delegation(
                            objective=f"{objective} (Context: Fulfilling request for {pending_request_obj.capability_needed} from {pending_request_obj.requested_by or 'unknown'} - {pending_request_obj.details or 'No details'})", 
                            history=orchestration_history, 
                            shared_task_context=shared_task_context
                        )
                        if delegation_reason:
                            self.logger.error(f"{log_prefix} Fallback delegation failed for pending request {pending_request_obj.capability_needed}: {delegation_reason}")
                            final_status = STATUS_ERROR
                            final_reason = REASON_DELEGATION_FAILED # Or maybe REASON_CAPABILITY_NOT_FOUND?
                            final_answer = f"Orchestration failed: Could not fulfill pending request for '{pending_request_obj.capability_needed}' ({delegation_reason})"
                            await notify_task_error(shared_task_context.task_id, final_reason, {"details": "Pending request handling failed"})
                            break # Exit loop
                        else:
                             self.logger.info(f"{log_prefix} OrchestratorFragment delegated pending request to '{component_name}'.")

                # --- Decide Next Action (if no pending request was handled) --- #
                if not pending_request_handled:
                    # <<< Existing logic using plan or _get_next_step_delegation >>>
                    current_plan = shared_task_context.task_data.get("current_plan")
                    next_step_index = shared_task_context.task_data.get("next_plan_step_index", 0)

                    if isinstance(current_plan, list) and isinstance(next_step_index, int) and next_step_index < len(current_plan):
                        plan_step = current_plan[next_step_index]
                        if isinstance(plan_step, dict):
                            component_name = plan_step.get("component", "tool_executor")
                            sub_task = plan_step.get("task") or plan_step.get("description") or str(plan_step)
                        else:
                            component_name = "tool_executor"
                            sub_task = str(plan_step)
                        self.logger.info(f"{log_prefix} Executing plan step {next_step_index + 1}/{len(current_plan)}: Target='{component_name}', Task='{sub_task}'")
                        shared_task_context.task_data["next_plan_step_index"] = next_step_index + 1
                    else:
                        self.logger.info(f"{log_prefix} No active plan step. Asking OrchestratorFragment for next delegation.")
                        (
                            component_name,
                            sub_task,
                            delegation_reason,
                        ) = await self._get_next_step_delegation(objective, orchestration_history, shared_task_context)
                        if delegation_reason:
                            self.logger.error(f"{log_prefix} Failed to get delegation from OrchestratorFragment: {delegation_reason}")
                            final_status = STATUS_ERROR; final_reason = delegation_reason; final_answer = f"Orchestration failed: Could not determine next step ({delegation_reason})"
                            await notify_task_error(shared_task_context.task_id, final_reason, {"details": "OrchestratorFragment delegation failed"}); break
                
                # <<< Simplified Execution - component_name and sub_task should be set now >>>
                fragment_success = False
                fragment_result = None
                component_executed = component_name 

                if component_name:
                    shared_task_context.status = f"executing_{component_name}"
                    try:
                        fragment_result_dict = await self._execute_fragment_task(
                            component_name=component_name, sub_task=sub_task,
                            objective=objective, shared_task_context=shared_task_context,
                        )
                        if isinstance(fragment_result_dict, dict):
                            fragment_status = fragment_result_dict.get("status")
                            fragment_success = fragment_status == STATUS_SUCCESS
                            fragment_result = fragment_result_dict
                            if not fragment_success:
                                self.logger.error(f"{log_prefix} FragmentExecutor execution failed for '{component_name}'. Status: {fragment_status}, Reason: {fragment_result_dict.get('reason')}")
                        else:
                            self.logger.error(f"{log_prefix} _execute_fragment_task returned non-dict result for {component_name}: {fragment_result_dict}")
                            fragment_success = False
                            fragment_result = {"status": STATUS_ERROR, "reason": REASON_EXECUTOR_CALL_FAILED, "message": "Internal orchestrator error: _execute_fragment_task invalid return."}
                    except FragmentExecutionError as fee:
                        error_message = str(fee)
                        self.logger.error(f"{log_prefix} FragmentExecutionError for '{component_name}': {error_message} (Status: {fee.status}, Reason: {fee.reason})", exc_info=True)
                        fragment_success = False; fragment_result = {"status": fee.status, "reason": fee.reason, "message": error_message}
                    except Exception as e:
                        self.logger.exception(f"{log_prefix} Unexpected error during fragment execution for '{component_name}':")
                        fragment_success = False; fragment_result = {"status": STATUS_ERROR, "reason": REASON_FRAGMENT_FAILED, "message": f"Unexpected error executing {component_name}: {e}"}
                else:
                    self.logger.warning(f"{log_prefix} No component determined for execution in this step. Skipping execution.")
                    continue 

                # --- Process Fragment Execution Result --- 
                if fragment_result:
                    result_summary = json.dumps(fragment_result)
                    orchestration_history.append((component_executed, result_summary))
                    self.logger.debug(f"{log_prefix} Added to history: ({component_executed}, {result_summary[:200]}...) ")
                    
                    # Update context (e.g., plan)
                    if component_executed == "PlannerFragment" and fragment_success:
                        new_plan = fragment_result.get("plan")
                        if isinstance(new_plan, list):
                            self.logger.info(f"{log_prefix} Received new plan from PlannerFragment. Updating context.")
                            shared_task_context.task_data["current_plan"] = new_plan
                            shared_task_context.task_data["next_plan_step_index"] = 0
                    
                    # Check for task completion
                    if fragment_success and fragment_result.get("task_complete", False):
                        final_status = STATUS_SUCCESS
                        final_answer = fragment_result.get("answer") or fragment_result.get("result") or "Task completed successfully."
                        task_completed_successfully = True
                        self.logger.info(f"{log_prefix} Task marked as complete by '{component_executed}'. Final answer: {final_answer}")
                        shared_task_context.result = final_answer
                        shared_task_context.status = "completed"
                        # Don't break yet, check for action intent below
                    
                    # <<< Check for Action Intent (after processing result) >>>
                    action_intent_obj = shared_task_context.action_intent
                    if action_intent_obj:
                        self.logger.info(f"{log_prefix} Detected Action Intent from '{component_executed}': Skill='{action_intent_obj.skill_target}'")
                        shared_task_context.action_intent = None 
                        action_was_executed = True 
                        
                        action_result = await self.action_executor.execute_intent(intent=action_intent_obj, shared_task_context=shared_task_context)
                        
                        action_summary = json.dumps(action_result)
                        orchestration_history.append((f"ActionExecutor ({action_intent_obj.skill_target})", action_summary))
                        await notify_task_update(shared_task_context.task_id, f"Executed action '{action_intent_obj.skill_target}'. Status: {action_result.get('status')}")
                        self.logger.debug(f"{log_prefix} Action result: {action_summary}")

                        if action_result.get("status") != STATUS_SUCCESS:
                            self.logger.error(f"{log_prefix} Action Intent execution failed: {action_result}")
                            shared_task_context.error_info = {"stage": "action_execution", "details": action_result}
                            final_status = STATUS_ERROR
                            final_reason = action_result.get("reason", REASON_ACTION_EXECUTION_FAILED)
                            final_answer = f"Orchestration failed during action execution: {action_result.get('message')}"
                            task_completed_successfully = False 
                            break 

                # --- Handle Fragment Failure --- 
                if not fragment_success:
                    self.logger.error(f"{log_prefix} Step {current_step} failed due to fragment/executor error.")
                    final_status = fragment_result.get("status", STATUS_ERROR)
                    final_reason = fragment_result.get("reason", REASON_FRAGMENT_FAILED)
                    final_answer = fragment_result.get("message", f"Task failed during execution of {component_executed}.")
                    shared_task_context.error_info = {"stage": f"execution_{component_executed}", "details": fragment_result}
                    shared_task_context.status = "failed"
                    await notify_task_error(shared_task_context.task_id, final_reason, shared_task_context.error_info)
                    break 

                # --- Loop Completion Check --- 
                if task_completed_successfully:
                     self.logger.info(f"{log_prefix} Task completed successfully after step {current_step}.")
                     break 

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
                self.monitor_task.cancel(); await asyncio.sleep(0) # Allow cancellation to propagate
                try: await self.monitor_task
                except asyncio.CancelledError: self.logger.info(f"{log_prefix} Chat monitor task successfully cancelled.")
                except Exception as monitor_cancel_err: self.logger.error(f"{log_prefix} Error waiting for monitor task cancellation: {monitor_cancel_err}")
            else: self.logger.info(f"{log_prefix} Monitor task was already done or not started.")

            # --- Invoke Learning Cycle --- #
            await self._invoke_learning_cycle(objective, orchestration_history, final_status, shared_task_context)

            # --- Notify Task Completion/Error --- #
            if final_status == STATUS_SUCCESS: await notify_task_completion(shared_task_context.task_id, final_answer)
            elif final_status != "stopped_by_user" and shared_task_context.status not in ["failed", "failed_critical", "failed_max_steps"]:
                 await notify_task_error(shared_task_context.task_id, final_reason if 'final_reason' in locals() else REASON_UNKNOWN, {"details": final_answer})

        # --- Return Final Result --- #
        return {
            "status": final_status, "answer": final_answer, "task_id": task_id,
            "history": orchestration_history, "final_context": shared_task_context.to_dict()
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
