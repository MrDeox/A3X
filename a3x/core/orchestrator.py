import asyncio
import logging
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from a3x.core.llm_interface import LLMInterface
from a3x.fragments.registry import FragmentRegistry
from a3x.core.tool_registry import ToolRegistry
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
)

# Type hint for MemoryManager to avoid circular import
from typing import TYPE_CHECKING
from a3x.core.context import FragmentContext

if TYPE_CHECKING:
    from a3x.core.memory.memory_manager import MemoryManager
    from a3x.core.context import SharedTaskContext


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
    ):
        self.fragment_registry = fragment_registry
        self.tool_registry = tool_registry
        self.memory_manager = memory_manager
        self.llm_interface = llm_interface
        self.workspace_root = workspace_root
        self.logger = agent_logger.getChild("Orchestrator")
        self.monitor_task: Optional[asyncio.Task] = None

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
            # Use methods that return IDs/Names instead of full descriptions
            available_fragment_ids = list(self.fragment_registry.get_all_definitions().keys()) # Get list of registered fragment IDs
            available_skill_names = list(self.tool_registry.list_tools().keys()) # Get list of skill names
            task_context_dict = await shared_task_context.get_task_data_copy() # Use the new method

            # Ensure history format is compatible (assuming OrchestratorFragment expects List[Dict])
            # The current history is List[Tuple[str, str]] - adapt if needed
            formatted_history = [{"thought_action": h[0], "observation": h[1]} for h in history]

            # 2. Instantiate and execute the OrchestratorFragment
            # Create a default FragmentDef for this instantiation
            from a3x.fragments.base import FragmentDef # Local import
            orchestrator_fragment_def = FragmentDef(
                name="OrchestratorInternal", # Give it a distinct name
                fragment_class=OrchestratorFragment, 
                description="Internal fragment used by TaskOrchestrator for delegation decisions.",
                category="Orchestration"
                # Skills/ManagedSkills can be empty as it likely doesn't execute tools directly
            )
            orchestrator_fragment = OrchestratorFragment(fragment_def=orchestrator_fragment_def)

            decision_result = await orchestrator_fragment.execute(
                objective=objective,
                history=formatted_history, # Pass formatted history
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
            decision_status = decision_result.get("status", "error") # Assume error if status missing

            if decision_status != STATUS_SUCCESS:
                message = decision_result.get("message", "OrchestratorFragment decision failed.")
                self.logger.error(f"OrchestratorFragment decision failed: {message}")
                return None, None, REASON_DELEGATION_FAILED # Use delegation failed reason

            if not component_name or not sub_task:
                self.logger.error(
                    f"OrchestratorFragment response missing 'component_name' or 'sub_task': {decision_result}"
                )
                return None, None, REASON_DELEGATION_FAILED

            # 4. Validate Component Name (ensure it exists in fragments registry)
            # PlannerFragment might be handled specially later, but should still be in registry
            if component_name not in self.fragment_registry.get_all_definitions():
                self.logger.error(
                    f"OrchestratorFragment delegated to unknown fragment: '{component_name}'"
                )
                return None, None, REASON_DELEGATION_FAILED

            self.logger.info(
                f"[_get_next_step_delegation] OrchestratorFragment decided: Delegate to '{component_name}' for sub-task '{sub_task[:100]}...'"
            )
            return component_name, sub_task, None # Return None for the reason string on success

        except Exception as e:
            self.logger.exception("[_get_next_step_delegation] Error executing/processing OrchestratorFragment:")
            return None, None, REASON_DELEGATION_FAILED # Treat internal fragment error as delegation failure

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
        self.logger.info(
            f"{log_prefix} Executing fragment '{component_name}' for sub-task: '{sub_task[:100]}...'"
        )

        fragment_executor = FragmentExecutor(
            fragment_registry=self.fragment_registry,
            tool_registry=self.tool_registry,
            memory_manager=self.memory_manager,
            logger=self.logger.getChild("FragmentExecutor"),
        )

        try:
            allowed_skills = self._get_allowed_skills(
                component_name, self.tool_registry, self.fragment_registry
            )
            if not allowed_skills and component_name != "PlannerFragment":
                self.logger.error(
                    f"{log_prefix} No allowed skills found or determined for fragment '{component_name}'. Cannot execute."
                )
                return {
                    "status": STATUS_ERROR,
                    "reason": REASON_NO_ALLOWED_SKILLS,
                    "message": f"No skills available/allowed for {component_name}",
                }

            result_dict = await fragment_executor.execute(
                fragment_name=component_name,
                sub_task=sub_task,
                objective=objective,
                shared_context=shared_task_context,
                allowed_skills=allowed_skills,
                fragment_history=current_fragment_history,
            )
            self.logger.info(
                f"{log_prefix} FragmentExecutor returned for '{component_name}': Status - {result_dict.get('status')}"
            )
            self.logger.debug(
                f"{log_prefix} FragmentExecutor result dict: {result_dict}"
            )
            return result_dict

        except FragmentExecutionError as fee:
            # Use str(fee) to get the message passed during initialization
            error_message = str(fee)
            self.logger.error(
                f"{log_prefix} FragmentExecutionError for '{component_name}': {error_message} (Status: {fee.status}, Reason: {fee.reason})"
            )
            fragment_success = False
            fragment_result = {
                "status": fee.status,
                "reason": fee.reason,
                "message": error_message, # Use the extracted message
            }
            return fragment_result

        except Exception as e:
            self.logger.exception(
                f"{log_prefix} Unexpected error calling FragmentExecutor for '{component_name}':"
            )
            fragment_success = False
            fragment_result = {
                "status": STATUS_ERROR,
                "reason": REASON_EXECUTOR_CALL_FAILED,
                "message": f"Orchestrator failed to execute fragment: {e}",
            }
            return fragment_result

    async def _invoke_learning_cycle(
        self,
        objective: str,
        main_history: List,
        final_status: str,
        shared_context: "SharedTaskContext",
    ):
        """Placeholder for invoking the meta-learning cycle."""
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
            # Call the MemoryManager's learning method
            await self.memory_manager.learn_from_task(learning_data)
            self.logger.info(f"{log_prefix} Learning cycle completed successfully.")

        except Exception:
            self.logger.exception(
                f"{log_prefix} Error during learning cycle invocation:"
            )
            # Decide if this error should impact the overall task status or just be logged

    async def orchestrate(
        self, objective: str, max_steps: Optional[int] = None
    ) -> Dict:
        """Main orchestration loop to achieve the objective."""
        # <<< ADD Task ID generation >>>
        import uuid

        task_id = str(uuid.uuid4())
        log_prefix = f"[Orchestrator Task {task_id}]"
        self.logger.info(
            f"{log_prefix} Starting orchestration for objective: '{objective[:100]}...'"
        )

        # --- Initialize Task State --- #
        from a3x.core.context import SharedTaskContext

        shared_task_context = SharedTaskContext(task_id)
        await shared_task_context.update_data("objective", objective)
        await shared_task_context.update_data("status", "starting")

        orchestration_history: List[Tuple[str, str]] = []
        current_step = 0
        final_status = "in_progress"
        final_answer = "Task did not complete."
        task_completed_successfully = False
        # <<< ADD Plan state >>>
        await shared_task_context.update_data("current_plan", None)
        await shared_task_context.update_data("next_plan_step_index", 0)

        # --- Start Chat Monitor --- #
        # <<< ADD Chat Monitor logic >>>
        monitor_queue = asyncio.Queue()
        # Pass the queue and other required dependencies to the monitor task
        self.monitor_task = asyncio.create_task(
            chat_monitor_task(
                task_id=task_id,
                shared_task_context=shared_task_context, # Pass the context instance
                llm_interface=self.llm_interface, # Pass orchestrator's LLM interface
                tool_registry=self.tool_registry,
                fragment_registry=self.fragment_registry,
                memory_manager=self.memory_manager,
                workspace_root=self.workspace_root
            )
        )
        # await shared_task_context.set_monitor_queue(monitor_queue) # Queue is now part of shared_task_context init
        self.logger.info(f"{log_prefix} Chat monitor started.")

        # --- Resolve max_steps --- #
        if max_steps is None:
            # Get from config or use a default
            try:
                from a3x.core.config import ORCHESTRATOR_MAX_STEPS

                max_steps = ORCHESTRATOR_MAX_STEPS
            except ImportError:
                max_steps = 10  # Default if not configured
            self.logger.info(f"{log_prefix} Using max_steps: {max_steps}")

        # --- Orchestration Loop --- #
        try:
            while current_step < max_steps and not task_completed_successfully:
                current_step += 1
                self.logger.info(
                    f"{log_prefix} --- Step {current_step}/{max_steps} --- "
                )
                await shared_task_context.update_data("status", f"step_{current_step}")
                await notify_task_update(
                    shared_task_context.task_id, f"Starting step {current_step}"
                )

                # --- Check Chat Monitor for Interrupts --- #
                try:
                    monitor_message = monitor_queue.get_nowait()
                    if monitor_message == "STOP":
                        self.logger.warning(
                            f"{log_prefix} Received STOP signal from monitor. Halting task."
                        )
                        final_status = "stopped_by_user"
                        final_answer = "Task stopped by user interaction."
                        await notify_task_error(
                            shared_task_context.task_id,
                            "user_interrupt",
                            {"details": final_answer},
                        )
                        break  # Exit loop
                    # Handle other messages if needed
                except asyncio.QueueEmpty:
                    pass  # No message, continue normally

                # --- Decide Next Action: Use Plan or Ask LLM --- #
                current_plan = await shared_task_context.get_data("current_plan")
                next_step_index = await shared_task_context.get_data(
                    "next_plan_step_index"
                )

                component_name = None
                sub_task = None
                delegation_reason = None
                is_direct_executable = False

                if current_plan and next_step_index < len(current_plan):
                    # Execute next step from the plan
                    sub_task = current_plan[next_step_index]
                    # Assume plan steps imply ToolExecutorFragment unless specified otherwise
                    # <<< TODO: Enhance plan format to specify fragment? >>>
                    component_name = (
                        "ToolExecutorFragment"  # Default to executor for plan steps
                    )
                    self.logger.info(
                        f"{log_prefix} Executing plan step {next_step_index + 1}/{len(current_plan)}: '{sub_task}' via {component_name}"
                    )
                    await shared_task_context.update_data(
                        "next_plan_step_index", next_step_index + 1
                    )

                else:
                    # No plan or plan finished, ask LLM for next delegation
                    self.logger.info(
                        f"{log_prefix} No active plan step. Asking LLM for next delegation."
                    )
                    (
                        component_name,
                        sub_task,
                        delegation_reason,
                    ) = await self._get_next_step_delegation(
                        objective, orchestration_history, shared_task_context
                    )

                    if delegation_reason:
                        self.logger.error(
                            f"{log_prefix} Failed to get delegation from LLM: {delegation_reason}"
                        )
                        final_status = STATUS_ERROR
                        final_reason = delegation_reason
                        final_answer = f"Orchestration failed: Could not determine next step ({delegation_reason})"
                        await notify_task_error(
                            shared_task_context.task_id,
                            final_reason,
                            {"details": "LLM delegation failed"},
                        )
                        break  # Exit loop

                    # Check if LLM chose PlannerFragment directly
                    if component_name == "PlannerFragment":
                        is_direct_executable = True
                        self.logger.info(
                            f"{log_prefix} LLM chose PlannerFragment. Will execute directly."
                        )
                        # Planner doesn't need a sub_task in the same way, but pass it along if provided
                        # sub_task might contain context like "regenerate plan due to error X"

                # --- Execute Delegated Task --- #
                fragment_success = False
                fragment_result = None

                try:
                    if is_direct_executable and component_name == "PlannerFragment":
                        # Special handling for directly calling the Planner
                        self.logger.info(
                            f"{log_prefix} Directly executing PlannerFragment."
                        )
                        planner_fragment_def = (
                            self.fragment_registry.get_fragment_definition(
                                "PlannerFragment"
                            )
                        )
                        if not planner_fragment_def:
                            raise FragmentExecutionError(
                                "PlannerFragment definition not found in registry.",
                                status=STATUS_ERROR,
                                reason=REASON_FRAGMENT_NOT_FOUND,
                            )
                        # TODO: This assumes PlannerFragment takes tool_registry in constructor.
                        #       If not, it needs to be passed via context or execute args.
                        planner_instance = planner_fragment_def.fragment_class(
                             # REMOVE: fragment_def=planner_fragment_def # Pass the def
                             # Does it need tool_registry here? Check PlannerFragment.__init__
                             # tool_registry=self.tool_registry 
                        )
                        
                        # Create a FragmentContext for the direct execution
                        planner_fragment_context = FragmentContext(
                            logger=self.logger.getChild(f"PlannerDirect"), # Use a child logger
                            llm_interface=self.llm_interface,
                            tool_registry=self.tool_registry,
                            fragment_registry=self.fragment_registry,
                            shared_task_context=shared_task_context, # Pass the shared context
                            workspace_root=self.workspace_root,
                            memory_manager=self.memory_manager
                        )

                        # Execute the planner's method directly, passing objective and the FragmentContext
                        fragment_result = await planner_instance.execute(
                             objective=objective, 
                             context=planner_fragment_context 
                        )
                        fragment_success = (
                            fragment_result.get("status") == STATUS_SUCCESS
                        )

                    elif component_name:
                        fragment_result_dict = await self._execute_fragment_task(
                            component_name=component_name,
                            sub_task=sub_task,
                            objective=objective,
                            shared_task_context=shared_task_context,
                        )
                        # Determine success based on status from result_dict
                        if isinstance(fragment_result_dict, dict):
                            fragment_status = fragment_result_dict.get("status")
                            fragment_success = fragment_status == STATUS_SUCCESS
                            fragment_result = fragment_result_dict
                            if not fragment_success:
                                self.logger.error(
                                    f"{log_prefix} FragmentExecutor execution failed for '{component_name}'. Result Status: {fragment_status}, Reason: {fragment_result_dict.get('reason')}"
                                )
                        else:
                            # This case indicates an issue with _execute_fragment_task itself returning non-dict
                            self.logger.error(
                                f"{log_prefix} _execute_fragment_task returned non-dict result for {component_name}: {fragment_result_dict}"
                            )
                            fragment_success = False
                            fragment_result = {
                                "status": STATUS_ERROR,
                                "reason": REASON_EXECUTOR_CALL_FAILED,
                                "message": "Internal orchestrator error: _execute_fragment_task did not return a dictionary.",
                            }

                    else:
                        # Should not happen if delegation succeeded
                        self.logger.error(
                            f"{log_prefix} Internal error: component_name was None after delegation check."
                        )
                        raise Exception("Internal logic error in orchestrator loop.")

                except FragmentExecutionError as fee:
                    # Use str(fee) to get the message passed during initialization
                    error_message = str(fee)
                    self.logger.error(
                        f"{log_prefix} FragmentExecutionError for '{component_name}': {error_message} (Status: {fee.status}, Reason: {fee.reason})"
                    )
                    fragment_success = False
                    fragment_result = {
                        "status": fee.status,
                        "reason": fee.reason,
                        "message": error_message, # Use the extracted message
                    }
                except Exception as e:
                    self.logger.exception(
                        f"{log_prefix} Unexpected error during fragment execution for '{component_name}':"
                    )
                    fragment_success = False
                    fragment_result = {
                        "status": STATUS_ERROR,
                        "reason": REASON_ORCHESTRATION_CRITICAL_ERROR,
                        "message": f"Unexpected orchestrator error: {e}",
                    }

                # --- Process Fragment Result ---
                final_answer = None
                plan_generated = False
                request_replan = False
                result_data = None
                status_from_result = STATUS_ERROR
                reason_from_result = REASON_UNKNOWN
                message_from_result = "Invalid or missing fragment result."

                if isinstance(fragment_result, dict):
                    status_from_result = fragment_result.get("status", STATUS_ERROR)
                    reason_from_result = fragment_result.get("reason", REASON_UNKNOWN)
                    message_from_result = fragment_result.get(
                        "message", "No message provided."
                    )

                    # Check for final answer (common key expected from executor or potentially direct fragment)
                    final_answer = fragment_result.get("final_answer")
                    # Check for replan request (standardized status)
                    request_replan = status_from_result == "request_replan"
                    # Extract data payload (might contain plan from planner or history from executor)
                    result_data = fragment_result.get("data")

                    # Specific handling for PlannerFragment direct call result
                    if is_direct_executable and component_name == "PlannerFragment":
                        if fragment_success:
                            if isinstance(result_data, dict) and result_data.get(
                                "plan"
                            ):
                                plan_generated = True
                                self.logger.info(
                                    f"{log_prefix} PlannerFragment returned a plan."
                                )
                            else:
                                self.logger.warning(
                                    f"{log_prefix} PlannerFragment succeeded but result data did not contain a 'plan'. Result: {fragment_result}"
                                )
                        # If planner failed, fragment_success is False, handled by main status check below.

                    if isinstance(result_data, dict):
                        fragment_history = result_data.get("history", [])
                        if fragment_history:
                            self.logger.info(
                                f"{log_prefix} Received {len(fragment_history)} history entries from {component_name} execution."
                            )
                            for entry_index, entry in enumerate(fragment_history):
                                if isinstance(entry, tuple) and len(entry) == 2:
                                    log_message = f"ReAct Step {entry_index + 1}:\nThought/Action:\n{entry[0]}\nObservation:\n{entry[1]}"
                                    await shared_task_context.add_history(
                                        "Orchestrator", log_message
                                    )
                                else:
                                    await shared_task_context.add_history(
                                        "Orchestrator",
                                        f"History Entry {entry_index + 1} (raw from {component_name}): {str(entry)[:500]}",
                                    )

                # --- Update Orchestration History ---
                orchestrator_observation = f"Fragment '{component_name}' executed. Status: {status_from_result}."
                if message_from_result and status_from_result != STATUS_SUCCESS:
                    orchestrator_observation += f" Message: {message_from_result[:150]}{'...' if len(message_from_result)>150 else ''}"
                if plan_generated:
                    orchestrator_observation += " Plan generated."
                if request_replan:
                    orchestrator_observation += " Replanning requested."
                if final_answer:
                    orchestrator_observation += f" Final Answer provided: {final_answer[:100]}{'...' if len(final_answer)>100 else ''}"

                orchestrator_thought = f"Completed execution delegation to '{component_name}' for sub-task '{sub_task}'."
                orchestration_history.append(
                    (orchestrator_thought, orchestrator_observation)
                )
                self.logger.info(
                    f"{log_prefix} Updated orchestration history: {orchestrator_thought} -> {orchestrator_observation}"
                )
                await shared_task_context.add_history(
                    "Orchestrator", orchestrator_observation
                )

                # --- Check for Plan Generation (if Planner was called) ---
                if plan_generated:
                    # Plan was generated, store it and continue loop (will execute plan next)
                    validated_plan = result_data["plan"]
                    await shared_task_context.update_data(
                        "current_plan", validated_plan
                    )
                    await shared_task_context.update_data("next_plan_step_index", 0)
                    self.logger.info(
                        f"{log_prefix} New plan with {len(validated_plan)} steps stored from PlannerFragment. Continuing orchestration loop."
                    )
                    continue

                # --- Check for Replan Request ---
                if request_replan:
                    self.logger.warning(
                        f"{log_prefix} Fragment '{component_name}' requested replanning. Clearing current plan and continuing loop."
                    )
                    await shared_task_context.update_data("current_plan", None)
                    await shared_task_context.update_data("next_plan_step_index", 0)
                    await shared_task_context.add_history(
                        "Orchestrator",
                        f"Replanning requested by {component_name}. Reason: {message_from_result}",
                    )
                    continue

                # --- Check for Task Completion (Final Answer) ---
                if fragment_success and final_answer:
                    self.logger.info(
                        f"{log_prefix} Fragment {component_name} provided a final answer. Concluding task."
                    )
                    final_status = STATUS_SUCCESS
                    await notify_task_completion(
                        shared_task_context.task_id, final_answer
                    )
                    task_completed_successfully = True
                    break

                # --- Check for Fragment Failure ---
                if not fragment_success:
                    self.logger.error(
                        f"{log_prefix} Fragment '{component_name}' failed (Status: {status_from_result}). Stopping task."
                    )
                    final_status = STATUS_ERROR
                    final_reason = f"{REASON_FRAGMENT_FAILED}:{component_name}:{reason_from_result}"
                    final_answer = f"Task failed because fragment '{component_name}' encountered an error: {message_from_result}"
                    await notify_task_error(
                        task_id,
                        final_reason,
                        {"fragment": component_name, "details": message_from_result},
                    )
                    break

            # --- End of Orchestration Loop ---

            # Final status determination if loop finishes without break
            if final_status == "in_progress":
                self.logger.warning(
                    f"{log_prefix} Task reached max steps ({current_step}) without completion."
                )
                final_status = STATUS_MAX_ITERATIONS
                final_reason = REASON_MAX_STEPS_REACHED
                final_answer = (
                    "Task stopped after reaching the maximum number of steps."
                )
                await notify_task_error(
                    shared_task_context.task_id, final_reason, {"details": final_answer}
                )

        except Exception as e:
            self.logger.exception(
                "[Orchestrator] Critical error during orchestration loop:"
            )
            final_status = STATUS_ERROR
            final_reason = REASON_ORCHESTRATION_CRITICAL_ERROR
            final_answer = f"Orchestration failed due to critical error: {e}"
            # Ensure notification even on critical error
            if "shared_task_context" in locals():
                # Pass a dictionary with the error message instead of just the string representation
                error_details = {"error_type": type(e).__name__, "error_message": str(e)}
                await notify_task_error(
                    shared_task_context.task_id, final_reason, error_details
                )
            # Ensure history is preserved up to the error point
            if "orchestration_history" not in locals():
                orchestration_history = []

        # --- Learning Cycle Invocation (Optional) ---
        # await self._invoke_learning_cycle(objective, orchestration_history, final_status, shared_task_context)

        # --- Chat Monitor Cleanup ---
        self.logger.info(f"{log_prefix} Cleaning up Chat Monitor task...")
        if (
            hasattr(self, "monitor_task")
            and self.monitor_task
            and not self.monitor_task.done()
        ):
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                self.logger.info(f"{log_prefix} Chat Monitor task cancelled.")
            except Exception as mon_clean_err:
                self.logger.error(
                    f"{log_prefix} Error during chat monitor cleanup: {mon_clean_err}"
                )
            else:
                self.logger.info(
                    f"{log_prefix} Chat Monitor task already done or not started."
                )

        # --- Return final result ---
        return {
            "status": final_status,
            "final_answer": final_answer,
            "history": orchestration_history,
            "task_id": (
                shared_task_context.task_id
                if "shared_task_context" in locals()
                else None
            ),
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
        # If a fragment manages skills, only allow those. Otherwise, allow all.
        fragment_def = fragment_registry.get_fragment_definition(fragment_name)
        if fragment_def and fragment_def.managed_skills:
            self.logger.debug(
                f"Fragment '{fragment_name}' manages specific skills: {fragment_def.managed_skills}"
            )
            # Verify managed skills exist in the tool registry
            allowed_skills = []
            all_registered_tools = tool_registry.list_tools().keys()
            for skill_name in fragment_def.managed_skills:
                if skill_name in all_registered_tools:
                    allowed_skills.append(skill_name)
                else:
                    self.logger.warning(
                        f"Managed skill '{skill_name}' for fragment '{fragment_name}' not found in ToolRegistry."
                    )
            if not allowed_skills:
                self.logger.warning(
                    f"Fragment '{fragment_name}' defined managed skills, but none were found in registry. Allowing all skills as fallback."
                )
                allowed_skills = list(all_registered_tools)
        else:
            # Allow all registered skills if none are specifically managed or def not found
            allowed_skills = list(tool_registry.list_tools().keys())
            self.logger.debug(
                f"Fragment '{fragment_name}' does not manage specific skills or def not found. Allowing all {len(allowed_skills)} registered skills."
            )

        return allowed_skills
