import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List, Set, Tuple

from ..core.context import FragmentContext
from ..core.tool_registry import ToolRegistry
from .base import BaseFragment
from .base import FragmentDef

# Timeout in seconds to wait for a response after sending a directive
DIRECTIVE_TIMEOUT = 120.0

class ExecutorFragment(BaseFragment):
    """
    Listens for 'plan_sequence' messages and executes the contained actions sequentially.
    Each action is sent as an 'architecture_suggestion' directive.
    It waits for a result (e.g., 'refactor_result', 'mutation_result') for each action
    and posts a summary ('plan_execution_result') at the end.
    """
    def __init__(self, fragment_def: FragmentDef, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(fragment_def, tool_registry)
        self._logger = logging.getLogger(f"A3X.{self.get_name()}")
        self._current_directive_event = asyncio.Event()
        self._current_expected_directive: Optional[Dict[str, Any]] = None
        self._last_result_status: Optional[str] = None
        self._last_result_sender: Optional[str] = None
        self._is_executing_plan = False # Simple lock to prevent concurrent plan executions

    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """Handles incoming chat messages, routing plan sequences and results."""
        # <<< ADD HIGH-VISIBILITY WARNING LOG >>>
        self._logger.warning(f"[Executor ENTRY] Received message: Type='{message.get('type', 'NO_TYPE')}', Sender='{message.get('sender', 'NO_SENDER')}'")
        # <<< END HIGH-VISIBILITY WARNING LOG >>>
        
        # <<< ADD DEBUG LOGGING >>>
        msg_type_log = message.get("type", "NO_TYPE")
        sender_log = message.get("sender", "NO_SENDER")
        self._logger.debug(f"[Executor] Received message: Type='{msg_type_log}', Sender='{sender_log}'")
        if isinstance(msg_type_log, str) and msg_type_log.lower() == 'plan_sequence':
            self._logger.info(f"[Executor] *** Received PLAN_SEQUENCE message from {sender_log} ***")
            plan_id = message.get("content", {}).get("plan_id", "unknown")
            self._logger.debug(f"[Executor] Plan ID: {plan_id}")
        # <<< END DEBUG LOGGING >>>

        message_type = message.get("type")
        sender = message.get("sender")

        # Ignore messages sent by self to avoid loops, unless it's a plan sequence we need to process
        if sender == self.get_name() and message_type != 'plan_sequence':
             # self._logger.debug(f"Ignoring message from self: type={message_type}")
             return

        # Convert message_type to lowercase
        message_type_lower = message_type.lower() if isinstance(message_type, str) else None

        if message_type_lower == 'plan_sequence':
            if self._is_executing_plan:
                self._logger.warning("Received new plan_sequence while already executing another. Ignoring.")
                # Optionally, could queue plans or send a busy message back
                await self.post_chat_message(
                    context=context,
                    message_type="plan_execution_result",
                    content={
                        "status": "ignored",
                        "reason": "ExecutorFragment is already busy executing another plan.",
                        "original_plan_id": message.get("content", {}).get("plan_id", "unknown")
                    },
                    sender=self.get_name()
                )
                return
            # Start executing the plan
            asyncio.create_task(self._handle_plan_sequence(message, context))
            self._logger.info("Created task to handle plan sequence.")

        # Listen for results potentially related to the currently executed directive
        elif message_type_lower in ['refactor_result', 'mutation_result', 'skill_execution_result', 'file_manager_result']:
             await self._handle_result_message(message, context)
        # else:
            # self._logger.debug(f"Ignoring message type: {message_type} from sender: {sender}")

    async def _handle_plan_sequence(self, message: Dict[str, Any], context: FragmentContext):
        """Handles the sequential execution of actions in a plan_sequence message."""
        if self._is_executing_plan:
             self._logger.error("Concurrency Error: _handle_plan_sequence called while already executing.")
             return

        self._is_executing_plan = True
        content = message.get("content", {})
        actions: List[Dict[str, Any]] = content.get("actions", [])
        plan_id: str = content.get("plan_id", "unknown")
        self._logger.info(f"Starting execution of plan '{plan_id}' with {len(actions)} actions.")

        success_count = 0
        failure_count = 0
        timeout_count = 0
        involved_fragments: Set[str] = set()
        step_results = []

        start_time = time.monotonic()

        for i, action in enumerate(actions):
            step_start_time = time.monotonic()
            self._logger.info(f"Plan '{plan_id}' - Step {i+1}/{len(actions)}: Executing action: {action.get('action', 'N/A')} for target: {action.get('target', 'N/A')}")

            # Prepare for waiting
            self._current_expected_directive = action
            self._current_directive_event.clear()
            self._last_result_status = None
            self._last_result_sender = None
            step_status = "timeout" # Default status if no response
            step_detail = f"Timeout after {DIRECTIVE_TIMEOUT}s waiting for result."
            responding_fragment = None

            try:
                # Send the action as an architecture_suggestion directive
                await self.post_chat_message(
                    message_type='architecture_suggestion',
                    content=action,
                    # target_fragment=None # Broadcast by default
                )
                self._logger.debug(f"Plan '{plan_id}' - Step {i+1}: Directive sent. Waiting for result...")

                # Wait for the result message to arrive and set the event
                await asyncio.wait_for(self._current_directive_event.wait(), timeout=DIRECTIVE_TIMEOUT)

                # Event was set, result received
                step_status = self._last_result_status or "error" # If status is None somehow, count as error
                step_detail = f"Received result: status='{step_status}'"
                responding_fragment = self._last_result_sender
                self._logger.info(f"Plan '{plan_id}' - Step {i+1}: Received result: status='{step_status}' from '{responding_fragment}'")

            except asyncio.TimeoutError:
                self._logger.warning(f"Plan '{plan_id}' - Step {i+1}: Timed out after {DIRECTIVE_TIMEOUT}s waiting for result for action: {action}")
                timeout_count += 1

            except Exception as e:
                self._logger.error(f"Plan '{plan_id}' - Step {i+1}: Unexpected error executing action: {e}", exc_info=True)
                step_status = "error"
                step_detail = f"Executor error: {str(e)}"

            finally:
                # Record step outcome regardless of success/failure/timeout
                if step_status == "success":
                    success_count += 1
                else: # Any non-success (error, timeout, other failure status) counts as failure
                    failure_count += 1

                if responding_fragment and responding_fragment != self.get_name():
                    involved_fragments.add(responding_fragment)

                step_duration = time.monotonic() - step_start_time
                step_results.append({
                    "step": i + 1,
                    "action_details": action, # Include the original action for context
                    "status": step_status,
                    "responding_fragment": responding_fragment,
                    "details": step_detail,
                    "duration_seconds": round(step_duration, 2)
                })

                # Reset state for the next step (or cleanup)
                self._current_expected_directive = None
                self._last_result_status = None
                self._last_result_sender = None
                # Event is already cleared at the start of the next loop or handled here

        total_duration = time.monotonic() - start_time
        self._logger.info(f"Plan '{plan_id}' execution finished in {total_duration:.2f}s. Success: {success_count}, Failure: {failure_count} (including {timeout_count} timeouts)")

        # Post the final summary
        await self._post_summary(plan_id, success_count, failure_count, timeout_count, list(involved_fragments), step_results, total_duration)

        self._is_executing_plan = False # Release lock

    async def _handle_result_message(self, message: Dict[str, Any], context: FragmentContext):
        """Checks if an incoming result message matches the currently awaited directive.
        Uses normalized JSON comparison for robustness.
        """
        if not self._is_executing_plan or self._current_expected_directive is None:
            return

        content = message.get("content", {})
        received_original_directive = content.get("original_directive")
        sender = message.get("sender")

        self._logger.info(f"[Result Handler Entered] Received potential result from '{sender}'. Type: '{message.get('type')}'. Checking against expected action: '{self._current_expected_directive.get('action', 'N/A') if self._current_expected_directive else 'None'}")

        if not isinstance(received_original_directive, dict):
            self._logger.debug(f"[Result Handler] Discarding message from '{sender}' because original_directive is not a dict or is missing.")
            return

        # Compare normalized JSON strings for robustness
        try:
            self._logger.debug(f"[Result Handler] Checking result from '{sender}'.")
            self._logger.debug(f"[Result Handler] Expected Directive: {self._current_expected_directive}")
            self._logger.debug(f"[Result Handler] Received Original Directive: {received_original_directive}")
            
            expected_json = json.dumps(self._current_expected_directive, sort_keys=True)
            received_json = json.dumps(received_original_directive, sort_keys=True)

            if expected_json == received_json:
                self._logger.debug(f"Received matching result (JSON compare) from '{sender}' for directive: {self._current_expected_directive.get('action', 'N/A')}")
                self._last_result_status = content.get("status", "error")
                self._last_result_sender = sender
                self._current_directive_event.set() # Signal the waiting task
            else:
                # Optional: Log mismatch details for debugging
                self._logger.debug(f"[Result Handler] JSON comparison failed. Sender: '{sender}'.") # Log mismatch
                # self._logger.debug(f"Expected JSON: {expected_json}")
                # self._logger.debug(f"Received JSON: {received_json}")
        except TypeError as e:
            # Log error if objects are not JSON serializable (shouldn't happen with basic dicts)
            self._logger.error(f"Failed to compare directives using JSON: {e}. Expected: {self._current_expected_directive}, Received: {received_original_directive}")
        except Exception as e:
            self._logger.error(f"Unexpected error during JSON comparison: {e}", exc_info=True)

    async def _post_summary(self, plan_id: str, successes: int, failures: int, timeouts: int, fragments: List[str], step_results: List[Dict], duration: float):
        """Posts the final plan execution summary to the chat."""
        summary_content = {
            "type": "plan_execution_result",
            "plan_id": plan_id,
            "status": "completed",
            "total_steps": len(step_results),
            "successful_steps": successes,
            "failed_steps": failures, # Includes timeouts
            "timeout_steps": timeouts,
            "involved_fragments": fragments,
            "total_duration_seconds": round(duration, 2),
            "step_results": step_results
        }
        try:
            await self.post_chat_message(
                message_type="plan_execution_result",
                content=summary_content
            )
            self._logger.info(f"Posted execution summary for plan '{plan_id}'.")
        except Exception as e:
            self._logger.error(f"Failed to post plan execution summary: {e}", exc_info=True)

    def get_purpose(self) -> str:
        """Returns the fragment's defined purpose."""
        # Assuming FragmentDef stores the description which serves as the purpose
        return self._fragment_def.description

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the fragment."""
        current_step_target = None
        current_step_action = None
        if isinstance(self._current_expected_directive, dict): # Check if it's a dict before accessing
            current_step_target = self._current_expected_directive.get("target")
            current_step_action = self._current_expected_directive.get("action")

        return {
            "name": self.get_name(),
            "is_executing_plan": self._is_executing_plan,
            "current_step_target": current_step_target,
            "current_step_action": current_step_action,
        } 