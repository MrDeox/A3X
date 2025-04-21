import asyncio
import logging
from typing import Dict, Any, Optional

# Use try-except for relative imports if needed, ensure core components are accessible
try:
    from ..core.context import FragmentContext
    from ..core.tool_registry import ToolRegistry
    from .base import BaseFragment, FragmentDef
except ImportError:
    # Fallback for potential execution context issues
    from a3x.core.context import FragmentContext
    from a3x.core.tool_registry import ToolRegistry
    from a3x.fragments.base import BaseFragment, FragmentDef

# Configuration (adjust as needed)
REASSESSMENT_THRESHOLD = 1 # Number of identical failures before reassessing
RETRY_LIMIT = 1 # Maximum number of retries for a single failure
FAILURE_MEMORY_WINDOW = 600 # Seconds to remember failures for threshold calculation

class CoordinatorFragment(BaseFragment):
    """
    Listens for failed status messages via an internal loop and coordinates
    responses based on the number of failures for a specific subtask.
    It tracks failure counts in the ContextStore and logs confirmations.
    """
    def __init__(self, fragment_def: FragmentDef, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(fragment_def, tool_registry)
        self._logger = logging.getLogger(f"A3X.{self.get_name()}")
        self._message_queue = asyncio.Queue()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._context: Optional[FragmentContext] = None # Store context for the loop

    def set_context(self, context: FragmentContext):
        """Sets the context and starts the monitoring loop."""
        super().set_context(context)
        self._context = context # Store context for the loop
        # self.start() # <<< TEMPORARILY COMMENTED OUT TO PREVENT LOOP START >>>
        self._logger.warning("Coordinator monitoring loop start TEMPORARILY DISABLED for debugging.")

    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """Handles incoming messages to monitor task failures."""
        # <<< ADD DEBUG LOGGING >>>
        msg_type_log = message.get("type", "NO_TYPE")
        sender_log = message.get("sender", "NO_SENDER")
        content_log = message.get("content", {})
        status_log = content_log.get("status", "N/A") if isinstance(content_log, dict) else "N/A"

        self._logger.debug(f"[Coordinator] Received message: Type='{msg_type_log}', Sender='{sender_log}', Status='{status_log}'")

        # Specifically check for results from Executor
        if sender_log == "Executor" and isinstance(msg_type_log, str):
            msg_type_lower = msg_type_log.lower()
            if msg_type_lower in ['skill_execution_result', 'plan_execution_result'] and status_log == 'error':
                self._logger.info(f"[Coordinator] *** Received ERROR result from Executor: Type='{msg_type_log}', Status='{status_log}' ***")
                plan_id = content_log.get("plan_id", "unknown") if msg_type_lower == 'plan_execution_result' else "N/A"
                original_action = content_log.get("original_action", "unknown") if msg_type_lower == 'skill_execution_result' else "N/A"
                self._logger.info(f"[Coordinator] Details: PlanID='{plan_id}', FailedAction='{original_action}'")
                # Log that we *should* be processing this failure
                self._logger.info(f"[Coordinator] Processing failure event...")
        # <<< END DEBUG LOGGING >>>

        # Process message for failure monitoring - PUT RELEVANT MESSAGES ON QUEUE
        # Check if the message indicates a failure that should be tracked
        FAILURE_MESSAGE_TYPES = [
            "REFACTOR_RESULT", 
            "MUTATION_ATTEMPT", 
            "SKILL_EXECUTION_RESULT", 
            "PLAN_EXECUTION_RESULT",
            "ARCHITECT_RESULT",
            "MANAGER_RESULT",
            # Add other types that indicate failure
        ]
        if isinstance(msg_type_log, str) and msg_type_log.upper() in FAILURE_MESSAGE_TYPES and status_log == "error":
             self._logger.debug(f"[Coordinator] Queuing error message from {sender_log} for processing by monitoring loop.")
             await self._message_queue.put(message)
        # NOTE: We might also want to queue confirmations like reassess_success, retry_success etc.
        # if needed by the monitoring loop's logic (currently it only checks status in the loop)

    async def _monitoring_loop(self):
        """Continuously processes messages from the internal queue."""
        if not self._context:
            self._logger.error("Cannot start monitoring loop: context not set.")
            return

        self._logger.info("Coordinator monitoring loop started.")
        while not self._stop_event.is_set():
            try:
                # Wait for a message with a timeout to allow checking stop_event
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                if not message: # Should not happen with Queue but good practice
                    continue

                content = message.get("content", {})
                sender = message.get("sender")
                status = content.get("status")
                subtask_id = content.get("subtask_id")
                responsible_fragment = content.get("responsible_fragment", sender)

                self._logger.debug(f"Processing message from queue: Sender='{sender}', Status='{status}', Subtask='{subtask_id}'")

                if status == "failed":
                    if not subtask_id:
                        self._logger.warning(f"Received failed status message from '{sender}' without 'subtask_id'. Cannot track failure count.")
                        continue
                    if not responsible_fragment:
                         self._logger.warning(f"Received failed status message for subtask '{subtask_id}' without 'responsible_fragment' or 'sender'. Cannot target directive.")
                         continue

                    await self._handle_failure(subtask_id, responsible_fragment)

                elif status in ["reassess_success", "retry_success", "abort_acknowledged"]:
                    if not subtask_id:
                         self._logger.warning(f"Received confirmation status '{status}' from '{sender}' without 'subtask_id'.")
                         continue
                    # Log confirmation
                    self._logger.info(f"Received confirmation '{status}' from '{sender}' for subtask '{subtask_id}'.")
                    # Optional: Reset failure count on success (e.g., retry_success)
                    # if status == "retry_success":
                    #    context_key = f"task_failures:{subtask_id}"
                    #    try:
                    #        await self._context.store.set(context_key, 0)
                    #        self._logger.info(f"Reset failure count for subtask '{subtask_id}' after successful retry.")
                    #    except Exception as e:
                    #        self._logger.error(f"Failed to reset failure count for '{subtask_id}' in ContextStore: {e}")


                self._message_queue.task_done()

            except asyncio.TimeoutError:
                # No message received, loop continues to check stop_event
                continue
            except asyncio.CancelledError:
                self._logger.info("Monitoring loop cancelled.")
                break
            except Exception as e:
                self._logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                # Avoid tight loop on persistent error
                await asyncio.sleep(1)

        self._logger.info("Coordinator monitoring loop finished.")

    async def _handle_failure(self, subtask_id: str, responsible_fragment: str):
        """Handles the logic for processing a single failure event."""
        if not self._context: return # Should not happen if loop is running

        context_key = f"task_failures:{subtask_id}"
        failure_count = 0
        try:
            stored_value = await self._context.store.get(context_key)
            if isinstance(stored_value, int):
                failure_count = stored_value
            elif stored_value is not None:
                 self._logger.warning(f"Invalid value type in ContextStore for key '{context_key}'. Expected int, got {type(stored_value)}. Resetting count.")
        except Exception as e:
            self._logger.error(f"Failed to get failure count for '{context_key}' from ContextStore: {e}", exc_info=True)

        failure_count += 1
        self._logger.info(f"Failure {failure_count} recorded for subtask '{subtask_id}' by '{responsible_fragment}'.")

        action: Optional[str] = None
        if failure_count == 1:
            action = "reassess"
        elif failure_count == 2:
            action = "retry"
        else: # 3 or more
            action = "abort"

        if action:
            directive_content = {
                "type": "directive",
                "action": action,
                "target": responsible_fragment, # Target the specific fragment
                "reason": f"{failure_count} failure(s) recorded for subtask {subtask_id}",
                "subtask_id": subtask_id
            }
            try:
                 await self.post_chat_message(
                     message_type="directive",
                     content=directive_content,
                     target_fragment=responsible_fragment # Send directly to the target
                 )
                 self._logger.info(f"Sent '{action}' directive targeting '{responsible_fragment}' for subtask '{subtask_id}'.")
            except Exception as e:
                 self._logger.error(f"Failed to send directive for subtask '{subtask_id}': {e}", exc_info=True)

        try:
            await self._context.store.set(context_key, failure_count)
            self._logger.debug(f"Updated failure count for '{context_key}' to {failure_count} in ContextStore.")
        except Exception as e:
            self._logger.error(f"Failed to set failure count for '{context_key}' in ContextStore: {e}", exc_info=True)


    def start(self):
        """Starts the monitoring loop if not already running."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._logger.info("Starting coordinator monitoring task.")
            self._stop_event.clear()
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        else:
            self._logger.warning("Monitoring task already running.")

    def stop(self):
        """Signals the monitoring loop to stop."""
        self._logger.info("Stopping coordinator monitoring task.")
        self._stop_event.set()
        # Optionally, wait for the task to finish if needed,
        # but usually cancellation or stop event is sufficient.
        # if self._monitoring_task:
        #     self._monitoring_task.cancel() # Or rely on stop_event

    # Ensure loop stops if fragment is destroyed (if applicable in your framework)
    # def __del__(self):
    #     self.stop()

    def get_purpose(self) -> str:
        return self._fragment_def.description

    def get_status(self) -> Dict[str, Any]:
        task_running = self._monitoring_task and not self._monitoring_task.done()
        return {
            "name": self.get_name(),
            "status": "active" if task_running else "inactive",
            "monitoring_failures": task_running,
            "queue_size": self._message_queue.qsize() if hasattr(self._message_queue, 'qsize') else 'unknown'
        }

# Example FragmentDef (adjust as needed)
CoordinatorFragmentDef = FragmentDef(
    name="Coordinator",
    description="Monitors task failures via internal loop and sends coordinating directives (reassess, retry, abort).",
    # tools=[], # REMOVED - FragmentDef does not expect 'tools' or 'tools_enabled'
    # category="Coordination", # Optional: Set a category if desired
    # skills=[], # Optional: List any specific skills if needed by this fragment
    fragment_class=CoordinatorFragment
) 