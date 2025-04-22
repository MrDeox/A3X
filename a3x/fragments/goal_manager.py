import asyncio
import logging
import random
from typing import Dict, Any, Optional, List
import uuid
from collections import defaultdict
import time

from .base import BaseFragment, FragmentDef
from a3x.core.context import FragmentContext

logger = logging.getLogger(__name__)

# --- Configuration ---
GOAL_INJECTION_INTERVAL_SECONDS = 20
# Updated path for generated goal targets (now in data/runtime)
DIRECTIVE_TARGET_PREFIX = "data/runtime/generated/goals" # Directory for generated targets

# --- Predefined Goal Templates ---
GOAL_TEMPLATES = [
    {
        "action": "create_helper_module",
        "target_template": f"{DIRECTIVE_TARGET_PREFIX}/helpers/helper_{{}}.py",
        "message_template": "Create a utility function 'process_data(data: dict) -> str' in a new helper module."
    },
    {
        "action": "refactor_module",
        "target_template": f"{DIRECTIVE_TARGET_PREFIX}/refactor_targets/module_{{}}.py", # Assumes target exists
        "message_template": "Refactor the module '{target}' to improve readability and add docstrings."
    },
    {
        "action": "create_test_for_module",
        "target_template": f"{DIRECTIVE_TARGET_PREFIX}/helpers/helper_{{}}.py", # Test target likely relates to an existing module
        "message_template": "Create unit tests for the module '{target}' ensuring basic functionality coverage."
    },
    {
        "action": "create_helper_module",
        "target_template": f"{DIRECTIVE_TARGET_PREFIX}/utils/parser_{{}}.py",
        "message_template": "Generate a new module with a function 'parse_json(json_string: str) -> dict'."
    },
]

class GoalManagerFragment(BaseFragment):
    """
    Injects new goals (architecture suggestions/directives) into the system periodically
    to drive autonomous behavior, learning from past results.
    """
    def __init__(self, fragment_def: FragmentDef, tool_registry=None):
        super().__init__(fragment_def, tool_registry)
        self._goal_loop_task: Optional[asyncio.Task] = None
        self._fragment_context: Optional[FragmentContext] = None
        self.goal_stats: defaultdict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "success": 0,
                "failure": 0, # Includes error, failed, timeout
                "skipped": 0,
                "proposed": 0, # From mutation_attempt
                "total_attempts": 0,
                "last_attempt_time": 0.0,
                "last_status": "unknown",
                # Could add fields like 'associated_rewards', 'anomaly_reported'
            }
        )
        self._logger.info(f"[{self.get_name()}] Initialized with goal statistics tracking.")

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        """Returns a description of this fragment's purpose."""
        return f"Periodically injects new architectural goals (directives) into the system every {GOAL_INJECTION_INTERVAL_SECONDS} seconds to stimulate activity."

    def set_context(self, context: FragmentContext):
        """Receives the context needed for the Goal Manager."""
        # The context passed from the runner loop IS the SharedTaskContext
        shared_context = context 
        super().set_context(shared_context) # Call parent's set_context with the SharedTaskContext
        self._fragment_context = shared_context # Store the shared context if needed locally
        self._logger.info(f"[{self.get_name()}] Context received, ready to start goal loop.")
        # if not self._goal_loop_task or self._goal_loop_task.done(): # <<< COMMENT OUT START >>>
        #    self._start_goal_loop() # Start loop only if context is now available
        # else:
        #    self._logger.warning(f"[{self.get_name()}] Context set again, but goal loop seems already running.")

    def _start_goal_loop(self):
        """Starts the background task for injecting goals if not already running."""
        if self._goal_loop_task is None or self._goal_loop_task.done():
            if self._fragment_context: # Ensure context is set before starting
                self._logger.info(f"[{self.get_name()}] Starting goal injection loop...")
                self._goal_loop_task = asyncio.create_task(self._goal_injection_loop())
                self._goal_loop_task.add_done_callback(self._handle_loop_completion)
            else:
                self._logger.warning(f"[{self.get_name()}] Cannot start goal loop: FragmentContext not set.")
        else:
             self._logger.info(f"[{self.get_name()}] Goal injection loop already running.")

    def _handle_loop_completion(self, task: asyncio.Task):
        """Callback function to handle the completion of the goal loop task."""
        try:
            # Check if the task finished with an exception
            exception = task.exception()
            if exception:
                self._logger.error(f"[{self.get_name()}] Goal injection loop task failed:", exc_info=exception)
            else:
                self._logger.info(f"[{self.get_name()}] Goal injection loop task completed.")
        except asyncio.CancelledError:
            self._logger.info(f"[{self.get_name()}] Goal injection loop task was cancelled.")
        # Reset task reference
        self._goal_loop_task = None

    async def _goal_injection_loop(self):
        """The main loop that periodically generates and posts goals."""
        if not self._fragment_context:
             self._logger.error(f"[{self.get_name()}] Cannot run goal loop: FragmentContext missing.")
             return # Stop if context isn't set

        self._logger.info(f"[{self.get_name()}] Goal injection loop started. Interval: {GOAL_INJECTION_INTERVAL_SECONDS}s")
        await asyncio.sleep(5) # Initial delay before first goal

        while True:
            try:
                # --- TODO: Add logic here to select goal based on self.goal_stats --- 
                # Example: Prioritize novel targets or retry skipped ones? Avoid recently failed?
                # For now, keep random selection
                template = random.choice(GOAL_TEMPLATES)
                unique_id = str(uuid.uuid4())[:8]
                target_path = template["target_template"].format(unique_id)
                # --- End Goal Selection Logic --- #

                # Check if this specific target has failed too often recently (simplistic check)
                # A more robust check would look at the template or action type
                stats = self.goal_stats[target_path] # Get stats for this potential target
                if stats["failure"] > 2 and (time.time() - stats.get("last_attempt_time", 0) < 300): # Avoid if >2 failures in last 5 mins
                    self._logger.info(f"[{self.get_name()}] Skipping goal generation for '{target_path}' due to recent failures ({stats['failure']}).")
                    await asyncio.sleep(GOAL_INJECTION_INTERVAL_SECONDS / 2) # Shorter wait before trying next
                    continue # Try generating a different goal
                    
                # 2. Format the message
                message_text = template["message_template"].format(target=target_path)
                
                # 3. Construct the directive content
                directive_content = {
                    "type": "directive",
                    "action": template["action"],
                    "target": target_path,
                    "message": message_text
                }
                
                # 4. Construct the full message
                goal_message = {
                    "type": "architecture_suggestion",
                    "content": directive_content,
                    # Sender is implicitly handled by post_chat_message using context
                }

                # 5. Post the message using the stored context
                self._logger.info(f"[{self.get_name()}] Injecting new goal: Action='{template['action']}', Target='{target_path}'")
                await self.post_chat_message(
                     message_type="architecture_suggestion", 
                     content=directive_content
                 )

                # 6. Wait for the next interval
                await asyncio.sleep(GOAL_INJECTION_INTERVAL_SECONDS)

            except asyncio.CancelledError:
                self._logger.info(f"[{self.get_name()}] Goal injection loop stopping due to cancellation.")
                break
            except Exception as e:
                self._logger.exception(f"[{self.get_name()}] Error in goal injection loop:")
                # Avoid busy-looping on persistent errors
                await asyncio.sleep(GOAL_INJECTION_INTERVAL_SECONDS * 2)

    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """
        Handles incoming chat messages.
        Ensures context is set for the goal loop.
        Observes results of actions to update goal statistics.
        """
        if self._fragment_context is None:
             self._logger.warning(f"[{self.get_name()}] Context received via handle_realtime_chat. Ensuring goal loop starts.")
             self.set_context(context)
        
        msg_type = message.get("type")
        content = message.get("content", {})
        sender = message.get("sender", "Unknown")
        status = content.get("status", "unknown").lower() # Normalize status
        target_path = None
        action = "unknown"

        # Extract target path and action based on message type
        if msg_type == "REFACTOR_RESULT":
            target_path = content.get("target")
            action = content.get("original_action", "unknown")
        elif msg_type == "MUTATION_ATTEMPT":
            target_path = content.get("file_path")
            action = "mutation" # Assign specific action type
        elif msg_type == "ANOMALY":
             target_path = content.get("file_path") # Anomalies might relate to a file path
             action = content.get("issue", "unknown_anomaly") # Use issue type as action indicator
             status = "anomaly_reported" # Use a distinct status
        # We don't directly update stats based on rewards, but log them
        elif msg_type == "REWARD":
            target_fragment = content.get("target")
            amount = content.get("amount", 0)
            reason = content.get("reason", "")
            self._logger.info(f"[{self.get_name()} OBSERVED] Result='{msg_type}', Rewarded='{target_fragment}', Amount='{amount}', Sender='{sender}', Reason='{reason[:50]}...'")
            return # No further stat update for rewards here
        else:
            return # Ignore other message types for stat updates

        # --- Update Goal Statistics --- 
        if target_path:
            # Normalize target path if needed (e.g., remove leading ./)
            # For now, use as is
            stats = self.goal_stats[target_path] 
            stats["total_attempts"] += 1
            stats["last_attempt_time"] = time.time()
            stats["last_status"] = status

            log_prefix = f"[{self.get_name()} OBSERVED]"

            if status == "success":
                stats["success"] += 1
                self._logger.info(f"{log_prefix} Result='{msg_type}', Status='{status}', Target='{target_path}', Action='{action}', Sender='{sender}'")
            elif status in ["error", "failed", "timeout", "anomaly_reported"]:
                 stats["failure"] += 1
                 self._logger.warning(f"{log_prefix} Result='{msg_type}', Status='{status}', Target='{target_path}', Action='{action}', Sender='{sender}'")
            elif status == "skipped":
                 stats["skipped"] += 1
                 self._logger.info(f"{log_prefix} Result='{msg_type}', Status='{status}', Target='{target_path}', Action='{action}', Sender='{sender}'")
            elif status == "proposed": # From mutation_attempt
                 stats["proposed"] += 1
                 self._logger.info(f"{log_prefix} Result='{msg_type}', Status='{status}', Target='{target_path}', Action='{action}', Sender='{sender}'")
            else:
                 # Log unexpected statuses
                 self._logger.warning(f"{log_prefix} Result='{msg_type}', Status='UNEXPECTED({status})', Target='{target_path}', Action='{action}', Sender='{sender}'")

            # Optional: Prune old stats?
            # if len(self.goal_stats) > MAX_STATS_ENTRIES: prune_logic()
        else:
             # Log messages of relevant types that didn't have a clear target path
             if msg_type in ["REFACTOR_RESULT", "MUTATION_ATTEMPT", "ANOMALY"]:
                  self._logger.warning(f"[{self.get_name()} OBSERVED] Result='{msg_type}' Status='{status}' Sender='{sender}' but could not determine target path from content: {content}")

        # --- Process Specific Message Types ---
        self._logger.debug(f"[GoalManager] Checking message type: Type='{msg_type}', Keys in content: {list(content.keys()) if isinstance(content, dict) else 'N/A'}")
        if msg_type == "REFACTOR_RESULT":
            # Existing code for refactor_result handling...
            # Note: The actual processing logic for results still needs to be added here.
            # For now, we are just fixing the type check and stat updates.
            pass # Placeholder - Add actual logic later if needed beyond stat updates

    # Optional: Add cleanup for the task on shutdown
    async def shutdown(self):
        """Cleans up the background goal loop task."""
        if self._goal_loop_task and not self._goal_loop_task.done():
            self._logger.info(f"[{self.get_name()}] Requesting cancellation of goal injection loop task...")
            self._goal_loop_task.cancel()
            try:
                await self._goal_loop_task # Wait for cancellation to complete
            except asyncio.CancelledError:
                 self._logger.info(f"[{self.get_name()}] Goal injection loop task successfully cancelled during shutdown.")
            except Exception as e:
                 self._logger.error(f"[{self.get_name()}] Error during goal loop task cleanup:", exc_info=e)
        self._logger.info(f"[{self.get_name()}] Shutdown complete.") 