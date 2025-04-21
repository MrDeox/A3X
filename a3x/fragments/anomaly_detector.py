# a3x/fragments/anomaly_detector.py

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, DefaultDict
from collections import defaultdict
import time

# Core A³X components
from a3x.fragments.registry import fragment
from a3x.fragments.base import BaseFragment, FragmentDef
from a3x.core.context import FragmentContext
from a3x.core.tool_registry import ToolRegistry
# Import PROJECT_ROOT for path manipulation
try:
    from a3x.core.config import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent.parent # Fallback
    logging.getLogger(__name__).error("Failed to import PROJECT_ROOT from config in AnomalyDetectorFragment")

logger = logging.getLogger(__name__)

# --- Configuration for Anomaly Detection ---
# Thresholds can be adjusted or moved to a config file later
CONSECUTIVE_FAILURE_THRESHOLD = 3 # Trigger anomaly after 3 *consecutive* failures on the same file
FRAGMENT_ATTEMPT_THRESHOLD = 10   # Trigger anomaly if a single fragment makes too many attempts overall (could be refined by time window)
# How often to reset fragment attempt counts (in seconds) to avoid permanent flagging. None = never reset.
FRAGMENT_ATTEMPT_RESET_INTERVAL_SECONDS = 3600 # Reset fragment counts every hour

@fragment(
    name="AnomalyDetector",
    description="Listens for mutation/refactor results and detects potentially problematic patterns like repeated failures or excessive attempts.",
    category="Monitoring", # Or "QualityAssurance"
    skills=[], # This fragment primarily observes and reports
    managed_skills=[]
)
class AnomalyDetectorFragment(BaseFragment):
    """
    Monitors mutation and refactoring attempts, identifying and reporting anomalies.
    """

    def __init__(self, fragment_def: FragmentDef, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(fragment_def, tool_registry=tool_registry)
        # Internal state to track attempts and failures
        # Tracks stats per file path
        self.file_stats: DefaultDict[str, Dict[str, Any]] = defaultdict(lambda: {"total_attempts": 0, "consecutive_failures": 0, "last_status_time": 0})
        # Tracks attempts per fragment source
        self.fragment_attempt_counts: DefaultDict[str, int] = defaultdict(int)
        self.last_fragment_reset_time: float = time.time()
        self.memory_coins = 0 # Initialize balance

        self._logger.info(f"Fragment '{self.metadata.name}' initialized. Monitoring for anomalies... Balance: {self.memory_coins}")
        self._logger.info(f"Anomaly thresholds: ConsecutiveFailures={CONSECUTIVE_FAILURE_THRESHOLD}, FragmentAttempts={FRAGMENT_ATTEMPT_THRESHOLD}")

    def get_purpose(self) -> str:
        """Returns the purpose of this fragment."""
        return "Monitors code modification attempts for anomalous patterns (repeated failures, excessive attempts) and reports them."

    def get_balance(self) -> int:
        """Returns the current memory coin balance."""
        return self.memory_coins

    async def execute_task(
        self,
        objective: str, # e.g., "Monitor modification attempts"
        tools: list = None,
        context: Optional[FragmentContext] = None,
    ) -> str:
        """Main entry point."""
        if not context:
             self._logger.error("FragmentContext is required for AnomalyDetectorFragment.")
             return "Error: Context not provided."

        self._logger.info(f"[{self.get_name()}] Ready and listening for modification results via handle_realtime_chat.")
        # Like Mutator, relies on handle_realtime_chat being called by the system
        return "AnomalyDetectorFragment is active and ready to analyze modification results."

    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """Handles incoming messages, specifically mutation_attempt and refactor_result."""
        msg_type = message.get("type")
        sender = message.get("sender", "Unknown")
        content = message.get("content", {})

        # --- Check if message is relevant ---
        if msg_type not in ["mutation_attempt", "refactor_result"]:
            # self._logger.debug(f"[{self.get_name()}] Ignoring message type: {msg_type}")
            return # Not interested in other types

        self._logger.debug(f"[{self.get_name()}] Received relevant message type '{msg_type}' from '{sender}'.")

        # --- Extract common information ---
        file_path_relative = None
        status = content.get("status")
        details_str = content.get("details", "") # refactor_result might have JSON string
        details_dict = {}

        # Try to parse details if it's a JSON string (from refactor_result)
        if isinstance(details_str, str):
             try:
                 details_dict = json.loads(details_str)
             except json.JSONDecodeError:
                 # If details wasn't JSON, it might be a simple string or None
                 details_dict = {"details_string": details_str}
        elif isinstance(details_str, dict): # If details was already a dict
             details_dict = details_str

        # Get file path - handle different keys
        if "file_path" in content:
            file_path_relative = content.get("file_path")
        elif "target_file" in content: # Added check for Mutator messages
            file_path_relative = content.get("target_file")
        elif "target" in content: # StructureAutoRefactor messages
            file_path_relative = content.get("target")
        elif "generated_path" in details_dict: # Check parsed details (e.g., from sandbox failure)
             # Need to convert potential absolute path from details
             abs_path = details_dict.get("generated_path")
             try:
                 if abs_path and PROJECT_ROOT and Path(PROJECT_ROOT).is_dir():
                     file_path_relative = str(Path(abs_path).resolve().relative_to(Path(PROJECT_ROOT).resolve()))
                 else:
                     self._logger.warning(f"[{self.get_name()}] Cannot determine relative path from details: {details_dict}. PROJECT_ROOT='{PROJECT_ROOT}'")
             except Exception as e:
                 self._logger.warning(f"[{self.get_name()}] Error converting path from details '{abs_path}': {e}")
        elif "corrected_path" in details_dict: # Check parsed details from successful correction
            file_path_relative = details_dict.get("corrected_path")


        if not file_path_relative or not status:
            self._logger.warning(f"[{self.get_name()}] Skipping message type '{msg_type}' due to missing file path or status. Content: {content}")
            return

        # Ensure path is relative (it should be by now, but double-check)
        if Path(file_path_relative).is_absolute():
             self._logger.warning(f"[{self.get_name()}] Received absolute path '{file_path_relative}' unexpectedly. Skipping.")
             return

        # --- Update Stats and Check for Anomalies ---
        await self._update_stats_and_check_anomalies(file_path_relative, status, sender, context, details_dict)

        # --- Handle Reward Messages (Separate from anomaly logic) --- 
        if msg_type == "reward":
             target = content.get("target")
             if target == self.get_name(): # Reward is for me
                 amount = content.get("amount")
                 reason = content.get("reason", "No reason specified")
                 sender = message.get("sender", "Unknown")
                 if isinstance(amount, int) and amount > 0:
                     self.memory_coins += amount
                     self._logger.info(f"[{self.get_name()}] Received {amount} memory_coins from '{sender}' for: '{reason}'. New balance: {self.memory_coins}")
                 else:
                      self._logger.warning(f"[{self.get_name()}] Received invalid reward amount '{amount}' from '{sender}' for target '{target}'. Ignoring.")
             # else: Reward was for someone else, ignore

    async def _update_stats_and_check_anomalies(self, file_path: str, status: str, sender: str, context: FragmentContext, details: Dict):
        """Updates internal counters and checks anomaly rules."""
        current_time = time.time()

        # --- File Stats ---
        stats = self.file_stats[file_path]
        stats["total_attempts"] += 1
        stats["last_status_time"] = current_time

        is_failure = status in ["error", "failed", "timeout"] # Define what constitutes a failure

        if is_failure:
            stats["consecutive_failures"] += 1
            self._logger.warning(f"[{self.get_name()}] Failure recorded for '{file_path}' (Consecutive: {stats['consecutive_failures']}). Source: {sender}.")
        else: # Success, proposed, skipped, no_change etc.
            if stats["consecutive_failures"] > 0:
                 self._logger.info(f"[{self.get_name()}] Consecutive failure streak for '{file_path}' reset after status: {status}.")
            stats["consecutive_failures"] = 0 # Reset on non-failure

        # Check for repeated failures on the same file
        if stats["consecutive_failures"] >= CONSECUTIVE_FAILURE_THRESHOLD:
            # Report the anomaly first
            await self._report_anomaly(
                context=context,
                issue_type="repeated_failure",
                file_path=file_path,
                suspect_fragment=sender,
                details_msg=f"Detected {stats['consecutive_failures']} consecutive failures/errors for file '{file_path}'. Last reported by '{sender}'.",
                extra_details=details # Pass along details from the triggering message
            )
            
            # <<< ADD: Trigger Mutator >>>
            self._logger.info(f"[{self.get_name()}] Threshold met. Requesting mutation for '{file_path}'.")
            mutation_request_content = {
                "target": file_path,
                "failure_reason": details, # Pass the details dict from the last failed message
                "triggering_fragment": sender
            }
            await self.post_chat_message(
                context=context,
                message_type="mutation_request", 
                content=mutation_request_content
                # target_fragment="Mutator" # Optional: Target Mutator specifically if routing exists
            )
            # <<< END ADD >>>
            
            # Reset count after reporting and requesting mutation to avoid immediate re-trigger
            self._logger.info(f"[{self.get_name()}] Resetting consecutive failure count for '{file_path}' after requesting mutation.")
            stats["consecutive_failures"] = 0 

        # --- Fragment Stats ---
        # Check if it's time to reset fragment counts
        if FRAGMENT_ATTEMPT_RESET_INTERVAL_SECONDS and (current_time - self.last_fragment_reset_time > FRAGMENT_ATTEMPT_RESET_INTERVAL_SECONDS):
             self._logger.info(f"[{self.get_name()}] Resetting fragment attempt counts (Interval: {FRAGMENT_ATTEMPT_RESET_INTERVAL_SECONDS}s).")
             self.fragment_attempt_counts.clear()
             self.last_fragment_reset_time = current_time

        self.fragment_attempt_counts[sender] += 1
        attempts_by_sender = self.fragment_attempt_counts[sender]

        # Check for excessive attempts by a single fragment
        if attempts_by_sender == FRAGMENT_ATTEMPT_THRESHOLD: # Report only when threshold is first hit
            await self._report_anomaly(
                context=context,
                issue_type="excessive_attempts",
                file_path=None, # Anomaly relates to the fragment, not a specific file
                suspect_fragment=sender,
                details_msg=f"Fragment '{sender}' has made {attempts_by_sender} modification attempts within the current tracking window.",
                extra_details={"current_attempt_count": attempts_by_sender}
            )
        elif attempts_by_sender > FRAGMENT_ATTEMPT_THRESHOLD:
             # Log subsequent attempts by the flagged fragment but don't spam reports
             self._logger.debug(f"[{self.get_name()}] Fragment '{sender}' continues attempts ({attempts_by_sender}) above threshold.")


    async def _report_anomaly(self, context: FragmentContext, issue_type: str, file_path: Optional[str], suspect_fragment: str, details_msg: str, extra_details: Optional[Dict] = None):
        """Formats and broadcasts an anomaly report."""
        self._logger.warning(f"ANOMALY DETECTED: Type='{issue_type}', File='{file_path}', Suspect='{suspect_fragment}', Details='{details_msg}'")

        anomaly_message = {
            "type": "anomaly",
            "issue": issue_type,
            "file_path": file_path, # Can be None
            "suspect_fragment": suspect_fragment,
            "details": details_msg,
            "extra_context": extra_details or {}
        }
        try:
            await self.post_chat_message(
                context=context,
                message_type="anomaly", # Specific message type for anomalies
                content=anomaly_message
                # target_fragment=None # Broadcast
            )
            self._logger.info(f"[{self.get_name()}] Broadcasted anomaly report: {issue_type} for {file_path or suspect_fragment}.")
        except Exception as e:
            self._logger.error(f"[{self.get_name()}] Failed to broadcast anomaly report: {e}") 