import asyncio
import logging
from typing import Dict, Any, Optional
from collections import defaultdict
import time

from .base import BaseFragment, FragmentDef
from a3x.core.context import FragmentContext

logger = logging.getLogger(__name__)

# Configuration
PERFORMANCE_ANALYSIS_INTERVAL_SECONDS = 60 # Analyze performance every minute
MIN_OBSERVATIONS_THRESHOLD = 20 # Min messages observed from a fragment before considering deactivation
INEFFECTIVENESS_ERROR_RATIO = 0.8 # Suggest deactivation if error+skip ratio exceeds this (and successes are low)
MAX_SUCCESS_FOR_DEACTIVATION = 1 # Max number of successes allowed to still consider deactivation

class PerformanceMonitorFragment(BaseFragment):
    """Monitors the performance of other fragments and suggests deactivating ineffective ones."""

    def __init__(self, fragment_def: FragmentDef, tool_registry=None):
        super().__init__(fragment_def, tool_registry)
        self._fragment_context: Optional[FragmentContext] = None
        # Stats: fragment_performance[sender_name][status] = count
        self.fragment_performance: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._analysis_loop_task: Optional[asyncio.Task] = None
        self._logger.info(f"[{self.get_name()}] Initialized. Will analyze fragment performance every {PERFORMANCE_ANALYSIS_INTERVAL_SECONDS} seconds.")

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        """Returns a description of this fragment's purpose."""
        return f"Monitors results messages (refactor_result, mutation_attempt, etc.) to track fragment performance. Suggests deactivating fragments with consistently low success rates (every {PERFORMANCE_ANALYSIS_INTERVAL_SECONDS}s)."

    # Override set_context to store context and start the analysis loop
    def set_context(self, context: FragmentContext):
        """Receives the context and starts the performance analysis loop."""
        # The context passed from the runner loop IS the SharedTaskContext
        shared_context = context
        super().set_context(shared_context) # Call parent's set_context
        self._fragment_context = shared_context # Store the shared context
        self._logger.info(f"[{self.get_name()}] Context received, starting performance analysis loop.")
        self._start_analysis_loop() # Start the loop now that context is available

    async def _start_analysis_loop(self):
        if self._analysis_loop_task is None or self._analysis_loop_task.done():
            logger.info("Starting performance analysis loop.")
            # Temporarily disable automatic loop start for debugging dispatcher queue issues
            # self._analysis_loop_task = asyncio.create_task(
            #     self._performance_analysis_loop(), name="PerformanceAnalysisLoop"
            # )
            logger.warning("PerformanceMonitorFragment analysis loop DISABLED for debugging.")
        else:
            logger.warning("Performance analysis loop already running or not finished.")

    async def _performance_analysis_loop(self):
        """Periodically calls the analysis function."""
        while True:
            try:
                await asyncio.sleep(PERFORMANCE_ANALYSIS_INTERVAL_SECONDS)
                self._logger.info(f"[{self.get_name()}] Performing periodic fragment performance analysis...")
                await self._analyze_performance()
            except asyncio.CancelledError:
                self._logger.info(f"[{self.get_name()}] Performance analysis loop cancelled.")
                break
            except Exception as e:
                self._logger.exception(f"[{self.get_name()}] Error in performance analysis loop:")
                await asyncio.sleep(PERFORMANCE_ANALYSIS_INTERVAL_SECONDS) # Wait before retrying

    def _handle_loop_completion(self, task: asyncio.Task):
        """Callback to log completion or errors of the analysis loop."""
        # Similar logging as other loop handlers
        try:
            exception = task.exception()
            if exception:
                self._logger.error(f"[{self.get_name()}] Performance analysis loop task failed:", exc_info=exception)
            else:
                self._logger.info(f"[{self.get_name()}] Performance analysis loop task completed.")
        except asyncio.CancelledError:
            self._logger.info(f"[{self.get_name()}] Performance analysis loop task was cancelled.")
        self._analysis_loop_task = None

    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """Processes incoming messages to update performance statistics."""
        # Ensure context is set (can happen if this fragment starts late)
        if self._fragment_context is None:
            self.set_context(context)

        msg_type = message.get("type")
        sender = message.get("sender", "Unknown")
        content = message.get("content", {})
        status = content.get("status", "unknown").lower() if isinstance(content, dict) else "unknown"

        # List of message types that indicate a fragment performed an action with a result
        RESULT_MESSAGE_TYPES = [
            "REFACTOR_RESULT",
            "MUTATION_ATTEMPT",
            "MANAGER_RESULT",
            "ARCHITECT_RESULT",
            # Add other relevant result types here
        ]

        if sender != "Unknown" and msg_type in RESULT_MESSAGE_TYPES:
            # Ignore results sent by self or system components if needed
            if sender == self.get_name():
                 return

            # Update stats for the sending fragment
            self.fragment_performance[sender][status] += 1
            self.fragment_performance[sender]["total_results"] += 1 # Track total results observed
            self._logger.debug(f"[{self.get_name()}] Recorded '{status}' result from {sender}. Total results for {sender}: {self.fragment_performance[sender]['total_results']}")

    async def _analyze_performance(self):
        """Analyzes collected performance statistics and suggests deactivation for ineffective fragments."""
        if not self._fragment_context:
            self._logger.error(f"[{self.get_name()}] Cannot analyze performance: FragmentContext not set.")
            return

        fragments_to_deactivate = []
        for fragment_name, stats in self.fragment_performance.items():
            total_results = stats.get("total_results", 0)
            successes = stats.get("success", 0)
            errors = stats.get("error", 0) + stats.get("failed", 0) # Combine error types
            skips = stats.get("skipped", 0) + stats.get("no_change", 0) # Combine skip types
            failures_and_skips = errors + skips

            # Check deactivation criteria
            if total_results >= MIN_OBSERVATIONS_THRESHOLD and successes <= MAX_SUCCESS_FOR_DEACTIVATION:
                # Check if division by zero would occur
                if total_results > 0:
                    failure_skip_ratio = failures_and_skips / total_results
                    if failure_skip_ratio >= INEFFECTIVENESS_ERROR_RATIO:
                        self._logger.warning((
                            f"[{self.get_name()}] Fragment '{fragment_name}' flagged for potential deactivation. "
                            f"Stats: TotalResults={total_results}, Successes={successes}, Errors={errors}, Skips={skips} "
                            f"(Failure/Skip Ratio: {failure_skip_ratio:.2f} >= {INEFFECTIVENESS_ERROR_RATIO})"
                        ))
                        fragments_to_deactivate.append(fragment_name)
                elif failures_and_skips > 0: # Handle case where total_results is 0 but there are failures/skips (edge case)
                     self._logger.warning((
                            f"[{self.get_name()}] Fragment '{fragment_name}' flagged for potential deactivation. "
                            f"Stats: TotalResults=0, Successes={successes}, Errors={errors}, Skips={skips} "
                     ))
                     fragments_to_deactivate.append(fragment_name)


        # --- Post Deactivation Suggestions --- 
        if fragments_to_deactivate:
            # For now, suggest deactivating the first one found to avoid flooding
            target_fragment_name = fragments_to_deactivate[0]
            self._logger.info(f"[{self.get_name()}] Proposing deactivation of fragment: {target_fragment_name}")

            directive_content = {
                "type": "directive",
                "action": "deactivate_fragment",
                "target": target_fragment_name, # Name of the fragment to deactivate
                "message": f"Suggesting deactivation of fragment '{target_fragment_name}' due to low success rate and high error/skip ratio based on {self.fragment_performance[target_fragment_name]['total_results']} observed results."
            }

            try:
                await self.post_chat_message(
                    message_type="deactivation_proposal",
                    content={
                        "fragment_name": target_fragment_name,
                        "reason": f"Low success rate and high error/skip ratio based on {self.fragment_performance[target_fragment_name]['total_results']} observed results.",
                        "metrics": {
                            "total_results": self.fragment_performance[target_fragment_name]["total_results"],
                            "successes": self.fragment_performance[target_fragment_name]["success"],
                            "errors": self.fragment_performance[target_fragment_name]["error"] + self.fragment_performance[target_fragment_name]["failed"],
                            "skips": self.fragment_performance[target_fragment_name]["skipped"] + self.fragment_performance[target_fragment_name]["no_change"]
                        }
                    },
                    target_fragment="Deactivator"
                )
                self._logger.info(f"[{self.get_name()}] Posted deactivation proposal for {target_fragment_name} to Deactivator. Reason: Low success rate and high error/skip ratio based on {self.fragment_performance[target_fragment_name]['total_results']} observed results.")
            except Exception as e:
                self._logger.error(f"[{self.get_name()}] Failed to post deactivation proposal for {target_fragment_name}: {e}", exc_info=True)
        else:
            self._logger.info(f"[{self.get_name()}] Performance analysis complete. No fragments met deactivation criteria.")

    async def shutdown(self):
        """Cleans up the background analysis loop task.""" 
        if self._analysis_loop_task and not self._analysis_loop_task.done():
            self._logger.info(f"[{self.get_name()}] Requesting cancellation of performance analysis loop task...")
            self._analysis_loop_task.cancel()
            try:
                await self._analysis_loop_task
            except asyncio.CancelledError:
                 self._logger.info(f"[{self.get_name()}] Performance analysis loop task successfully cancelled.")
            except Exception as e:
                 self._logger.error(f"[{self.get_name()}] Error during analysis loop task cleanup:", exc_info=e)
        self._logger.info(f"[{self.get_name()}] Shutdown complete.") 