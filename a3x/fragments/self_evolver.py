import asyncio
import logging
from typing import Dict, Any, Optional
from collections import defaultdict, deque
import time
import random
import json # Added for debug logging stats

from .base import BaseFragment, FragmentDef
from a3x.core.context import FragmentContext

logger = logging.getLogger(__name__)

# Configuration
ANALYSIS_INTERVAL_MESSAGES = 50 # Analyze after every N messages
FAILURE_THRESHOLD_RATIO = 0.6 # Suggest improvement if failure rate exceeds this
RECENT_HISTORY_LENGTH = 100 # How many message digests to keep for repetition checks

class SelfEvolverFragment(BaseFragment):
    """
    Observes system behavior, identifies potential inefficiencies or gaps,
    and suggests the creation of new fragments to address them.
    """
    def __init__(self, fragment_def: FragmentDef, tool_registry=None):
        super().__init__(fragment_def, tool_registry)
        self._fragment_context: Optional[FragmentContext] = None
        self._message_count = 0
        # Statistics Tracking
        # fragment_stats[sender_name][status] = count
        self.fragment_stats: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        # action_stats[action_type][status] = count
        self.action_stats: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        # directive_stats[action_type][status] = count (for received directives)
        self.directive_stats: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Track last N message digests (type:sender:status:target_digest) to detect loops
        self.recent_history: deque[str] = deque(maxlen=RECENT_HISTORY_LENGTH)
        self._logger.info(f"[{self.get_name()}] Initialized. Will analyze system behavior every {ANALYSIS_INTERVAL_MESSAGES} messages.")

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        """Returns a description of this fragment's purpose."""
        return "Observes system messages to identify patterns (e.g., high failure rates, repetitive actions, coverage gaps) and suggests creating new fragments via 'create_fragment' directives to improve the system."

    def set_context(self, context: FragmentContext):
        """Receives the context needed for the SelfEvolver."""
        # The context passed from the runner loop IS the SharedTaskContext
        shared_context = context 
        super().set_context(shared_context) # Call parent's set_context with the SharedTaskContext
        self._fragment_context = shared_context # Store the shared context
        self._logger.info(f"[{self.get_name()}] Context received.")

    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """Processes all incoming messages to update statistics and potentially trigger analysis."""
        if self._fragment_context is None:
             self._logger.warning(f"[{self.get_name()}] Context received via handle_realtime_chat.")
             self.set_context(context) # Ensure context is set if first message

        self._message_count += 1
        msg_type = message.get("type")
        sender = message.get("sender", "Unknown")
        content = message.get("content", {})
        status = content.get("status", "unknown").lower() if isinstance(content, dict) else "unknown"
        action = content.get("original_action", content.get("action")) if isinstance(content, dict) else None
        target = content.get("target", content.get("file_path")) if isinstance(content, dict) else None # Simple target extraction

        # --- Update Statistics ---
        if sender != "Unknown":
            self.fragment_stats[sender][status] += 1
            self.fragment_stats[sender]["total"] += 1 # Keep total count per fragment

        if action:
             self.action_stats[action][status] += 1
             self.action_stats[action]["total"] += 1

        # Track directives specifically
        if msg_type == "ARCHITECTURE_SUGGESTION" and isinstance(content, dict) and content.get("type") == "directive":
             directive_action = content.get("action")
             if directive_action:
                  # Track directive receipt separately if needed, TBD
                  pass

        # Update recent history (using a simple digest)
        target_digest = str(target)[:30] if target else "None"
        history_digest = f"{msg_type}:{sender}:{status}:{target_digest}"
        self.recent_history.append(history_digest)

        # --- Trigger Analysis Periodically ---
        if self._message_count % ANALYSIS_INTERVAL_MESSAGES == 0:
            self._logger.info(f"[{self.get_name()}] Reached {self._message_count} messages. Triggering analysis...")
            await self._analyze_and_suggest()

    async def _analyze_and_suggest(self):
        """Analyzes collected statistics and recent history to identify areas for improvement."""
        if not self._fragment_context:
             self._logger.error(f"[{self.get_name()}] Cannot perform analysis: FragmentContext not set.")
             return

        suggestions = []

        # 1. Check for high fragment failure rates
        for fragment, stats in self.fragment_stats.items():
            total = stats.get("total", 0)
            failures = stats.get("error", 0) + stats.get("failed", 0)
            if total > 10 and failures / total > FAILURE_THRESHOLD_RATIO: # Avoid triggering on very few messages
                suggestion = {
                     "target": f"a3x/fragments/generated/assistant_for_{fragment.lower()}.py",
                     "message": f"Fragment '{fragment}' has a high failure rate ({failures}/{total}). Create an assistant fragment to analyze its errors, monitor its state, or suggest specific corrections for its common failure modes."
                }
                suggestions.append(suggestion)
                self._logger.warning(f"[{self.get_name()}] High failure rate detected for '{fragment}' ({failures}/{total}). Suggesting assistant.")

        # 2. Check for repetitive skipped actions (simple check)
        # Count occurrences of "skipped" status for specific actions
        for action, stats in self.action_stats.items():
             total = stats.get("total", 0)
             skipped = stats.get("skipped", 0)
             if total > 15 and skipped / total > 0.7: # If >70% of attempts for an action are skipped
                  suggestion = {
                       "target": f"a3x/fragments/generated/handler_for_{action.lower()}.py",
                       "message": f"Action '{action}' is frequently skipped ({skipped}/{total}). Create a dedicated fragment to handle '{action}' directives more effectively or investigate why it's being skipped."
                  }
                  suggestions.append(suggestion)
                  self._logger.warning(f"[{self.get_name()}] High skip rate detected for action '{action}' ({skipped}/{total}). Suggesting handler.")

        # 3. TODO: Add more sophisticated analysis (e.g., detecting loops in recent_history, identifying uncovered directive types)

        # --- Post Suggestions ---
        if suggestions:
             # Randomly pick one suggestion for now to avoid flooding
             chosen_suggestion = random.choice(suggestions)
             self._logger.info(f"[{self.get_name()}] Proposing creation of fragment: {chosen_suggestion['target']}")

             directive_content = {
                 "type": "directive",
                 "action": "create_fragment", # Specific action for creating fragments
                 "target": chosen_suggestion["target"],
                 "message": chosen_suggestion["message"]
             }

             try:
                 await self.post_chat_message(
                     context=self._fragment_context,
                     message_type="architecture_suggestion",
                     content=directive_content
                 )
                 self._logger.info(f"[{self.get_name()}] Posted 'create_fragment' directive for {chosen_suggestion['target']}")
                 # Optional: Reset some stats after making a suggestion?
             except Exception as e:
                 self._logger.exception(f"[{self.get_name()}] Failed to post create_fragment suggestion:")
        else:
             self._logger.info(f"[{self.get_name()}] Analysis complete. No major issues detected requiring new fragments at this time.")

    async def shutdown(self):
        """Optional cleanup actions on shutdown."""
        self._logger.info(f"[{self.get_name()}] Shutdown complete. Final message count: {self._message_count}")
        # Log final stats (optional)
        self._logger.debug(f"Final Fragment Stats: {json.dumps(self.fragment_stats, default=str, indent=2)}") # Use default=str for defaultdict
        self._logger.debug(f"Final Action Stats: {json.dumps(self.action_stats, default=str, indent=2)}") # Use default=str for defaultdict 