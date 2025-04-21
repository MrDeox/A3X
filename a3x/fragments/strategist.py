import asyncio
import logging
import random
import re
import uuid
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
from pathlib import Path

from .base import BaseFragment, FragmentDef
from a3x.core.context import FragmentContext

logger = logging.getLogger(__name__)

# Configuration
GOAL_GENERATION_THRESHOLD = 2 # Generate a goal after processing N summaries
# Regex to parse heuristics from summary (adjust if format changes)
HEURISTIC_PATTERN = re.compile(r"- Count=(\d+): Sender='(.*?)', Action='(.*?)', Target='(.*?)'")
# Where GoalManager creates files (to mimic target generation)
GOAL_TARGET_PREFIX = "a3x/generated/strategic"

class StrategistFragment(BaseFragment):
    """Analyzes learned heuristics and generates strategic goals based on successful patterns."""

    def __init__(self, fragment_def: FragmentDef, tool_registry=None):
        super().__init__(fragment_def, tool_registry)
        self._fragment_context: Optional[FragmentContext] = None
        self._processed_summaries_count = 0
        # Store the most promising heuristics observed recently
        self.promising_heuristics: List[Tuple[int, str, str, str]] = []
        self._logger.info(f"[{self.get_name()}] Initialized. Will generate goal every {GOAL_GENERATION_THRESHOLD} summaries.")

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        """Returns a description of this fragment's purpose."""
        return "Analyzes learning_summary messages to identify successful patterns (heuristics) and generates new 'generate_goal' directives inspired by them."

    def set_context(self, context: FragmentContext):
        """Receives the context needed for the Strategist."""
        # The context passed from the runner loop IS the SharedTaskContext
        shared_context = context
        super().set_context(shared_context) # Call parent's set_context
        self._fragment_context = shared_context # Store the shared context
        self._logger.info(f"[{self.get_name()}] Context received.")

    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """Processes incoming messages, looking for learning_summary."""
        if self._fragment_context is None:
            self.set_context(context)

        msg_type = message.get("type")
        sender = message.get("sender") # Should be ConsolidatorFragment
        content = message.get("content")

        if msg_type == "LEARNING_SUMMARY" and isinstance(content, dict):
            self._logger.info(f"[{self.get_name()}] Received learning summary from {sender}.")
            summary_text = content.get("summary_text")
            if summary_text:
                self._processed_summaries_count += 1
                self._parse_and_store_heuristics(summary_text)

                if self._processed_summaries_count >= GOAL_GENERATION_THRESHOLD:
                    self._logger.info(f"[{self.get_name()}] Reached threshold ({self._processed_summaries_count}/{GOAL_GENERATION_THRESHOLD}). Generating strategic goal...")
                    await self._generate_strategic_goal()
                    self._processed_summaries_count = 0 # Reset counter
                    self.promising_heuristics = [] # Clear stored heuristics for next cycle

    def _parse_and_store_heuristics(self, summary_text: str):
        """Parses the summary text and stores the extracted heuristics."""
        found_heuristics = HEURISTIC_PATTERN.findall(summary_text)
        if found_heuristics:
            self._logger.debug(f"[{self.get_name()}] Parsed {len(found_heuristics)} heuristics from summary.")
            for count_str, sender, action, target_pattern in found_heuristics:
                try:
                    count = int(count_str)
                    # Store as tuple: (count, sender, action, target_pattern)
                    self.promising_heuristics.append((count, sender, action, target_pattern))
                except ValueError:
                    self._logger.warning(f"Could not parse count '{count_str}' in heuristic line.")
            # Sort by count descending
            self.promising_heuristics.sort(key=lambda x: x[0], reverse=True)
        else:
             self._logger.debug(f"[{self.get_name()}] No heuristics found matching pattern in summary.")

    async def _generate_strategic_goal(self):
        """Selects a promising heuristic and generates a corresponding goal."""
        if not self._fragment_context:
            self._logger.error(f"[{self.get_name()}] Cannot generate goal: FragmentContext not set.")
            return
        if not self.promising_heuristics:
            self._logger.info(f"[{self.get_name()}] No promising heuristics available to generate a goal.")
            return

        # Select the top heuristic (highest count)
        top_heuristic = self.promising_heuristics[0]
        count, h_sender, h_action, h_target_pattern = top_heuristic

        # --- Generate Goal Based on Heuristic --- 
        # Simple strategy: Replicate the successful action with a new target
        # More complex strategies could involve combining heuristics, modifying actions, etc.

        new_target_name = f"{h_action}_{uuid.uuid4().hex[:8]}" # Generate a unique name
        new_target_path = f"{GOAL_TARGET_PREFIX}/{new_target_name}" # Default path

        # Adapt target path based on pattern if possible (simple example)
        if h_target_pattern.startswith("extension:"):
            ext = h_target_pattern.split(':')[1]
            new_target_path += ext
        elif h_target_pattern.startswith("parent_dir:"):
             parent = h_target_pattern.split(':')[1]
             # Ensure parent exists conceptually within our target prefix
             safe_parent = Path(parent).name # Use only the last part to avoid path traversal issues
             new_target_path = f"{GOAL_TARGET_PREFIX}/{safe_parent}/{new_target_name}.py" # Assume .py if dir
        else:
            new_target_path += ".py" # Default to .py

        goal_message_template = f"Based on the observed success of '{h_sender}' performing '{h_action}' on targets like '{h_target_pattern}' (observed {count} times), attempt the action '{h_action}' on the new target '{new_target_path}'."

        self._logger.info(f"[{self.get_name()}] Generating goal inspired by heuristic: Sender='{h_sender}', Action='{h_action}', Target='{h_target_pattern}'")

        goal_directive_content = {
            "type": "directive",
            "action": h_action, # Use the action from the heuristic
            "target": new_target_path,
            "message": goal_message_template,
            "source_heuristic": { # Add context about the source
                 "sender": h_sender,
                 "action": h_action,
                 "target_pattern": h_target_pattern,
                 "count": count
             }
        }

        try:
            # Post using the standard architecture_suggestion type
            await self.post_chat_message(
                context=self._fragment_context,
                message_type="architecture_suggestion",
                content=goal_directive_content
            )
            self._logger.info(f"[{self.get_name()}] Posted strategic 'generate_goal' directive targeting '{new_target_path}' based on heuristic.")
        except Exception as e:
            self._logger.exception(f"[{self.get_name()}] Failed to post strategic goal directive:")

    async def shutdown(self):
        """Optional cleanup actions on shutdown."""
        self._logger.info(f"[{self.get_name()}] Shutdown complete.") 