import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict

from .base import BaseFragment, FragmentDef
from a3x.core.context import FragmentContext

logger = logging.getLogger(__name__)

# Configuration
SUMMARY_INTERVAL = 40 # Post summary after every N relevant messages processed
MIN_COUNT_FOR_SUMMARY = 3 # Minimum count for a heuristic to be included in the summary

class ConsolidatorFragment(BaseFragment):
    """Observes successful actions and consolidates recurring patterns into simple heuristics."""

    def __init__(self, fragment_def: FragmentDef, tool_registry=None):
        super().__init__(fragment_def, tool_registry)
        self._fragment_context: Optional[FragmentContext] = None
        self._processed_success_count = 0
        # Heuristics: heuristics[(sender, action, target_pattern)] = count
        self.heuristics: defaultdict[tuple, int] = defaultdict(int)
        self._logger.info(f"[{self.get_name()}] Initialized. Will summarize heuristics every {SUMMARY_INTERVAL} successes.")

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        """Returns a description of this fragment's purpose."""
        return "Observes successful results (refactor_result, manager_result, etc.) to identify and store recurring patterns (heuristics) about what tends to work."

    def set_context(self, context: FragmentContext):
        """Receives the context needed for the Consolidator."""
        # The context passed from the runner loop IS the SharedTaskContext
        shared_context = context 
        super().set_context(shared_context) # Call parent's set_context
        self._fragment_context = shared_context # Store the shared context
        self._logger.info(f"[{self.get_name()}] Context received.")

    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """Processes incoming messages, looking for successful results to update heuristics."""
        if self._fragment_context is None:
            self.set_context(context)

        msg_type = message.get("type")
        sender = message.get("sender", "Unknown")
        content = message.get("content", {})
        status = content.get("status") if isinstance(content, dict) else None

        # Only process successful results from known types
        RELEVANT_SUCCESS_TYPES = [
            "REFACTOR_RESULT",
            "MANAGER_RESULT",
            "MUTATION_ATTEMPT", # Consider success here
            "ARCHITECT_RESULT",
            "DEACTIVATOR_RESULT" # Even successful simulation is a pattern
        ]

        if status == "success" and msg_type in RELEVANT_SUCCESS_TYPES and sender != "Unknown":
            self._processed_success_count += 1
            
            action = content.get("original_action")
            target = content.get("target") or content.get("target_fragment")
            
            if not action or not target:
                 # Try to get action/target from original directive if nested
                 original_directive = content.get("original_directive")
                 if isinstance(original_directive, dict):
                     action = action or original_directive.get("action")
                     target = target or original_directive.get("target")

            if action and target:
                 # --- Simple Pattern Extraction --- 
                 # Use sender, action, and target file extension or parent dir as pattern
                 target_pattern = "unknown_target_type"
                 try:
                     p = Path(target)
                     if p.suffix:
                         target_pattern = f"extension:{p.suffix}"
                     elif p.parent != Path('.'):
                         target_pattern = f"parent_dir:{p.parent}"
                     else:
                         target_pattern = f"name:{p.name}" # Fallback to name if no suffix/parent
                 except Exception:
                     target_pattern = "invalid_target_path"

                 heuristic_key = (sender, action, target_pattern)
                 self.heuristics[heuristic_key] += 1
                 self._logger.debug(f"[{self.get_name()}] Updated heuristic: {heuristic_key} -> {self.heuristics[heuristic_key]}")
                 # ----------------------------------

                 # Check if it's time to summarize
                 if self._processed_success_count % SUMMARY_INTERVAL == 0:
                     await self._post_learning_summary()
            else:
                self._logger.debug(f"[{self.get_name()}] Received success message type {msg_type} from {sender}, but couldn't extract clear action/target pattern. Content: {content}")
        
        # <<< NEW LOGIC: Process successful PLAN execution results >>>
        elif msg_type == "plan_execution_result" and status == "success":
             plan_id = content.get("plan_id", "unknown")
             self._logger.info(f"[{self.get_name()}] Detected SUCCESSFUL plan execution: ID='{plan_id}'")
             
             # Attempt to extract the topic from the plan steps
             topic = "unknown_topic"
             step_results = content.get("step_results", [])
             if isinstance(step_results, list):
                 for step in step_results:
                     action_details = step.get("action_details", {})
                     if isinstance(action_details, dict):
                         skill = action_details.get("skill")
                         if skill == "research_topic":
                             params = action_details.get("parameters", {})
                             if isinstance(params, dict):
                                 topic = params.get("topic", "unknown_topic")
                                 break # Found the research topic
             
             # Generate the learning summary message
             summary_message_content = {
                 # type/origin will be set by post_chat_message
                 # "type": "learning_summary", 
                 # "origin": self.get_name(), 
                 "summary": f"Plan '{plan_id}' involving topic '{topic}' completed successfully.", # Simple summary for now
                 "topic": topic,
                 "plan_id": plan_id
             }
             
             try:
                 await self.post_chat_message(
                     message_type="learning_summary",
                     content=summary_message_content
                 )
                 self._logger.info(f"[{self.get_name()}] Posted learning_summary for successful plan '{plan_id}'.")
             except Exception as e:
                 self._logger.error(f"[{self.get_name()}] Failed to post learning_summary for plan '{plan_id}': {e}", exc_info=True)
        # <<< END NEW LOGIC >>>

    async def _post_learning_summary(self):
        """Formats and posts a summary of the currently learned heuristics."""
        if not self._fragment_context:
             self._logger.error(f"[{self.get_name()}] Cannot post summary: FragmentContext not set.")
             return
        if not self.heuristics:
             self._logger.info(f"[{self.get_name()}] No heuristics learned yet, skipping summary.")
             return

        summary_lines = [f"Learned Heuristics Summary (Count >= {MIN_COUNT_FOR_SUMMARY}):"]
        heuristics_found = 0
        # Sort by count descending for relevance
        sorted_heuristics = sorted(self.heuristics.items(), key=lambda item: item[1], reverse=True)

        for (sender, action, target_pattern), count in sorted_heuristics:
            if count >= MIN_COUNT_FOR_SUMMARY:
                 summary_lines.append(f"  - Count={count}: Sender='{sender}', Action='{action}', Target='{target_pattern}'")
                 heuristics_found += 1

        if heuristics_found == 0:
             summary_lines.append("  (No heuristics met the minimum count threshold yet.)")

        summary_text = "\n".join(summary_lines)
        self._logger.info(f"[{self.get_name()}] Posting learning summary:\n{summary_text}")

        summary_content = {
            "summary_text": summary_text,
            "total_heuristics_tracked": len(self.heuristics),
            "heuristics_in_summary": heuristics_found
        }

        try:
            await self.post_chat_message(
                message_type="learning_summary",
                content=summary_content
            )
            self._logger.info(f"[{self.get_name()}] Posted learning summary.")
            # Reset state after summarizing
            self.heuristics.clear()
            self._processed_success_count = 0
        except Exception as e:
            self._logger.exception(f"[{self.get_name()}] Failed to post learning summary:")

    async def shutdown(self):
        """Optional cleanup actions on shutdown."""
        self._logger.info(f"[{self.get_name()}] Shutdown complete. Processed {self._processed_success_count} success messages.")
        # Log final heuristics (optional)
        # self._logger.debug(f"Final Heuristics: {json.dumps(dict(self.heuristics), default=str, indent=2)}") 