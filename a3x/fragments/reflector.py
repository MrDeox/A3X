import asyncio
import logging
import json
from typing import Dict, Any, Optional
from collections import defaultdict
from pathlib import Path

from .base import BaseFragment, FragmentDef
from a3x.core.context import FragmentContext

logger = logging.getLogger(__name__)

# Configuration
REFLECTION_INTERVAL = 50 # Analyze and reflect every N messages processed
MIN_ACTION_COUNT_FOR_REPORT = 3
MIN_FRAGMENT_ERROR_COUNT_FOR_REPORT = 2

class ReflectorFragment(BaseFragment):
    """Observes system results and suggestions, reflecting periodically on performance patterns."""

    def __init__(self, fragment_def: FragmentDef, tool_registry=None):
        super().__init__(fragment_def, tool_registry)
        self._fragment_context: Optional[FragmentContext] = None
        self._processed_message_count = 0
        # --- Statistics --- 
        # action_stats[(action, status)] = count
        self.action_stats: defaultdict[tuple, int] = defaultdict(int)
        # fragment_errors[fragment_name] = error_count
        self.fragment_errors: defaultdict[str, int] = defaultdict(int)
        # target_type_stats[(target_type, status)] = count
        self.target_type_stats: defaultdict[tuple, int] = defaultdict(int)
        # suggestion_tracker[action] = count
        self.suggestion_tracker: defaultdict[str, int] = defaultdict(int)
        self._logger.info(f"[{self.get_name()}] Initialized. Will reflect every {REFLECTION_INTERVAL} messages.")

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        """Returns a description of this fragment's purpose."""
        return "Observes system activity (results and suggestions) to periodically generate reflective summaries about successful actions, fragment errors, and target type performance."

    def set_context(self, context: FragmentContext):
        """Receives the context needed for the Reflector."""
        # The context passed from the runner loop IS the SharedTaskContext
        shared_context = context
        super().set_context(shared_context) # Call parent's set_context
        self._fragment_context = shared_context # Store the shared context
        self._logger.info(f"[{self.get_name()}] Context received.")

    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """Processes incoming messages to update internal statistics for reflection."""
        if self._fragment_context is None:
            self.set_context(context)

        self._processed_message_count += 1
        msg_type = message.get("type")
        sender = message.get("sender", "Unknown")
        content = message.get("content", {})

        # --- Update stats based on message type --- 
        if msg_type.endswith("_RESULT") and isinstance(content, dict) and sender != "Unknown":
            self._process_result_message(sender, content)
        elif msg_type == "ARCHITECTURE_SUGGESTION" and isinstance(content, dict):
            self._process_suggestion_message(content)
        elif msg_type == "learning_summary" and isinstance(content, dict):
            await self._process_learning_summary(content)

        # --- Trigger Reflection Periodically ---
        if self._processed_message_count % REFLECTION_INTERVAL == 0:
            await self._generate_reflection()

    def _process_result_message(self, sender: str, content: dict):
        """Update stats based on a result message."""
        status = content.get("status")
        action = content.get("original_action")
        target = content.get("target") or content.get("target_fragment")
        
        # Try to get action/target from original directive if nested
        original_directive = content.get("original_directive")
        if not action and isinstance(original_directive, dict):
            action = original_directive.get("action")
        if not target and isinstance(original_directive, dict):
            target = original_directive.get("target")

        if action and status:
            self.action_stats[(action, status)] += 1

            # Track fragment errors
            if status in ["error", "failed"]:
                self.fragment_errors[sender] += 1

            # Track target type stats
            if target:
                target_type = self._get_target_type(target)
                self.target_type_stats[(target_type, status)] += 1
            
    def _process_suggestion_message(self, content: dict):
        """Update stats based on a suggestion message."""
        if content.get("type") == "directive":
            action = content.get("action")
            if action:
                 self.suggestion_tracker[action] += 1
    
    def _get_target_type(self, target: str) -> str:
        """Extracts a simple type from the target path (e.g., extension or 'dir')."""
        try:
            p = Path(target)
            if p.suffix:
                return f"extension:{p.suffix}"
            # Basic check if it looks like a directory path (heuristic)
            if not p.name or '/.' in target or target.endswith('/'): 
                return "type:directory"
            return "type:file_no_ext"
        except Exception:
            return "type:invalid_path"

    async def _process_learning_summary(self, content: dict):
        """Processes a learning summary message and generates an immediate reflection."""
        self._logger.info(f"[{self.get_name()}] Received learning_summary: {content}")
        plan_id = content.get("plan_id")
        topic = content.get("topic")
        summary = content.get("summary") # Summary provided by Consolidator

        reflection_text = "Default reflection on learning summary."
        if topic and topic != "unknown_topic":
            # Simple reflection based on topic success
            reflection_text = f"Reflection: Learning about '{topic}' (Plan: {plan_id}) was successful. " \
                              f"This suggests that pursuing knowledge in areas like '{topic}' is a viable path. " \
                              f"Original summary stated: \"{summary}\""
        elif plan_id:
             reflection_text = f"Reflection: Plan '{plan_id}' completed successfully, but the specific topic was unclear. " \
                               f"Success is positive, but understanding *why* is key. Original summary: \"{summary}\""
        else:
            reflection_text = f"Reflection: A plan completed successfully, but key details (plan_id, topic) were missing from the summary: {content}"

        self._logger.info(f"[{self.get_name()}] Generated reflection based on learning summary: {reflection_text}")

        reflection_content = {
            "reflection_text": reflection_text,
            "source_type": "learning_summary",
            "source_content": content # Include the original summary for context
        }

        try:
            await self.post_chat_message(
                message_type="reflection_result", # Use a distinct type for this kind of reflection
                content=reflection_content
            )
            self._logger.info(f"[{self.get_name()}] Posted reflection_result based on learning summary for plan '{plan_id}'.")

            # <<< NEW: Trigger Knowledge Synthesis >>>
            if topic and plan_id: # Only synthesize if we have key info
                synthesize_content = {
                    "topic": topic,
                    "successful_plan_id": plan_id,
                    "reflection": reflection_text, # Send the generated reflection text
                    "original_summary": content # Include original summary for full context
                }
                await self.post_chat_message(
                    message_type="synthesize_knowledge",
                    content=synthesize_content,
                    target_fragment="KnowledgeSynthesizer" # Target the new fragment
                )
                self._logger.info(f"[{self.get_name()}] Sent synthesize_knowledge directive to KnowledgeSynthesizer for topic '{topic}'.")
            else:
                self._logger.warning(f"[{self.get_name()}] Skipping knowledge synthesis due to missing topic or plan_id in learning summary.")
            # <<< END NEW >>>

        except Exception as e:
            self._logger.error(f"[{self.get_name()}] Failed to post reflection_result or synthesize_knowledge: {e}", exc_info=True)

    async def _generate_reflection(self):
        """Analyzes statistics and generates a reflective summary."""
        if not self._fragment_context:
             self._logger.error(f"[{self.get_name()}] Cannot generate reflection: Context not set.")
             return

        report_lines = [f"Reflection Report (last ~{REFLECTION_INTERVAL} messages):"]

        # 1. Successful Actions
        successful_actions = defaultdict(int)
        for (action, status), count in self.action_stats.items():
            if status == "success":
                successful_actions[action] += count
        
        report_lines.append(f"\n[Successful Actions (Count >= {MIN_ACTION_COUNT_FOR_REPORT})]")
        found_successful = False
        sorted_success = sorted(successful_actions.items(), key=lambda item: item[1], reverse=True)
        for action, count in sorted_success:
             if count >= MIN_ACTION_COUNT_FOR_REPORT:
                  report_lines.append(f"  - '{action}': {count} successes")
                  found_successful = True
        if not found_successful: report_lines.append("  (None met threshold)")

        # 2. Fragment Errors
        report_lines.append(f"\n[Fragment Errors (Count >= {MIN_FRAGMENT_ERROR_COUNT_FOR_REPORT})]")
        found_errors = False
        sorted_errors = sorted(self.fragment_errors.items(), key=lambda item: item[1], reverse=True)
        for fragment, count in sorted_errors:
             if count >= MIN_FRAGMENT_ERROR_COUNT_FOR_REPORT:
                  report_lines.append(f"  - '{fragment}': {count} errors/failures")
                  found_errors = True
        if not found_errors: report_lines.append("  (None met threshold)")

        # 3. Target Type Success Rate (Simple Example)
        report_lines.append("\n[Target Type Success Rate (basic)]")
        target_type_success = defaultdict(lambda: {'success': 0, 'total': 0})
        for (target_type, status), count in self.target_type_stats.items():
            target_type_success[target_type]['total'] += count
            if status == 'success':
                 target_type_success[target_type]['success'] += count
        
        found_target_rates = False
        for target_type, counts in sorted(target_type_success.items()):
             total = counts['total']
             if total >= MIN_ACTION_COUNT_FOR_REPORT: # Reuse threshold
                 success = counts['success']
                 rate = (success / total) * 100 if total > 0 else 0
                 report_lines.append(f"  - '{target_type}': {success}/{total} successes ({rate:.1f}%)")
                 found_target_rates = True
        if not found_target_rates: report_lines.append("  (Not enough data for target types)")

        # 4. Suggestions Made (Optional)
        # report_lines.append("\n[Suggestions Made]")
        # ... (add logic if needed)

        reflection_text = "\n".join(report_lines)
        self._logger.info(f"[{self.get_name()}] Posting reflection:\n{reflection_text}")

        reflection_content = {
            "reflection_summary": reflection_text,
            "stats_snapshot": { # Include raw stats if useful downstream
                 "action_stats": dict(self.action_stats),
                 "fragment_errors": dict(self.fragment_errors),
                 "target_type_stats": dict(self.target_type_stats)
             }
        }

        # Reset stats after reflection?
        # self.action_stats.clear()
        # self.fragment_errors.clear()
        # self.target_type_stats.clear()

        if reflection_text:
            self._logger.info("Generated reflection summary.")
            try:
                await self.post_chat_message(
                    message_type="reflection",
                    content={"reflection_summary": reflection_text}
                )
                self._logger.info("Posted reflection summary.")
                # Reset state after reflecting
                self._processed_message_count = 0 # Reset count relative to last reflection
            except Exception as e:
                self._logger.error(f"Failed to post reflection: {e}", exc_info=True)
        else:
            self._logger.info("No reflection generated from current message window.")

    async def shutdown(self):
        """Optional cleanup actions on shutdown."""
        self._logger.info(f"[{self.get_name()}] Shutdown complete. Processed {self._processed_message_count} total messages.")

# <<< ADD FRAGMENT DEFINITION >>>
ReflectorFragmentDef = FragmentDef(
    name="Reflector",
    description="Analyzes recent messages and system state to generate reflections.",
    fragment_class=ReflectorFragment,
    skills=["refletir_mensagens"],
    managed_skills=["refletir_mensagens"],
    prompt_template="Analise as mensagens recentes e o estado do sistema para gerar reflexões."
)
# <<< END FRAGMENT DEFINITION >>> 