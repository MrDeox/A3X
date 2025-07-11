import asyncio
import logging
import json
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional

from .base import BaseFragment, FragmentDef
from a3x.core.context import FragmentContext

logger = logging.getLogger(__name__)

class KnowledgeSynthesizerFragment(BaseFragment):
    """
    Receives 'synthesize_knowledge' directives triggered by successful reflections
    and attempts to generate new cognitive artifacts (skills, plans, knowledge modules).
    """

    def __init__(self, fragment_def: FragmentDef, tool_registry=None):
        super().__init__(fragment_def, tool_registry)
        self._fragment_context: Optional[FragmentContext] = None
        self._logger.info(f"[{self.get_name()}] Initialized. Ready to synthesize knowledge.")

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        """Returns a description of this fragment's purpose."""
        return "Listens for 'synthesize_knowledge' directives and attempts to generate new skills, plans, or knowledge modules based on successful learning cycles."

    def set_context(self, context: FragmentContext):
        """Receives the context needed for the synthesizer."""
        shared_context = context
        super().set_context(shared_context) 
        self._fragment_context = shared_context 
        self._logger.info(f"[{self.get_name()}] Context received.")

    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """Processes incoming directives, specifically 'synthesize_knowledge'."""
        if self._fragment_context is None:
            self.set_context(context)

        msg_type = message.get("type")
        sender = message.get("sender", "Unknown")
        content = message.get("content")

        if msg_type == "synthesize_knowledge" and isinstance(content, dict):
            self._logger.info(f"[{self.get_name()}] Received synthesize_knowledge directive from {sender}.")
            
            topic = content.get("topic")
            plan_id = content.get("successful_plan_id")
            reflection = content.get("reflection")
            original_summary = content.get("original_summary") 

            if not all([topic, plan_id, reflection, original_summary]):
                self._logger.error(f"[{self.get_name()}] Missing required fields in synthesize_knowledge message: {content}")
                return

            self._logger.info(f"  > Synthesizing knowledge for Topic: {topic}")
            self._logger.info(f"  > Based on Plan ID: {plan_id}")
            self._logger.info(f"  > Using Reflection: {reflection[:100]}...")
            
            topic_slug = topic.replace(' ', '_').lower() # For filenames/keys

            # --- Perform Synthesis Actions --- 
            await self._synthesize_plan_template(topic_slug, original_summary)
            await self._suggest_skill_code(topic_slug, topic, reflection)
            await self._create_knowledge_document(topic_slug, topic, reflection, original_summary)
            await self._suggest_strategy(topic_slug, topic)

    async def _synthesize_plan_template(self, topic_slug: str, original_summary: dict):
        """Generates a plan template from the successful plan and saves it to ContextStore.""" 
        if not self._fragment_context or not self._fragment_context.store:
            self._logger.error(f"[{self.get_name()}] Cannot save plan template: ContextStore not available.")
            return
            
        actions = original_summary.get("actions", []) # Get actions from the original summary
        if not actions:
            self._logger.warning(f"[{self.get_name()}] No actions found in original summary for topic '{topic_slug}' to create template.")
            return
            
        plan_template = {
            "objective_template": f"Learn about {topic_slug}", # Generic objective
            "actions": actions # Use the exact successful actions
        }
        template_key = f"plan_template:{topic_slug}"
        
        try:
            await self._fragment_context.store.set(template_key, plan_template)
            self._logger.info(f"[{self.get_name()}] Saved plan template to ContextStore: Key='{template_key}'")
        except Exception as e:
            self._logger.error(f"[{self.get_name()}] Failed to save plan template '{template_key}' to ContextStore: {e}", exc_info=True)

    async def _suggest_skill_code(self, topic_slug: str, topic: str, reflection: str):
        """Generates example skill code and posts it as a suggestion."""
        skill_name = f"learn_{topic_slug}"
        # Basic code generation using textwrap and f-string
        suggested_code = textwrap.dedent(f"""
        import logging
        from typing import Dict, Any
        
        # Potential Skill generated by KnowledgeSynthesizer
        # Based on successful learning of: {topic}
        # Reflection: {reflection[:150]}...

        logger = logging.getLogger(__name__)

        async def {skill_name}(**kwargs) -> Dict[str, Any]:
            \"\"\"Attempts to encapsulate the learned process for {topic}.\"\"\"
            logger.info(f"Executing synthesized skill: {skill_name}")
            # TODO: Implement actual logic based on successful plan steps
            # Placeholder implementation:
            result_summary = f'Successfully executed synthesized skill for {topic}.'
            key_points = [
                f'Synthesized key point 1 about {topic}',
                f'Synthesized key point 2 about {topic}'
            ]
            logger.info(f"Skill {skill_name} completed.")
            return {{
                "status": "success",
                "data": {{
                    "summary": result_summary,
                    "key_points": key_points
                }}
            }}
        """)
        
        message_content = {
            "skill_name": skill_name,
            "suggested_code": suggested_code,
            "source_topic": topic,
            "source_reflection": reflection
        }
        
        try:
            await self.post_chat_message(
                message_type="suggested_skill_code",
                content=message_content,
                # target_fragment="Architect" # Or broadcast?
            )
            self._logger.info(f"[{self.get_name()}] Posted suggested_skill_code for '{skill_name}'.")
        except Exception as e:
            self._logger.error(f"[{self.get_name()}] Failed to post suggested_skill_code for '{skill_name}': {e}", exc_info=True)

    async def _create_knowledge_document(self, topic_slug: str, topic: str, reflection: str, original_summary: dict):
        """Generates a markdown document summarizing the knowledge."""
        if not self.tool_registry:
            self._logger.error(f"[{self.get_name()}] Cannot create knowledge document: ToolRegistry not available.")
            return

        # Ensure knowledge_base directory exists
        knowledge_base_dir = "knowledge_base"
        try:
            # Use FileManagerSkill to create directory
            fm_skill = self.tool_registry.get_tool("FileManagerSkill")
            if fm_skill:
                 await fm_skill.create_directory(directory_path=knowledge_base_dir, exist_ok=True)
                 self._logger.debug(f"[{self.get_name()}] Ensured directory exists: {knowledge_base_dir}")
            else:
                 self._logger.error(f"[{self.get_name()}] FileManagerSkill not found in registry.")
                 return # Cannot proceed without file manager
        except Exception as e:
            self._logger.error(f"[{self.get_name()}] Failed to ensure directory '{knowledge_base_dir}' exists: {e}", exc_info=True)
            return # Cannot proceed if directory fails
            
        file_path = f"{knowledge_base_dir}/{topic_slug}.md"
        actions = original_summary.get("actions", [])
        plan_id = original_summary.get("plan_id", "N/A")
        
        md_content = f"# Knowledge Summary: {topic}\n\n"
        md_content += f"**Source Plan ID:** {plan_id}\n\n"
        md_content += f"**Reflection:**\n{reflection}\n\n"
        md_content += f"**Successful Plan Actions:**\n"
        if actions:
            for i, action in enumerate(actions):
                action_type = action.get('type', '')
                action_name = action.get('action', '')
                skill = action.get('skill', 'N/A')
                params = action.get('parameters', '{}')
                message = action.get('message', '(No message)')
                md_content += f"  {i+1}. **{action_name}** (Type: {action_type}, Skill: {skill})\n"
                md_content += f"     Params: `{json.dumps(params)}`\n"
                md_content += f"     Message: {message}\n"
        else:
            md_content += "  (No actions found in summary)\n"
            
        try:
            # Use write_file tool
            write_tool = self.tool_registry.get_tool("write_file") # Get the specific tool function
            if write_tool:
                 result = await write_tool(file_path=file_path, content=md_content)
                 if result.get("status") == "success":
                     self._logger.info(f"[{self.get_name()}] Saved knowledge document: {file_path}")
                 else:
                      self._logger.error(f"[{self.get_name()}] Failed to write knowledge document '{file_path}'. Tool status: {result.get('status')}")
            else:
                 self._logger.error(f"[{self.get_name()}] write_file tool not found in registry.")
        except Exception as e:
            self._logger.error(f"[{self.get_name()}] Error writing knowledge document '{file_path}': {e}", exc_info=True)

    async def _suggest_strategy(self, topic_slug: str, topic: str):
        """Formulates a simple strategy suggestion based on the successful topic."""
        # Simple logic to determine next topic (mirroring Planner for now)
        next_topic = "unknown"
        if topic == "python basics":
            next_topic = "intermediate python concepts"
        elif topic == "data structures":
            next_topic = "algorithm analysis"
        # Add more rules or make this smarter
        
        if next_topic == "unknown":
             self._logger.info(f"[{self.get_name()}] No defined next topic for '{topic}'. Skipping strategy suggestion.")
             return

        suggestion_content = {
             "type": "strategy_suggestion", # Keep type consistent maybe?
             "origin": self.get_name(),
             "topic": topic,
             "suggestion": f"Based on the synthesized knowledge from successfully learning '{topic}', recommend expanding focus to '{next_topic}'.",
             "next_topic": next_topic
        }
        
        try:
            await self.post_chat_message(
                message_type="strategy_suggestion", # Or maybe ARCHITECTURE_SUGGESTION with a specific subtype?
                content=suggestion_content,
                target_fragment="Strategist" # Target the Strategist
            )
            self._logger.info(f"[{self.get_name()}] Posted strategy_suggestion to Strategist regarding next topic: '{next_topic}'.")
        except Exception as e:
            self._logger.error(f"[{self.get_name()}] Failed to post strategy_suggestion for next topic '{next_topic}': {e}", exc_info=True)

    async def shutdown(self):
        """Optional cleanup actions on shutdown."""
        self._logger.info(f"[{self.get_name()}] Shutdown complete.")

# Define the FragmentDef
KnowledgeSynthesizerFragmentDef = FragmentDef(
    name="KnowledgeSynthesizer",
    description="Synthesizes new knowledge artifacts (skills, plans, etc.) from successful learning cycles.",
    fragment_class=KnowledgeSynthesizerFragment,
    category="Learning", # Example category
    skills=["FileManagerSkill", "write_file"] # Add skills needed for knowledge doc generation
) 