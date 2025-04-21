import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Use the fragment decorator if available in your project structure
# from a3x.fragments.registry import fragment 
from .base import BaseFragment, FragmentDef
from a3x.core.context import FragmentContext
from a3x.core.tool_registry import ToolRegistry

# Attempt to get PROJECT_ROOT, fallback if needed
try:
    from a3x.core.config import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    logging.getLogger(__name__).warning(f"Could not import PROJECT_ROOT from core.config, using fallback: {PROJECT_ROOT}")

logger = logging.getLogger(__name__)

# @fragment( # Uncomment if using the decorator
#     name="Architect",
#     description="Listens for 'create_fragment' directives and generates/writes the corresponding fragment code.",
#     category="Meta",
#     skills=["generate_module_from_directive", "write_file"],
# )
class ArchitectFragment(BaseFragment):
    """Listens for 'create_fragment' directives and attempts to generate and write the new fragment code."""

    def __init__(self, fragment_def: FragmentDef, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(fragment_def, tool_registry)
        self._logger.info(f"[{self.get_name()}] Initialized.")

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        """Returns a description of this fragment's purpose."""
        return "Listens for 'create_fragment' directives and uses generation and file writing skills to create new fragments within the system."

    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """Handles incoming chat messages, looking for 'create_fragment' directives."""
        msg_type = message.get("type")
        sender = message.get("sender")
        content = message.get("content")

        # Check if it's an architecture suggestion message containing a create_fragment directive
        if (
            msg_type == "ARCHITECTURE_SUGGESTION"
            and isinstance(content, dict)
            and content.get("type") == "directive"
            and content.get("action") == "create_fragment"
        ):
            directive = content
            target_path_str = directive.get("target") # Expected relative path like a3x/fragments/new_fragment.py
            directive_message = directive.get("message") # The prompt describing the fragment

            if not target_path_str or not directive_message:
                self._logger.warning(f"[{self.get_name()}] Received invalid 'create_fragment' directive (missing target or message): {directive}")
                return

            self._logger.info(f"[{self.get_name()}] Received 'create_fragment' directive from {sender} for target: {target_path_str}")
            await self._handle_create_fragment_directive(directive, context)
        else:
            # Log other messages at debug level if needed
            # self._logger.debug(f"[{self.get_name()}] Ignoring message type '{msg_type}' from {sender}.")
            pass

    async def _handle_create_fragment_directive(self, directive: Dict[str, Any], context: FragmentContext):
        """Handles the logic for generating and writing a new fragment based on a directive."""
        target_path_str = directive.get("target")
        directive_message = directive.get("message")
        result_status = "error"
        result_summary = "Failed to create fragment."
        result_details = ""

        generated_code = None
        generation_status = "unknown"
        write_status = "unknown"

        try:
            # 1. Generate the fragment code
            self._logger.info(f"[{self.get_name()}] Attempting to generate fragment code for: {target_path_str}")
            # We reuse 'generate_module_from_directive' but the prompt guides it towards fragment creation
            generate_result = await self._tool_registry.execute_tool(
                "generate_module_from_directive",
                {
                    # Pass the descriptive message as the core directive/prompt
                    "directive": directive_message,
                    # The target path might help the generation context, but the primary input is the message
                    "target_path": str(Path(PROJECT_ROOT) / target_path_str)
                }
            )

            generation_status = generate_result.get("status", "error")

            if generation_status != "success":
                result_summary = f"Fragment generation skill failed."
                result_details = str(generate_result.get("data", {}).get("message", "No details provided."))
                self._logger.error(f"[{self.get_name()}] {result_summary} Target: {target_path_str}. Details: {result_details}")
            else:
                # Use 'code' key based on the skill's expected output (adjust if needed)
                generated_code = generate_result.get("data", {}).get("code")
                if not generated_code:
                    result_summary = "Fragment generation skill succeeded but returned no code."
                    result_details = str(generate_result)
                    self._logger.error(f"[{self.get_name()}] {result_summary} Target: {target_path_str}")
                else:
                    self._logger.info(f"[{self.get_name()}] Successfully generated code for {target_path_str}. Length: {len(generated_code)}")

                    # 2. Write the generated code to the file
                    # Ensure the target path is relative for the write_file skill
                    if Path(target_path_str).is_absolute():
                         self._logger.warning(f"[{self.get_name()}] Directive target '{target_path_str}' was absolute. Attempting to use relative path.")
                         try:
                              target_path_relative = str(Path(target_path_str).relative_to(PROJECT_ROOT))
                         except ValueError:
                              result_summary = "Generated code but target path is absolute and outside project root."
                              result_details = f"Target: {target_path_str}, Project Root: {PROJECT_ROOT}"
                              self._logger.error(f"[{self.get_name()}] {result_summary}")
                              generated_code = None # Prevent writing
                    else:
                        target_path_relative = target_path_str

                    if generated_code:
                        self._logger.info(f"[{self.get_name()}] Attempting to write fragment code to: {target_path_relative}")
                        write_result = await self._tool_registry.execute_tool(
                            "write_file",
                            {
                                "file_path": target_path_relative,
                                "content": generated_code,
                                "overwrite": False # Avoid overwriting existing fragments accidentally
                            }
                        )
                        write_status = write_result.get("status", "error")

                        if write_status == "success":
                            result_status = "success"
                            result_summary = f"Successfully created new fragment: {target_path_relative}"
                            result_details = f"Generated based on directive: {directive_message[:100]}..."
                            self._logger.info(f"[{self.get_name()}] {result_summary}")

                            # --- Post suggestion to register the new fragment ---
                            try:
                                # Derive a potential name from the filename
                                fragment_name_from_path = Path(target_path_relative).stem.replace('_', ' ').title().replace(' ', '')
                                
                                registration_directive_content = {
                                    "type": "directive",
                                    "action": "register_fragment",
                                    "name": fragment_name_from_path, # Best guess for name
                                    "path": target_path_relative,    # Path where it was created
                                    # "skills": [], # TODO: Skills should ideally be provided in the original create_fragment directive
                                }
                                await self.post_chat_message(
                                    message_type="architecture_suggestion",
                                    content=registration_directive_content
                                )
                                self._logger.info(f"[{self.get_name()}] Posted 'register_fragment' suggestion for {target_path_relative}")
                            except Exception as post_e:
                                # Log error but don't change the overall success status of the creation
                                self._logger.error(f"[{self.get_name()}] Failed to post register_fragment suggestion for {target_path_relative}: {post_e}")
                            # ---------------------------------------------------------
                        else:
                            result_summary = f"Generated code but failed to write fragment file."
                            result_details = str(write_result.get("data", {}).get("message", "Write skill failed."))
                            self._logger.error(f"[{self.get_name()}] {result_summary} Target: {target_path_relative}. Details: {result_details}")

        except Exception as e:
            self._logger.exception(f"[{self.get_name()}] Unexpected error handling create_fragment directive for {target_path_str}:")
            result_status = "error"
            result_summary = "Unexpected error during fragment creation."
            result_details = str(e)

        # 3. Send result back via chat
        await self.broadcast_result_via_chat(context, directive, result_status, result_summary, result_details)

    async def broadcast_result_via_chat(self, context: FragmentContext, original_directive: Dict, status: str, summary: str, details: str):
        """Broadcasts the result of handling the create_fragment directive."""
        # Reusing refactor_result structure, but clearly indicating it's about fragment creation
        result_message_content = {
            "type": "architect_result", # Specific type for this fragment's results
            "status": status,
            "target": original_directive.get("target", "unknown"),
            "original_action": "create_fragment",
            "summary": summary,
            "details": details,
            "original_directive": original_directive
        }
        try:
            await self.post_chat_message(
                message_type="architect_result", # Use the specific type
                content=result_message_content
            )
            self._logger.info(f"[{self.get_name()}] Broadcasted architect_result for target '{result_message_content['target']}': {status}")
        except Exception as e:
            self._logger.error(f"[{self.get_name()}] Failed to broadcast architect_result via chat: {e}") 