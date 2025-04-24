import logging
import re
import json  # Import json module
from typing import Dict, Any, Optional, List, Tuple
import inspect
from pathlib import Path
import time

# <<< Import base and decorator >>>
from .base import FragmentContext # Keep FragmentContext if needed
from .manager_fragment import ManagerFragment # <<< UPDATED IMPORT
from ..core.context import _ToolExecutionContext # Keep this for tool context
from .registry import fragment
from ..core.llm_interface import LLMInterface # Import LLMInterface if not already
from ..core.tool_registry import ToolRegistry, ToolInfo # Use ..core
# from a3x.core.logging_config import get_logger # <<< REMOVE INCORRECT IMPORT

# logger = get_logger(__name__) # <<< OLD INCORRECT USAGE
logger = logging.getLogger(__name__) # <<< CORRECT USAGE: Use standard logging module

# --- Define Skills Managed by this Manager ---
# This can be defined here or potentially derived if skills register their manager
FILE_OPS_SKILLS = [
    "read_file",
    "write_file",
    "list_directory",
    "append_to_file",
    "delete_path",
    "create_directory",
]

# <<< Apply the decorator >>>
@fragment(
    name="FileOpsManager",
    description="Coordinates file operations by selecting and executing the appropriate file skill.",
    category="Management",
    managed_skills=FILE_OPS_SKILLS
)
class FileOpsManager(ManagerFragment): # Inherit from ManagerFragment
    """Manager Fragment responsible for handling file operations."""

    # MANAGED_SKILLS = FILE_OPS_SKILLS # Set as class attribute for potential discovery fallback

    # <<< ADDED: Implement the abstract method >>>
    def get_purpose(self) -> str:
        """Returns a description of the fragment's purpose."""
        # Reuse the description from the decorator metadata if possible, or define it here.
        # Assuming metadata is accessible or just hardcoding based on decorator
        return "Coordinates file operations by selecting and executing the appropriate file skill."

    async def execute(self, context: FragmentContext, sub_task: str) -> Dict[str, Any]:
        """
        Parses the sub_task to determine the file operation and arguments,
        then calls the appropriate FileManagerSkill method.
        """
        logger = context.logger
        tool_registry = context.tool_registry
        llm_interface = context.llm_interface
        # <<< FIX: Access initial_objective directly >>>
        original_objective = context.shared_task_context.initial_objective or 'Not specified' # Get the original objective

        logger.info(f"FileOpsManager received sub-task: {sub_task}")
        logger.debug(f"Original task objective: {original_objective}") # Log original objective

        # 1. Determine which file skill to use (LLM call)
        # (This assumes tool_registry has get_schema or similar)
        managed_skills = ['read_file', 'write_file', 'list_directory', 'append_to_file', 'delete_path', 'create_directory']
        skill_schemas = {name: tool_registry.get_tool_details(name) for name in managed_skills if tool_registry.get_tool_details(name)}
        
        prompt = f"""
Given the user request: '{sub_task}'

Which of the following file operation skills is the most appropriate?
Available Skills:
{json.dumps(skill_schemas, indent=2)}

Respond ONLY with a JSON object containing the skill name: {{"skill_name": "chosen_skill"}}"
"""
        
        messages = [
            {"role": "system", "content": "You are an expert at choosing the correct file operation tool."},
            {"role": "user", "content": prompt}
        ]

        chosen_skill_name = None
        try:
            response_str = ""
            # <<< FIX: Revert to using async call_llm with stream=False >>>
            async for chunk in llm_interface.call_llm(messages=messages, stream=False, temperature=0.1):
                response_str += chunk
            
            # <<< FIX: Improve JSON extraction regex >>>
            # json_match = re.search(r'{\\s*.*\\s*}', response_str, re.DOTALL)
            json_match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", response_str, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                # Extract the actual JSON string from the first non-empty group
                json_str = next((group for group in json_match.groups() if group), None)
                if json_str:
                    decision = json.loads(json_str)
                    chosen_skill_name = decision.get('skill_name')
                    if chosen_skill_name not in managed_skills:
                        logger.warning(f"LLM chose an invalid skill '{chosen_skill_name}'. Available: {managed_skills}")
                        chosen_skill_name = None # Reset if invalid
                else:
                    logger.warning(f"Regex matched, but failed to extract JSON group from LLM: {response_str}")
            else:
                logger.warning(f"Could not extract skill choice JSON from LLM: {response_str}")
                # chosen_skill_name remains None

        except Exception as e:
            logger.error(f"Error during LLM call for skill choice: {e}")
            # chosen_skill_name remains None, proceed to fallback

        # Fallback or error if LLM fails or returns invalid skill
        if not chosen_skill_name:
            logger.info("LLM did not provide a valid skill choice. Attempting keyword fallback...")
            # Simple fallback: check for keywords in the sub_task
            sub_task_lower = sub_task.lower()
            if "read" in sub_task_lower or "get content" in sub_task_lower:
                chosen_skill_name = "read_file"
            elif "write" in sub_task_lower or "save" in sub_task_lower or "create file" in sub_task_lower: # Added "create file"
                chosen_skill_name = "write_file"
            elif "list" in sub_task_lower or "show files" in sub_task_lower:
                chosen_skill_name = "list_directory"
            elif "append" in sub_task_lower or "add to file" in sub_task_lower:
                chosen_skill_name = "append_to_file"
            elif "delete" in sub_task_lower or "remove" in sub_task_lower:
                chosen_skill_name = "delete_path"
            elif "create directory" in sub_task_lower or "make folder" in sub_task_lower:
                chosen_skill_name = "create_directory"
            # else: # No keyword match
            #     chosen_skill_name remains None

        # If still no skill chosen after LLM and fallback, return error
        if not chosen_skill_name:
            logger.error(f"Could not determine appropriate file skill via LLM or fallback for: {sub_task}")
            return {"status": "error", "message": "Could not determine appropriate file skill."}
        
        logger.info(f"Chosen file skill: {chosen_skill_name}")

        # 2. Extract arguments for the chosen skill (LLM call)
        skill_schema = tool_registry.get_tool_details(chosen_skill_name)
        if not skill_schema:
            return {"status": "error", "message": f"Could not retrieve schema for skill '{chosen_skill_name}'."}

        # Get only parameter properties (excluding description etc.)
        params_schema = skill_schema.get('parameters', {}).get('properties', {})
        # Filter out self/ctx parameters if they exist in the schema explicitly
        relevant_params_schema = {k: v for k, v in params_schema.items() if k not in ('self', 'ctx')}

        # <<< MODIFICATION: Use the ORIGINAL OBJECTIVE for argument extraction >>>
        arg_prompt = f"""
Given the original user objective: '{original_objective}'
And the chosen file skill: '{chosen_skill_name}' which should be used to achieve part of the objective.

Extract the arguments for the '{chosen_skill_name}' skill based on its expected parameters:
{json.dumps(relevant_params_schema, indent=2)}

Infer the values from the original user objective. Pay close attention to required arguments like 'content' for writing files or 'path' for most operations.
If an argument (like 'content') is mentioned in the objective, extract it accurately.
If an argument is not mentioned and not required, omit it.
If a required argument is not mentioned, you *must* try to infer a sensible default or ask for clarification (but for now, return an empty JSON if required args are missing).

Respond ONLY with a valid JSON object containing the extracted arguments (key: value).
Example for 'write_file' from objective "save 'hello' to report.txt": {{"path": "report.txt", "content": "hello"}}
Example for 'read_file' from objective "read config.json": {{"path": "config.json"}}
Example for 'delete_path' from objective "delete temp.log": {{"path": "temp.log"}}
Example for 'list_directory' from objective "list files in /data": {{"path": "/data"}}
"""
        
        arg_messages = [
            {"role": "system", "content": "You are an expert at extracting structured arguments for file operations from a user's overall objective."},
            {"role": "user", "content": arg_prompt}
        ]

        extracted_args = {}
        try:
            response_str = ""
            # <<< FIX: Revert to using async call_llm with stream=False >>>
            async for chunk in llm_interface.call_llm(messages=arg_messages, stream=False, temperature=0.0):
                 response_str += chunk
            
            # <<< FIX: Improve JSON extraction regex >>>
            # json_match = re.search(r'{\\s*.*\\s*}', response_str, re.DOTALL)
            json_match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", response_str, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                # Extract the actual JSON string from the first non-empty group
                json_str = next((group for group in json_match.groups() if group), None)
                if json_str:
                    extracted_args = json.loads(json_str)
                    logger.info(f"Extracted arguments for {chosen_skill_name}: {extracted_args}")
                else:
                    logger.warning(f"Regex matched, but failed to extract argument JSON group from LLM response: {response_str}")
            else:
                logger.warning(f"Could not extract arguments JSON from LLM response: {response_str}")
                # Optionally attempt regex fallback here for common patterns like paths

        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON arguments for {chosen_skill_name}: {json_err}. Response was: {response_str}")
            # Decide: error out or proceed with empty args?
            return {"status": "error", "message": f"LLM argument extraction failed (JSON decode)."}
        except Exception as e:
            logger.error(f"Error during LLM call for argument extraction: {e}")
            return {"status": "error", "message": f"Failed to extract arguments for {chosen_skill_name}: {e}"}

        # 3. Execute the chosen skill with extracted arguments
        logger.info(f"Executing skill '{chosen_skill_name}' with args: {extracted_args}")

        try:
            # Ensure tool_executor exists in context
            if not hasattr(context, 'tool_executor'):
                logger.error(f"ToolExecutor not found in context.")
                return {"status": "error", "message": "Internal error: ToolExecutor missing from context."}
            
            # Use the tool_executor instance from the context
            tool_result_wrapped = await context.tool_executor.execute_tool(
                tool_name=chosen_skill_name,
                tool_input=extracted_args, # Changed action_input to tool_input
                context=context # Pass FragmentContext for the skill
            )

            # Unwrap the result from execute_tool
            final_result = tool_result_wrapped.get('result', {})
            metrics = tool_result_wrapped.get('metrics', {})
            logger.info(f"Skill '{chosen_skill_name}' executed. Status: {final_result.get('status', 'N/A')}. Metrics: {metrics}")

            # Ensure the result has a status, default to error if missing from skill
            if 'status' not in final_result:
                logger.warning(f"Skill '{chosen_skill_name}' result missing 'status'. Defaulting to error.")
                final_result['status'] = 'error'
                final_result['message'] = final_result.get('message', f"Skill {chosen_skill_name} did not return a status.")
            
            return final_result

        except Exception as e:
            logger.exception(f"Error executing skill '{chosen_skill_name}':")
            return {"status": "error", "message": f"Failed to execute skill '{chosen_skill_name}': {e}"}