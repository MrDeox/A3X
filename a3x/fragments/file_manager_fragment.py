import logging
import re
import json  # Import json module
from typing import Dict, Any, Optional, List, Tuple
import inspect
from pathlib import Path

# <<< Import base and decorator >>>
from .base import ManagerFragment, FragmentContext # Correct import from base
from ..core.context import _ToolExecutionContext # Keep this for tool context
from .registry import fragment
from ..core.llm_interface import LLMInterface # Import LLMInterface if not already
from ..core.tool_registry import ToolRegistry, ToolInfo # Use ..core
# from a3x.core.logging_config import get_logger # <<< REMOVE INCORRECT IMPORT
# from a3x.core.tool_executor import execute_tool # <<< REMOVE INCORRECT IMPORT

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

    async def execute(self, sub_task: str, context: FragmentContext) -> Dict[str, Any]:
        """
        Determines the appropriate file operation tool (skill) using an LLM based on the sub-task,
        then executes that tool.
        """
        # Use the logger from the passed context
        logger = context.logger 
        logger.info(f"[FileOpsManager] Received sub-task: {sub_task}...")

        # --- LLM Call to Select Tool ---
        tool_name = None
        action_input = None
        try:
            # Get schemas for managed skills
            managed_schemas = {}
            if hasattr(self, 'metadata') and self.metadata.managed_skills:
                for skill_name in self.metadata.managed_skills:
                    tool_schema = context.tool_registry.get_tool_details(skill_name)
                    if tool_schema:
                        # Exclude 'self' and 'ctx' etc.
                        params = tool_schema.get('parameters', {})
                        props = params.get('properties', {}).copy()
                        props.pop('self', None)
                        props.pop('ctx', None)
                        props.pop('resolved_path', None)
                        props.pop('original_path_str', None)
                        
                        cleaned_schema = tool_schema.copy()
                        if 'parameters' in cleaned_schema:
                            cleaned_schema['parameters'] = cleaned_schema['parameters'].copy()
                            cleaned_schema['parameters']['properties'] = props
                        
                        managed_schemas[skill_name] = cleaned_schema
                    else:
                        logger.warning(f"Could not retrieve schema for managed skill: {skill_name}")
            else:
                 logger.warning(f"Fragment {self.metadata.name if hasattr(self, 'metadata') else 'Unknown'} has no managed_skills in its metadata.")
            
            if not managed_schemas:
                 logger.error("No managed skill schemas found for FileOpsManager.")
                 return {"status": "error", "message": "FileOpsManager has no skills configured."}
            
            prompt = f"""
Given the sub_task: '{sub_task}'

And the available file operation tools this manager can use:
{json.dumps(managed_schemas, indent=2)} 

Choose the single best tool to accomplish the sub-task and determine the necessary arguments.
Respond ONLY with a JSON object containing 'tool_name' and 'action_input' (which itself is an object containing the arguments for the chosen tool).

Example Response:
{{
  "tool_name": "write_file",
  "action_input": {{
    "file_path": "path/to/output.txt",
    "content": "This is the content."
  }}
}}

If no tool is suitable, respond with: {{"tool_name": null, "action_input": null}}
"""
            llm_input = [{"role": "user", "content": prompt}]
            llm_response_str = ""
            
            async for chunk in context.llm_interface.call_llm(messages=llm_input, stream=False):
                 llm_response_str += chunk

            logger.debug(f"LLM tool selection response: {llm_response_str}")
            if llm_response_str.strip().startswith("```json"):
                 llm_response_str = llm_response_str.strip()[7:-3].strip()
            elif llm_response_str.strip().startswith("```"):
                 llm_response_str = llm_response_str.strip()[3:-3].strip()
                 
            parsed_response = json.loads(llm_response_str)
            tool_name = parsed_response.get("tool_name")
            action_input = parsed_response.get("action_input")
            
            if not tool_name or not isinstance(action_input, dict):
                if tool_name is None and action_input is None:
                     logger.warning(f"LLM determined no suitable tool for sub-task: {sub_task}")
                     return {"status": "error", "message": f"LLM could not map sub-task '{sub_task}' to a file operation."}
                else:
                    raise ValueError("LLM did not return valid tool_name and action_input dict.")
            logger.info(f"LLM selected tool: '{tool_name}' with args: {action_input}")

        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse LLM tool selection JSON: {json_err}. Response: {llm_response_str[:200]}")
            return {"status": "error", "message": f"Failed to parse LLM tool selection response: {json_err}"}
        except Exception as llm_err:
            logger.exception(f"Error during LLM call for tool selection: {llm_err}")
            return {"status": "error", "message": f"Error asking LLM for tool selection: {llm_err}"}
        # --- End LLM Call --- 
        
        # Execute the chosen tool
        if tool_name and action_input:
            logger.info(f"Executing tool: '{tool_name}' with args: {action_input}")
            try:
                # Retrieve the callable tool function
                tool_callable = context.tool_registry.get_tool(tool_name)
                
                # Retrieve instance info (needed for method calls)
                instance, _ = context.tool_registry.get_instance_and_tool(tool_name)
                skill_instance = instance # Alias for clarity

                # Check if the retrieved tool is actually callable
                if not callable(tool_callable):
                    logger.error(f"Retrieved tool '{tool_name}' is not callable.")
                    return {"status": "error", "message": f"Tool '{tool_name}' not found or not properly registered as callable."}
                
                logger.info(f"Found tool '{tool_name}'. Instance: {'Yes' if skill_instance else 'No'}.")

                # Prepare the specific context expected by the skill/tool
                tool_ctx = _ToolExecutionContext(
                    logger=context.logger,
                    workspace_root=context.workspace_root, # <<< USE CONTEXT workspace_root >>>
                    llm_url=context.llm_interface.llm_url if context.llm_interface else None,
                    tools_dict=context.tool_registry,
                    llm_interface=context.llm_interface,
                    fragment_registry=context.fragment_registry,
                    shared_task_context=context.shared_task_context,
                    allowed_skills=None,
                    skill_instance=skill_instance,
                    memory_manager=context.memory_manager # <<< ADICIONADO ESTE ARGUMENTO >>>
                )

                # <<< REMOVING TEMPORARY MODIFICATION >>>
                # if tool_name == "write_file" and isinstance(action_input, dict):
                #     logger.warning("!!! INTRODUCING ARTIFICIAL ERROR FOR TESTING SELF-CORRECTION !!!")
                #     action_input['content'] = 123 # Force invalid type
                # <<< END REMOVAL >>>

                # Prepare arguments for the tool call
                # Filter action_input to only include parameters defined in the tool's schema
                # This prevents passing unexpected args like 'ctx' or decorator-injected args
                tool_sig = inspect.signature(tool_callable)
                valid_param_names = set(tool_sig.parameters.keys()) - {'self', 'ctx', 'resolved_path', 'original_path_str'}
                filtered_action_input = { k: v for k, v in action_input.items() if k in valid_param_names }
                logger.debug(f"Filtered action input for {tool_name}: {filtered_action_input}")

                # Call the tool with the correct arguments
                if skill_instance:
                    # It's a method, pass instance, context, and filtered args
                    logger.debug(f"Calling method {tool_name} on instance {type(skill_instance)} with context {type(tool_ctx)}")
                    execution_result_payload = await tool_callable(skill_instance, tool_ctx, **filtered_action_input)
                else:
                    # It's a standalone function, check if it needs context
                    if 'ctx' in tool_sig.parameters:
                        logger.debug(f"Calling standalone function {tool_name} with context {type(tool_ctx)}")
                        execution_result_payload = await tool_callable(tool_ctx, **filtered_action_input)
                    else:
                        logger.debug(f"Calling standalone function {tool_name} without context")
                        execution_result_payload = await tool_callable(**filtered_action_input)

                # Wrap result (simplified)
                final_status = "unknown"
                final_message = "No message provided"
                tool_data = None
                if isinstance(execution_result_payload, dict):
                     final_status = execution_result_payload.get("status", "unknown")
                     final_message = execution_result_payload.get("message", execution_result_payload.get("data", {}).get("message", "No message provided"))
                     tool_data = execution_result_payload.get("data") # Extract data if available
                else:
                     # Handle non-dict results if necessary, maybe log a warning
                     logger.warning(f"Tool '{tool_name}' returned non-dict result: {type(execution_result_payload)}")
                     final_message = f"Tool returned unexpected type: {type(execution_result_payload)}"
                     final_status = "error" # Assume error if format is wrong

                # --- Process Result ---
                logger.info(f"Tool '{tool_name}' execution finished with status: {final_status}")

                # <<< ADDED: Self-Correction Logic >>>
                if final_status == "error":
                    logger.warning(f"Tool '{tool_name}' failed. Attempting self-correction using LLM.")
                    original_error_message = final_message
                    try:
                        # Need ALL managed schemas for the correction prompt now
                        # (Re-fetch or ensure managed_schemas is available in this scope)
                        # Assuming managed_schemas from the initial LLM call is accessible here
                        if not managed_schemas: # Defensive check
                             logger.error("Cannot attempt self-correction: managed_schemas unavailable.")
                             raise ValueError("Managed schemas needed for correction prompt.")
                             
                        correction_prompt = f"""
The overall goal for this manager was the sub-task: '{sub_task}'

The manager initially chose the tool '{tool_name}' with parameters:
{json.dumps(filtered_action_input, indent=2)}

However, executing '{tool_name}' failed with the error: '{original_error_message}'

Here are ALL the tools this FileOpsManager can use:
{json.dumps(managed_schemas, indent=2)} # Pass all schemas now

Analyze the original sub-task, the chosen tool ({tool_name}), its parameters, the resulting error, and the full list of available tools.

Determine the best course of action:
1.  **Retry the SAME tool ('{tool_name}') with corrected parameters:** If the error seems fixable by adjusting the input arguments for '{tool_name}'.
2.  **Use an ALTERNATIVE tool:** If the error suggests '{tool_name}' was fundamentally the wrong tool for the sub-task (e.g., trying to list a non-existent directory might require 'create_directory' first).

Respond ONLY with a JSON object containing:
- 'tool_name': The name of the tool to use (either '{tool_name}' or an alternative).
- 'action_input': A JSON object with the parameters for the chosen tool.

Example Response (Correcting Parameters):
{{
  "tool_name": "write_file",
  "action_input": {{
    "filename": "data/output.txt",
    "content": "Corrected content.",
    "overwrite": true
  }}
}}

Example Response (Suggesting Alternative Tool):
{{
  "tool_name": "create_directory",
  "action_input": {{
    "directory": "a/non/existent/dir"
  }}
}}

If no correction or alternative seems possible, respond with: {{"tool_name": null, "action_input": null}}
"""
                        correction_llm_input = [{"role": "user", "content": correction_prompt}]
                        correction_response_str = ""
                        async for chunk in context.llm_interface.call_llm(messages=correction_llm_input, stream=False, temperature=0.1):
                            correction_response_str += chunk
                        
                        logger.debug(f"LLM self-correction response: {correction_response_str}")

                        # Attempt to parse the corrected JSON input (tool_name and action_input)
                        corrected_tool_name = None
                        corrected_action_input = None
                        retry_successful = False
                        try:
                            # Extract JSON part
                            if correction_response_str.strip().startswith("```json"):
                                json_str = correction_response_str.strip()[7:-3].strip()
                            elif correction_response_str.strip().startswith("```"):
                                json_str = correction_response_str.strip()[3:-3].strip()
                            else:
                                json_str = correction_response_str.strip()
                                
                            parsed_correction = json.loads(json_str)
                            corrected_tool_name = parsed_correction.get("tool_name")
                            corrected_action_input = parsed_correction.get("action_input")

                            if corrected_tool_name and isinstance(corrected_action_input, dict):
                                logger.info(f"LLM suggested trying tool '{corrected_tool_name}' with args: {corrected_action_input}")
                                
                                # --- Get callable, instance, signature for the CORRECTED tool ---
                                try:
                                    new_tool_callable = context.tool_registry.get_tool(corrected_tool_name)
                                    new_instance, _ = context.tool_registry.get_instance_and_tool(corrected_tool_name)
                                    new_skill_instance = new_instance

                                    if not callable(new_tool_callable):
                                        raise ValueError(f"Corrected tool '{corrected_tool_name}' is not callable.")

                                    new_tool_sig = inspect.signature(new_tool_callable)
                                    new_valid_param_names = set(new_tool_sig.parameters.keys()) - {'self', 'ctx', 'resolved_path', 'original_path_str'}
                                    
                                except Exception as tool_lookup_err:
                                     logger.error(f"Failed to retrieve details for corrected tool '{corrected_tool_name}': {tool_lookup_err}")
                                     raise ValueError(f"Invalid corrected tool suggested: {corrected_tool_name}") from tool_lookup_err
                                # ------------------------------------------------------------------

                                # Filter corrected args against the NEW tool's signature
                                filtered_corrected_action_input = { k: v for k, v in corrected_action_input.items() if k in new_valid_param_names }
                                logger.debug(f"Filtered corrected action input for {corrected_tool_name}: {filtered_corrected_action_input}")

                                # Re-call the potentially NEW tool with corrected arguments
                                logger.info(f"Executing retry with tool: '{corrected_tool_name}'")
                                if new_skill_instance:
                                    execution_result_payload = await new_tool_callable(new_skill_instance, tool_ctx, **filtered_corrected_action_input)
                                else:
                                    if 'ctx' in new_tool_sig.parameters:
                                        execution_result_payload = await new_tool_callable(tool_ctx, **filtered_corrected_action_input)
                                    else:
                                        execution_result_payload = await new_tool_callable(**filtered_corrected_action_input)

                                # Re-process the result of the retry
                                if isinstance(execution_result_payload, dict):
                                    final_status = execution_result_payload.get("status", "unknown")
                                    final_message = execution_result_payload.get("message", f"Retry ({corrected_tool_name}): No message provided")
                                    tool_data = execution_result_payload.get("data") 
                                    if final_status == "success":
                                        retry_successful = True
                                        logger.info(f"Self-correction retry with tool '{corrected_tool_name}' was successful.")
                                        # Optional: Update context with successful corrected action details if needed
                                        tool_name = corrected_tool_name # Reflect the tool that actually succeeded
                                        filtered_action_input = filtered_corrected_action_input # Reflect the args that succeeded
                                    else:
                                         logger.warning(f"Self-correction retry with '{corrected_tool_name}' failed with status: {final_status}, message: {final_message}")
                                else:
                                    logger.warning(f"Tool '{corrected_tool_name}' retry returned non-dict result: {type(execution_result_payload)}")
                                    final_message = f"Retry with {corrected_tool_name} returned unexpected type: {type(execution_result_payload)}"
                                    final_status = "error"

                            elif corrected_tool_name is None and corrected_action_input is None:
                                logger.warning("LLM determined no correction or alternative was possible.")
                                # Keep original error status and message
                            else:
                                logger.warning("LLM correction response did not contain valid tool_name and action_input dict.")
                                final_message = "LLM self-correction did not provide valid tool/arguments."
                                # Keep original error status

                        except (json.JSONDecodeError, ValueError) as corr_parse_err:
                            logger.error(f"Failed to parse or process LLM correction response: {corr_parse_err}. Response: {correction_response_str[:200]}")
                            final_message = f"Failed to parse/process LLM self-correction response: {corr_parse_err}"
                            # Keep original error status
                        except Exception as retry_err:
                            # Catch errors during the retry execution itself (e.g., permission denied on the new tool)
                            logger.exception(f"Error during self-correction retry execution of {corrected_tool_name or tool_name}: {retry_err}")
                            final_message = f"Error executing self-correction retry ({corrected_tool_name or tool_name}): {retry_err}"
                            final_status = "error" # Ensure status is error if retry execution fails

                    except Exception as correction_err:
                         logger.exception(f"Error during the self-correction LLM call process for {tool_name}: {correction_err}")
                         final_message = f"Original error: {original_error_message}. Failed during self-correction attempt: {correction_err}"
                         final_status = "error"

                # <<< END: Self-Correction Logic >>>

                # --- Update Shared Context --- 
                # ... existing code ...

            except Exception as tool_err:
                logger.exception(f"Error during tool execution: {tool_err}")
                return {"status": "error", "message": f"Error executing tool '{tool_name}': {tool_err}"}

        return {"status": final_status, "message": final_message, "data": tool_data}