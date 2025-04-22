import logging
from typing import Dict, Any, Optional
import json

from a3x.fragments.base import BaseFragment
from a3x.core.context import Context
from a3x.core.tool_executor import ToolExecutor # Assuming ToolExecutor is the way to call skills
from a3x.core.llm_interface import LLMInterface # To interact with the Professor LLM
from a3x.core.db_utils import add_episodic_record # <<< Added for heuristic logging

logger = logging.getLogger(__name__)

# Default prompt for self-improvement question
# Simplify internal quotes to avoid potential linter confusion
DEFAULT_EXPANSION_PROMPT = "How can I expand my capabilities to better achieve my goals? Suggest a new, simple function or skill I could create, described in a single sentence. Then, provide an A3L command using 'criar_fragmento' to create it. Example response format: Suggestion: Create a fragment to analyze sentiment. A3L: criar_fragmento nome='sentiment_analyzer' descricao='Analyzes the sentiment of a given text.'"

class AutonomousSelfStarterFragment(BaseFragment):
    \"\"\"
    A fragment that attempts to autonomously expand the system's capabilities.
    It queries an LLM for suggestions, interprets them, and tries to execute
    commands like 'criar_fragmento' to add new skills/fragments.
    \"\"\"

    def __init__(self, fragment_id: str = "autonomous_self_starter", llm_interface: Optional[LLMInterface] = None):
        super().__init__(fragment_id=fragment_id)
        # We might need an LLM interface injected or retrieved from context later
        self.llm_interface = llm_interface 
        logger.info(f"AutonomousSelfStarterFragment initialized.")

    async def execute(self, context: Optional[Context] = None, **kwargs: Any) -> Any:
        """
        Executes the autonomous expansion process:
        1. Gets the ToolExecutor from context.
        2. Formulates a query for the LLM to suggest a new capability.
        3. Calls the 'ask' skill to get the LLM's suggestion and A3L command.
        4. Interprets the response to extract the A3L command.
        5. Executes the extracted A3L command (e.g., criar_fragmento) using parse_and_execute.
        6. If successful and 'criar_fragmento' was used, triggers the 'validate_fragment' skill.
        7. Returns the result of the A3L command execution or an error dictionary.
        """
        logger.info(f"Executing {self.fragment_id}... Seeking self-improvement.")

        if not context:
            logger.error("Execution context is required for autonomous expansion.")
            return {"status": "error", "message": "Missing execution context."}

        # 1. Get ToolExecutor from context
        if not hasattr(context, 'tool_executor') or not isinstance(context.tool_executor, ToolExecutor):
            logger.error("ToolExecutor not found or invalid in context.")
            return {"status": "error", "message": "Missing or invalid ToolExecutor."}
        tool_executor: ToolExecutor = context.tool_executor

        # 2. Formulate question for Professor LLM (or use a default)
        expansion_query = kwargs.get("expansion_prompt", DEFAULT_EXPANSION_PROMPT)
        logger.debug(f"Formulating expansion query: {expansion_query}")

        # 3. Call the 'ask' skill
        ask_input = {
            # TODO: Determine the correct target for the 'ask' skill (e.g., 'professor_llm')
            "fragment_id": "professor_llm", # Placeholder - Needs correct target ID
            "query_text": expansion_query
        }
        try:
            logger.info(f"Asking LLM for expansion suggestions... Query: {ask_input['query_text']}")
            llm_response_dict = await tool_executor.execute_tool("ask", ask_input, context)

            if llm_response_dict.get("status") != "success":
                 raise Exception(f"LLM query failed: {llm_response_dict.get('message', 'Unknown error')}")

            llm_response_text = llm_response_dict.get("result")
            if not llm_response_text:
                 raise Exception("LLM response was empty or invalid.")

            logger.info(f"Received LLM suggestion: {llm_response_text[:100]}...")

        except Exception as e:
            logger.exception(f"Error querying LLM for expansion: {e}")
            return {"status": "error", "message": f"Failed to get expansion suggestion: {e}"}

        # 4. Interpret the LLM response to extract an A3L command
        a3l_command = None
        try:
            # Simple extraction based on the example format
            if "A3L:" in llm_response_text:
                a3l_command = llm_response_text.split("A3L:", 1)[1].strip()
                logger.info(f"Extracted A3L command: {a3l_command}")
            else:
                 logger.warning(f"Could not extract A3L command from LLM response: {llm_response_text}")
                 raise Exception("Response did not contain expected 'A3L:' prefix.")

            if not a3l_command:
                 raise Exception("Failed to extract or interpret A3L command.")

        except Exception as e:
            logger.exception(f"Error interpreting LLM response: {e}")
            return {"status": "error", "message": f"Failed to interpret LLM response: {e}"}

        # 5. Execute the extracted A3L command
        try:
            # Attempt to import dynamically to reduce top-level dependencies
            from a3x.a3lang.interpreter import parse_and_execute

            logger.info(f"Executing extracted A3L command: {a3l_command}")
            execution_result = await parse_and_execute(a3l_command, context)
            logger.info(f"A3L command execution result: {execution_result}")

            # 6. Trigger validation based on execution_result
            if execution_result.get("status") == "success" and "criar_fragmento" in a3l_command:
                new_fragment_path = execution_result.get("path")
                if new_fragment_path:
                    logger.info(f"Successfully created fragment at: {new_fragment_path}. Triggering validation.")
                    try:
                        validation_input = {"fragment_path": new_fragment_path}
                        # Ensure the skill name matches the one we created/validated
                        validation_result = await tool_executor.execute_tool("validate_fragment", validation_input, context)
                        logger.info(f"Validation result for {new_fragment_path}: {validation_result}") # Log regardless of outcome

                        # <<< Handle validation failure >>>
                        if validation_result.get("status") != "success":
                            error_message = validation_result.get("message", "Unknown validation error")
                            validation_details = validation_result.get("details", {})
                            logger.error(f"Validation failed for newly created fragment {new_fragment_path}: {error_message}")
                            # Log heuristic about the failure
                            try:
                                heuristic_metadata = {
                                    "fragment_path": new_fragment_path,
                                    "validation_error": error_message,
                                    "validation_details": validation_details,
                                    "source_command": a3l_command # Include the command that led to this
                                }
                                add_episodic_record(
                                    context="fragment_validation_failure",
                                    action=f"validate_fragment:{new_fragment_path}", # Action performed
                                    outcome="failure",
                                    metadata=heuristic_metadata
                                )
                                logger.info(f"Logged validation failure heuristic for {new_fragment_path}.")
                            except Exception as db_err:
                                logger.error(f"Failed to log validation failure heuristic: {db_err}")
                            
                            # Return the validation error result to stop further processing based on this failed fragment
                            return validation_result
                        else:
                             logger.info(f"Fragment {new_fragment_path} validated successfully.")
                             # TODO: Potentially trigger next steps like loading/using the fragment

                    except Exception as val_err:
                        # Log the specific validation error
                        logger.error(f"Exception occurred during validation call for {new_fragment_path}: {val_err}", exc_info=True)
                        # Return an error if the validation skill itself failed unexpectedly
                        return {"status": "error", "message": f"Exception during fragment validation call: {val_err}"}
                else:
                     logger.warning("Fragment creation (criar_fragmento) reported success, but no path was returned in the result.")
                     # Return the original success result, but with a warning implicit
                     # Or potentially return an error? For now, return original.

            # Return the result of the original A3L command execution (criar_fragmento or other)
            # If validation failed above, this line won't be reached for criar_fragmento.
            return execution_result

        except ImportError as e:
             logger.exception("Could not import parse_and_execute. Cannot execute extracted command.")
             return {"status": "error", "message": "Execution mechanism not available."}
        except Exception as e:
            # Use repr(a3l_command) in case it's None or not a string
            logger.exception(f"Error executing extracted A3L command {repr(a3l_command)}: {e}")
            return {"status": "error", "message": f"Failed to execute extracted command: {e}"} 