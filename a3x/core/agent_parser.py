import json
import logging
import re
from typing import Tuple, Optional, Dict, Any


def parse_llm_response(
    response: str, agent_logger: logging.Logger
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Parses the LLM's raw response string, expecting a multi-line text format
    with 'Thought:', 'Action:', and 'Action Input:' prefixes.
    The 'Action Input' value is expected to be a JSON string.
    Includes fallbacks for common Portuguese keywords.

    Returns: thought (str), action (str), action_input (dict)
             Returns None, None, None if parsing fails significantly.
    Raises: Does not raise JSONDecodeError directly anymore unless Action Input parsing fails.
    """
    agent_logger.debug(
        f"[Agent Parse DEBUG] Raw LLM Response (expecting ReAct Text):\n{response}"
    )

    thought = None
    action = None
    action_input_str = None
    action_input = {}

    try:
        # Use regex to find the sections, allowing for optional whitespace and case variations
        # Include Portuguese fallbacks
        thought_match = re.search(
            r"^[ \t]*(?:Thought|Pensamento):(.*)", response, re.MULTILINE | re.IGNORECASE
        )
        action_match = re.search(
            r"^[ \t]*(?:Action|Ação):(.*)", response, re.MULTILINE | re.IGNORECASE
        )
        action_input_match = re.search(
            # Note: Action Input fallback might be less common, but added for completeness
            r"^[ \t]*(?:Action Input|Input da Ação):(.*)",
            response,
            re.MULTILINE | re.IGNORECASE | re.DOTALL,
        ) # DOTALL for multi-line JSON
        final_answer_match = re.search(
            r"^[ \t]*(?:Final Answer|Resposta Final):(.*)", response, re.MULTILINE | re.IGNORECASE | re.DOTALL
        )

        if thought_match:
            thought = thought_match.group(1).strip()
            agent_logger.debug(f"[Agent Parse DEBUG] Extracted Thought: {thought}")

        if action_match:
            action = action_match.group(1).strip()
            agent_logger.debug(f"[Agent Parse DEBUG] Extracted Action: {action}")
        elif final_answer_match:
            # If Action is missing, but Final Answer is present, treat it as the final action
            action = "final_answer"
            # Attempt to extract the answer content for the action_input dictionary
            action_input_str = final_answer_match.group(1).strip()
            if action_input_str: # If content exists, wrap it in the expected structure
                action_input = {"answer": action_input_str}
            else: # If Final Answer line exists but is empty, use thought or placeholder
                action_input = {"answer": thought if thought else "(Final answer content not found)"}
            agent_logger.debug(f"[Agent Parse DEBUG] Extracted Final Answer as Action: {action}, Input: {action_input}")
        else:
            # Action (or Final Answer) is mandatory for the ReAct loop
            agent_logger.error(
                f"[Agent Parse ERROR] 'Action:' or 'Final Answer:' section not found in LLM response.\nResponse:\n{response}"
            )
            return None, None, None # Critical parsing failure

        # Process Action Input only if the action is NOT final_answer (handled above)
        if action != "final_answer":
            if action_input_match:
                action_input_str = action_input_match.group(1).strip()
                agent_logger.debug(
                    f"[Agent Parse DEBUG] Extracted Action Input String: {action_input_str}"
                )
                if action_input_str:
                    try:
                        # Clean potential markdown
                        if action_input_str.startswith("```json"):
                            action_input_str = (
                                action_input_str.removeprefix("```json")
                                .removesuffix("```")
                                .strip()
                            )
                        elif action_input_str.startswith("```"):
                            action_input_str = (
                                action_input_str.removeprefix("```")
                                .removesuffix("```")
                                .strip()
                            )

                        decoder = json.JSONDecoder()
                        action_input, end_pos = decoder.raw_decode(action_input_str)
                        agent_logger.debug(
                            f"[Agent Parse DEBUG] raw_decode parsed JSON up to position {end_pos}. Parsed object: {action_input}"
                        )
                        if end_pos < len(action_input_str.strip()):
                            trailing_data = action_input_str[end_pos:].strip()
                            agent_logger.warning(
                                f"[Agent Parse WARN] Trailing data found after JSON in Action Input: '{trailing_data[:100]}...'"
                            )
                        if not isinstance(action_input, dict):
                            agent_logger.warning(
                                f"[Agent Parse WARN] Action Input parsed but is not a dictionary ({type(action_input)}): {action_input}. Using empty dict."
                            )
                            action_input = {}
                        else:
                            agent_logger.debug(
                                "[Agent Parse DEBUG] Action Input parsed successfully (using raw_decode)."
                            )

                    except json.JSONDecodeError as e:
                        agent_logger.error(
                            f"[Agent Parse ERROR] Failed to decode Action Input string with raw_decode: {e}\nString was: '{action_input_str}'"
                        )
                        action_input = {
                            "_parse_error": f"Failed to decode JSON with raw_decode: {e}"
                        }
                else:
                    agent_logger.debug(
                        "[Agent Parse DEBUG] Action Input section found but was empty."
                    )
                    action_input = {}
            else:
                # Action Input is missing for a non-final_answer action
                agent_logger.warning(
                    f"[Agent Parse WARN] 'Action Input:' section not found for action '{action}'. Assuming empty input."
                )
                action_input = {} # Assume empty if section is missing

        agent_logger.info(
            f"[Agent Parse INFO] Text parsed successfully. Action: '{action}'"
        )
        return thought, action, action_input

    except Exception:
        agent_logger.exception(
            "[Agent Parse ERROR] Unexpected error during text parsing:"
        )
        return None, None, None


# --- Keep old JSON parser for potential future use or reference? ---
# (Original parse_llm_response function expecting direct JSON)
# def parse_llm_response_json(response: str, agent_logger: logging.Logger) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
#     ... (original implementation) ...
# -------------------------------------------------------------------
