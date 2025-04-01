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
    action_input = {}  # Default to empty dict

    try:
        # Use regex to find the sections, allowing for optional whitespace and case variations
        thought_match = re.search(
            r"^[ \t]*Thought:(.*)", response, re.MULTILINE | re.IGNORECASE
        )
        action_match = re.search(
            r"^[ \t]*Action:(.*)", response, re.MULTILINE | re.IGNORECASE
        )
        action_input_match = re.search(
            r"^[ \t]*Action Input:(.*)",
            response,
            re.MULTILINE | re.IGNORECASE | re.DOTALL,
        )  # DOTALL for multi-line JSON

        if thought_match:
            thought = thought_match.group(1).strip()
            agent_logger.debug(f"[Agent Parse DEBUG] Extracted Thought: {thought}")

        if action_match:
            action = action_match.group(1).strip()
            agent_logger.debug(f"[Agent Parse DEBUG] Extracted Action: {action}")
        else:
            # Action is mandatory for the ReAct loop to continue meaningfully
            agent_logger.error(
                f"[Agent Parse ERROR] 'Action:' section not found in LLM response.\nResponse:\n{response}"
            )
            # Return None tuple to signal critical parsing failure
            return None, None, None

        if action_input_match:
            action_input_str = action_input_match.group(1).strip()
            agent_logger.debug(
                f"[Agent Parse DEBUG] Extracted Action Input String: {action_input_str}"
            )
            # Try to parse the extracted string as JSON
            if action_input_str:
                try:
                    # Clean potential markdown code blocks if present
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

                    # Use raw_decode to parse only the initial JSON object
                    decoder = json.JSONDecoder()
                    action_input, end_pos = decoder.raw_decode(action_input_str)
                    agent_logger.debug(
                        f"[Agent Parse DEBUG] raw_decode parsed JSON up to position {end_pos}. Parsed object: {action_input}"
                    )
                    # Optional: Log if there was trailing data
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
                    # This error might still happen if the *start* of the string is not valid JSON
                    agent_logger.error(
                        f"[Agent Parse ERROR] Failed to decode Action Input string with raw_decode (invalid JSON start?): {e}\nString was: '{action_input_str}'"
                    )
                    action_input = {
                        "_parse_error": f"Failed to decode JSON with raw_decode: {e}"
                    }  # Include error info
            else:
                agent_logger.debug(
                    "[Agent Parse DEBUG] Action Input section found but was empty."
                )
                action_input = {}  # Explicitly empty if string is empty
        elif action != "final_answer":
            # If Action Input is missing BUT action is not final_answer, it might be problematic
            agent_logger.warning(
                f"[Agent Parse WARN] 'Action Input:' section not found for action '{action}'. Assuming empty input."
            )
            action_input = {}  # Assume empty if section is missing

        # Specific handling for final_answer if Action Input is missing/empty
        if action == "final_answer" and not action_input.get("answer"):
            agent_logger.warning(
                "[Agent Parse WARN] 'final_answer' action detected but 'answer' key is missing in Action Input. Using thought or placeholder."
            )
            # Use thought as answer if available, otherwise use a placeholder
            final_answer_content = (
                thought if thought else "(Final answer content not found)"
            )
            action_input = {"answer": final_answer_content}

        agent_logger.info(
            f"[Agent Parse INFO] Text parsed successfully. Action: '{action}'"
        )
        return thought, action, action_input

    except Exception:
        # Catch unexpected errors during regex or processing
        agent_logger.exception(
            "[Agent Parse ERROR] Unexpected error during text parsing:"
        )
        # Return None tuple to signal critical parsing failure
        return None, None, None


# --- Keep old JSON parser for potential future use or reference? ---
# (Original parse_llm_response function expecting direct JSON)
# def parse_llm_response_json(response: str, agent_logger: logging.Logger) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
#     ... (original implementation) ...
# -------------------------------------------------------------------
