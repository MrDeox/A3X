import json
import logging
import re
from typing import Tuple, Optional, Dict, Any, Union


def parse_llm_response(
    response: str, agent_logger: logging.Logger
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parses the LLM's raw response string, expecting a multi-line text format
    with 'Thought:', 'Action:', and potentially 'Action Input:' prefixes.
    It prioritizes finding 'Action:' and extracting the tool name and JSON input from it.
    If 'Action:' is missing, it checks for 'Final Answer:'.

    Returns: thought (str), action_name (str), action_input_str (str | None)
             Returns None, None, None if critical parsing fails.
    """
    agent_logger.debug(
        f"[Agent Parse DEBUG] Raw LLM Response (expecting ReAct Text):\\n{response}"
    )

    thought = None
    action_name = None
    action_input_str = None

    try:
        # Extract Thought (optional)
        thought_match = re.search(
            r"^[ \\t]*(?:Thought|Pensamento):(.*)", response, re.MULTILINE | re.IGNORECASE
        )
        if thought_match:
            thought = thought_match.group(1).strip()
            agent_logger.debug(f"[Agent Parse DEBUG] Extracted Thought: {thought}")

        # --- Primary Logic: Look for Action first ---
        action_match = re.search(
            r"^[ \\t]*(?:Action|Ação):(.*)", response, re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        
        if action_match:
            full_action_content = action_match.group(1).strip()
            agent_logger.debug(f"[Agent Parse DEBUG] Full Action content found: {full_action_content}")

            # --- Revised Parsing Logic ---
            # Split content into potential action name (first line) and the rest
            parts = full_action_content.split('\n', 1)
            potential_action_name = parts[0].strip()
            rest_of_content = parts[1].strip() if len(parts) > 1 else ""

            agent_logger.debug(f"[Agent Parse DEBUG] Split Action Name: '{potential_action_name}', Rest: '{rest_of_content}'")

            action_input_str = None # Default to no input

            # Now, look for Action Input and JSON *only* in the 'rest_of_content'
            if rest_of_content:
                # Check if the rest starts with "Action Input:" or similar
                action_input_marker_match = re.match(r"^[ \\t]*(?:Action Input|Input da Ação):(.*)", rest_of_content, re.IGNORECASE | re.DOTALL)
                if action_input_marker_match:
                    content_after_marker = action_input_marker_match.group(1).strip()
                    agent_logger.debug(f"[Agent Parse DEBUG] Found Action Input marker. Content after marker: '{content_after_marker}'")

                    # Try to find the first JSON object ({...}) within this remaining content
                    json_match = re.search(r"(\{.*\})", content_after_marker, re.DOTALL)
                    if json_match:
                        potential_json_str = json_match.group(1).strip()
                        # Clean potential markdown fences
                        cleaned_input_str = potential_json_str
                        if cleaned_input_str.startswith("```json"):
                            cleaned_input_str = cleaned_input_str.removeprefix("```json").removesuffix("```").strip()
                        elif cleaned_input_str.startswith("```"):
                            cleaned_input_str = cleaned_input_str.removeprefix("```").removesuffix("```").strip()

                        # Basic validation: Does it look like a JSON object?
                        if cleaned_input_str.startswith('{') and cleaned_input_str.endswith('}'):
                            action_input_str = cleaned_input_str
                            agent_logger.debug(f"[Agent Parse DEBUG] Extracted Valid JSON Input: '{action_input_str}'")
                        else:
                            agent_logger.warning(f"[Agent Parse WARN] Found JSON-like block after Action Input marker but failed validation: '{cleaned_input_str}'. Input set to None.")
                            # action_input_str remains None
                    else:
                         agent_logger.debug("[Agent Parse DEBUG] No JSON object found after Action Input marker. Input set to None.")
                         # action_input_str remains None
                else:
                    agent_logger.debug(f"[Agent Parse DEBUG] No 'Action Input:' marker found in rest_of_content: '{rest_of_content}'. Assuming no input.")
                    # action_input_str remains None
            else:
                 agent_logger.debug("[Agent Parse DEBUG] No content after the first line (Action Name). Assuming no input.")
                 # action_input_str remains None

            # Assign the confirmed action name
            action_name = potential_action_name
            # --- End Revised Parsing Logic ---

            # If action_name ended up empty (e.g., "Action:"), it's an error
            if not action_name:
                 agent_logger.error(f"[Agent Parse ERROR] Extracted empty action_name from content: {full_action_content}")
                 return None, None, None

            agent_logger.info(f"[Agent Parse INFO] Parsed from Action block. Action: '{action_name}'. Input String: '{action_input_str}'")
            return thought, action_name, action_input_str

        # --- Fallback Logic: Look for Final Answer if Action was missing ---
        final_answer_match = re.search(
            r"^[ \t]*(?:Final Answer|Resposta Final):(.*)", response, re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        if final_answer_match:
            action_name = "final_answer"
            raw_final_answer = final_answer_match.group(1).strip()
            # Always wrap final answer in JSON for consistency
            action_input_str = json.dumps({"answer": raw_final_answer})
            agent_logger.info(f"[Agent Parse INFO] Parsed from Final Answer block. Action: '{action_name}'. Input String: '{action_input_str}'")
            return thought, action_name, action_input_str

        # --- Error Case: Neither Action nor Final Answer found ---
        agent_logger.error(f"[Agent Parse ERROR] Neither 'Action:' nor 'Final Answer:' section found in response: {response[:500]}...")
        return None, None, None

    except Exception as e:
        agent_logger.exception(f"[Agent Parse ERROR] Unexpected error during text parsing: {e}")
        return None, None, None


# --- NEW FUNCTION ---
def parse_orchestrator_response(
    response: str, agent_logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """
    Parses the LLM's raw response string, expecting a direct JSON object
    containing 'fragment' and 'sub_task'.
    Handles responses potentially wrapped in markdown code blocks (```json ... ```).

    Returns: Parsed dictionary {fragment: str, sub_task: str} or None if parsing fails.
    """
    agent_logger.debug(
        f"[Agent Parse DEBUG] Raw Orchestrator LLM Response (expecting JSON):\\n{response}"
    )
    cleaned_response = response.strip()

    # Strip markdown code blocks if present
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[len("```json"):]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-len("```")]
        cleaned_response = cleaned_response.strip()
        agent_logger.debug("[Agent Parse DEBUG] Stripped ```json markdown block.")
    elif cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[len("```"):]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-len("```")]
        cleaned_response = cleaned_response.strip()
        agent_logger.debug("[Agent Parse DEBUG] Stripped ``` markdown block.")

    # Replace potential curly quotes just in case
    cleaned_response = cleaned_response.replace(""", '"').replace(""", '"')

    try:
        parsed_json = json.loads(cleaned_response)
        if isinstance(parsed_json, dict) and "fragment" in parsed_json and "sub_task" in parsed_json:
            agent_logger.info(
                f"[Agent Parse INFO] Orchestrator response parsed successfully: {parsed_json}"
            )
            return parsed_json
        else:
            agent_logger.error(
                f"[Agent Parse ERROR] Parsed JSON is not a dictionary or lacks required keys ('fragment', 'sub_task'). Parsed: {parsed_json}"
            )
            return None
    except json.JSONDecodeError as e:
        agent_logger.error(
            f"[Agent Parse ERROR] Failed to decode JSON from orchestrator response: {e}\\nCleaned String was: '{cleaned_response}'"
        )
        return None
    except Exception as e:
        agent_logger.exception(
            f"[Agent Parse ERROR] Unexpected error during orchestrator JSON parsing:"
        )
        return None


# --- Keep old JSON parser for potential future use or reference? ---
# (Original parse_llm_response function expecting direct JSON)
# def parse_llm_response_json(response: str, agent_logger: logging.Logger) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
#     ... (original implementation) ...
# -------------------------------------------------------------------
