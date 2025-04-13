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
            # Fallback: tentar extrair intenção de ação mesmo sem o formato estrito
            agent_logger.warning(
                "[Agent Parse WARN] 'Action:' ou 'Final Answer:' não encontrados. Tentando fallback de parsing adaptativo."
            )
            # Heurística: procurar comandos de skill em texto livre
            action_fallback = None
            action_input_fallback = {}
            # Exemplo: "use the write_file tool to create..." ou "utilize a ferramenta write_file para criar..."
            skill_match = re.search(r"(?:use|utilize|utilize a ferramenta|utilize a skill|execute|chame|call) (?:the )?(\w+)[\s_-]?(?:tool|skill|ferramenta)?", response, re.IGNORECASE)
            if skill_match:
                action_fallback = skill_match.group(1).strip()
                agent_logger.warning(f"[Agent Parse FALLBACK] Skill inferida: {action_fallback}")
                # Tentar extrair parâmetros básicos (ex: path, content) do texto
                path_match = re.search(r"(?:file|arquivo|path|diretório|directory)[\s:]*['\"]?([^\s'\"\,\)]+)", response, re.IGNORECASE)
                content_match = re.search(r"(?:content|conteúdo|texto|text)[\s:]*['\"]?([^\n'\"\,\)]+)", response, re.IGNORECASE)
                if path_match:
                    action_input_fallback["file_path"] = path_match.group(1)
                if content_match:
                    action_input_fallback["content"] = content_match.group(1)
                # Registrar heurística de parsing adaptativo
                try:
                    from a3x.core.learning_logs import log_heuristic_with_traceability
                    import datetime
                    plan_id = f"plan-parse-fallback-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    execution_id = f"exec-parse-fallback-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    heuristic = {
                        "type": "parsing_fallback",
                        "raw_response": response[:500],
                        "action_inferred": action_fallback,
                        "action_input_inferred": action_input_fallback,
                    }
                    log_heuristic_with_traceability(heuristic, plan_id, execution_id, validation_status="pending_manual")
                    agent_logger.info("[Agent Parse FALLBACK] Heurística de parsing adaptativo registrada.")
                except Exception as log_err:
                    agent_logger.warning(f"[Agent Parse FALLBACK] Falha ao registrar heurística de parsing: {log_err}")
                return thought, action_fallback, action_input_fallback
            else:
                agent_logger.error(
                    f"[Agent Parse ERROR] 'Action:' or 'Final Answer:' section not found in LLM response, e fallback também falhou.\nResponse:\n{response}"
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
                        # Replace curly quotes
                        cleaned_input_str = action_input_str.replace("\u201c", '"').replace("\u201d", '"') # " " -> "
                        if cleaned_input_str != action_input_str:
                            agent_logger.debug("[Agent Parse DEBUG] Replaced curly quotes in Action Input string.")

                        # Clean potential markdown
                        if cleaned_input_str.startswith("```json"):
                            cleaned_input_str = (
                                cleaned_input_str.removeprefix("```json")
                                .removesuffix("```")
                                .strip()
                            )
                        elif cleaned_input_str.startswith("```"):
                            cleaned_input_str = (
                                cleaned_input_str.removeprefix("```")
                                .removesuffix("```")
                                .strip()
                            )

                        decoder = json.JSONDecoder()
                        action_input, end_pos = decoder.raw_decode(cleaned_input_str)
                        agent_logger.debug(
                            f"[Agent Parse DEBUG] raw_decode parsed JSON up to position {end_pos}. Parsed object: {action_input}"
                        )
                        if end_pos < len(cleaned_input_str.strip()):
                            trailing_data = cleaned_input_str[end_pos:].strip()
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
