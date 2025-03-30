import json
import logging
from typing import Tuple, Optional, Dict, Any

def parse_llm_response(response: str, agent_logger: logging.Logger) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Parses the LLM's raw response string, expecting it to be a JSON object.
    Raises: json.JSONDecodeError if the response is not valid JSON.
    Returns: thought, action, action_input
    """
    agent_logger.debug(f"[Agent Parse DEBUG] Raw LLM Response (expecting JSON):\n{response}")
    try:
        data = json.loads(response)
        if not isinstance(data, dict):
             agent_logger.error(f"[Agent Parse ERROR] LLM Response is valid JSON but not an object (dict): {type(data)}")
             raise json.JSONDecodeError("Parsed JSON is not an object", response, 0) # Re-raise as decode error

        thought = data.get("Thought")
        action = data.get("Action")
        action_input = data.get("action_input") # Corrected key: lowercase, no space

        # Validação básica
        if not action:
             agent_logger.error(f"[Agent Parse ERROR] Required key 'Action' missing in parsed JSON: {data}")
             # Considerar levantar um erro aqui também ou retornar None para acionar fallback?
             # Por enquanto, vamos logar e retornar None para action, o que deve ser tratado no loop run
             return thought, None, action_input
        if action == "final_answer" and not action_input:
             agent_logger.warning(f"[Agent Parse WARN] 'final_answer' action received without 'action_input'. Creating default. JSON: {data}")
             action_input = {"answer": "Erro: Ação final solicitada sem fornecer a resposta."}
        elif action != "final_answer" and action_input is None:
             agent_logger.info(f"[Agent Parse INFO] Action '{action}' received without 'action_input'. Assuming empty dict. JSON: {data}")
             action_input = {} # Assume dict vazio se não houver input para outras ações

        # Garante que action_input é um dict se não for None
        if action_input is not None and not isinstance(action_input, dict):
             agent_logger.error(f"[Agent Parse ERROR] 'action_input' in JSON is not a dictionary: {type(action_input)}. Content: {action_input}. Treating as empty.")
             action_input = {} # Fallback para dict vazio se o tipo estiver errado

        agent_logger.info(f"[Agent Parse INFO] JSON parsed successfully. Action: '{action}'")
        return thought, action, action_input

    except json.JSONDecodeError as e:
        agent_logger.error(f"[Agent Parse ERROR] Failed to decode LLM response as JSON: {e}")
        agent_logger.debug(f"[Agent Parse DEBUG] Failed JSON content:\n{response}")
        raise e # Re-raise the exception to be caught by the caller (run loop)
    except Exception as e:
         # Captura outros erros inesperados durante o parse/validação
         agent_logger.exception(f"[Agent Parse ERROR] Unexpected error during JSON parsing/validation:")
         # Decide se re-levanta ou retorna None. Re-levantar como JSONDecodeError pode ser consistente.
         raise json.JSONDecodeError(f"Unexpected parsing error: {e}", response, 0) from e
