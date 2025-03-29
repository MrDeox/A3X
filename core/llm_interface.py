import requests
import json
import logging
import os
from typing import List, Dict, Any, Optional

# Configurações básicas (podem ser movidas para core/config.py depois)
LLAMA_DEFAULT_HEADERS = {
    "Content-Type": "application/json"
}
LLM_TIMEOUT = 180 # Segundos

# Logger para este módulo
logger = logging.getLogger(__name__)

# <<< CARREGAR REACT SCHEMA (Movido de agent.py) >>>
SCHEMA_FILE_PATH = os.path.join(os.path.dirname(__file__), 'react_output_schema.json')
REACT_SCHEMA = None # Renomeado de LLM_RESPONSE_SCHEMA
try:
    with open(SCHEMA_FILE_PATH, 'r', encoding='utf-8') as f:
        REACT_SCHEMA = json.load(f)
    logger.info(f"[LLM Interface] React JSON Schema carregado de {SCHEMA_FILE_PATH}")
except FileNotFoundError:
    logger.error(f"[LLM Interface ERROR] React JSON Schema não encontrado em {SCHEMA_FILE_PATH}.")
except json.JSONDecodeError as e:
    logger.error(f"[LLM Interface ERROR] Erro ao decodificar React JSON Schema de {SCHEMA_FILE_PATH}: {e}.")
except Exception as e:
    logger.error(f"[LLM Interface ERROR] Erro inesperado ao carregar React JSON Schema de {SCHEMA_FILE_PATH}: {e}.")
# <<< FIM CARREGAMENTO SCHEMA >>>

def call_llm(llm_url: str, messages: List[Dict[str, Any]], force_json_output: bool = False) -> str:
    """Chama o LLM local com a lista de mensagens. Força saída JSON se force_json_output=True E o REACT_SCHEMA foi carregado."""
    payload = {
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 1500, # Considerar se o planejamento precisa de mais tokens
        "stream": False
    }

    # <<< Use REACT_SCHEMA if force_json_output is True and schema is loaded >>>
    if force_json_output:
        if REACT_SCHEMA:
            payload["response_format"] = {
                "type": "json_object",
                "schema": REACT_SCHEMA
            }
            logger.debug(f"[LLM Interface] Forçando resposta JSON com schema React.")
        else:
            logger.warning("[LLM Interface] force_json_output=True mas o React Schema não foi carregado. Não é possível forçar formato.")
    else:
        logger.debug("[LLM Interface] Não forçando schema JSON.")

    headers = LLAMA_DEFAULT_HEADERS
    logger.info(f"[LLM Interface] Enviando requisição para: {llm_url}")
    # logger.debug(f"[LLM Interface] Payload: {json.dumps(payload, indent=2)}") # Debug

    try:
        response = requests.post(llm_url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
        response.raise_for_status() # Lança exceção para 4xx/5xx
        response_data = response.json()
        logger.debug(f"[LLM Interface] Resposta recebida: {response_data}")

        if 'choices' in response_data and response_data['choices']:
            message_content = response_data['choices'][0].get('message', {}).get('content', '').strip()
            if message_content:
                # Se um schema foi FORÇADO e CARREGADO, valida se a resposta é um JSON válido
                if force_json_output and REACT_SCHEMA:
                    try:
                        json.loads(message_content) # Validação
                        logger.debug("[LLM Interface] Resposta JSON validada com sucesso.")
                    except json.JSONDecodeError as e:
                        logger.error(f"[LLM Interface ERROR] LLM response is not valid JSON despite schema enforcement: {e}. Content: {message_content[:500]}...")
                        return f'Error: LLM returned invalid JSON: {e}' # Retorna erro específico
                # Retorna o conteúdo (seja JSON validado ou texto normal)
                return message_content
            else:
                logger.error(f"[LLM Interface ERROR] LLM response OK, but 'content' is empty. Response: {response_data}")
                return "Error: LLM returned empty content."
        else:
            logger.error(f"[LLM Interface ERROR] LLM response OK, but unexpected format. Response: {response_data}")
            return "Error: LLM returned unexpected response format."

    except requests.exceptions.Timeout as e:
        logger.error(f"[LLM Interface ERROR] Request timed out contacting LLM at {llm_url}: {e}")
        return f"Error: LLM request timed out ({e})."
    except requests.exceptions.RequestException as e:
        logger.error(f"[LLM Interface ERROR] Failed to connect/communicate with LLM at {llm_url}: {e}")
        return f"Error: Failed to connect to LLM server ({e})."
    except json.JSONDecodeError as e:
         # Se o response.json() falhar
         logger.error(f"[LLM Interface ERROR] Failed to decode LLM server response as JSON: {e}. Response text: {response.text[:500]}...")
         return f"Error: Failed to decode LLM server response ({e})."
    except Exception as e:
        logger.exception("[LLM Interface ERROR] Unexpected error during LLM call:")
        return f"Error: Unexpected error during LLM call ({e})."
