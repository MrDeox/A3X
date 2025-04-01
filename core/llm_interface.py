import logging
import requests
import json
import datetime
import os
import sys
import traceback
from typing import List, Dict, Optional, AsyncGenerator

# Initialize logger for this module *before* first use
llm_logger = logging.getLogger(__name__)

# Local imports (assuming config is accessible from core)
# <<< MODIFIED: Handle potential ImportError or missing LLAMA_SERVER_URL >>>
try:
    from .config import LLAMA_DEFAULT_HEADERS
    # Try importing LLAMA_SERVER_URL, but don't fail if it's missing/commented out
    from .config import LLAMA_SERVER_URL as _CONFIG_LLAMA_URL 
except ImportError:
    LLAMA_DEFAULT_HEADERS = {"Content-Type": "application/json"} # Sensible fallback
    _CONFIG_LLAMA_URL = None
    llm_logger.warning("Could not import config from core.config or LLAMA_SERVER_URL is missing. Using defaults.")

# Define a default URL within the module
_DEFAULT_LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

# <<< ADD REACT_SCHEMA Definition for testing >>>
REACT_SCHEMA = {
    "type": "object",
    "properties": {
        "Thought": {"type": "string"},
        "Action": {"type": "string"},
        "Action Input": {"type": "object"} # Allow any object for input initially
    },
    "required": ["Thought", "Action"]
}

# <<< MODIFIED: Make it an async generator and add stream param >>>
async def call_llm(
    messages: List[Dict[str, str]], 
    llm_url: Optional[str] = None, 
    stream: bool = False
) -> AsyncGenerator[str, None]:
    """
    Chama a API do LLM (Llama.cpp server) com a lista de mensagens.
    Pode operar em modo normal (retorna string completa) ou streaming (gera chunks).

    Args:
        messages: Lista de dicionários de mensagens (formato OpenAI).
        llm_url: URL opcional da API LLM (padrão para LLAMA_SERVER_URL de config).
        stream: Se True, ativa o modo streaming e gera chunks de texto.

    Yields:
        Chunks de texto (string) se stream=True.

    Raises:
        requests.exceptions.RequestException: Se houver erro na chamada HTTP.
        ValueError: Se a resposta da API não for JSON válido (modo não-streaming) ou se stream falhar.
        StopAsyncIteration: Quando o stream termina (se stream=True).
    """
    # <<< MODIFIED: Determine target URL robustly >>>
    # 1. Use llm_url if provided directly
    # 2. Use LLAMA_SERVER_URL from config if imported successfully
    # 3. Use the in-module default _DEFAULT_LLM_URL
    # 4. Use os.getenv as a final fallback (redundant if config loads .env, but safe)
    target_url = llm_url or _CONFIG_LLAMA_URL or os.getenv("LLAMA_SERVER_URL", _DEFAULT_LLM_URL)

    if not target_url:
        llm_logger.error("LLM API URL could not be determined. Check config and .env.")
        raise ValueError("LLM API URL could not be determined.")

    payload = {
        "messages": messages,
        "temperature": 0.7, # Ajuste conforme necessário
        "max_tokens": 2048, # Ajuste conforme necessário
        # <<< ADD stream parameter >>>
        "stream": stream 
        # Adicione outros parâmetros suportados pelo servidor Llama.cpp se necessário
        # "stop": ["Observation:"] # Exemplo, se o servidor suportar
    }
    headers = LLAMA_DEFAULT_HEADERS # Use global default directly

    llm_logger.debug(f"Enviando para LLM: URL={target_url}, Stream={stream}, Payload Messages Count={len(messages)}")
    if llm_logger.isEnabledFor(logging.DEBUG):
        try:
            # Log detalhado apenas se DEBUG estiver ativo
            detailed_payload_log = json.dumps(payload, indent=2, ensure_ascii=False)
            llm_logger.debug(f"Payload Detalhado:\n{detailed_payload_log}")
        except Exception as log_e:
            llm_logger.warning(f"Falha ao serializar payload para log detalhado: {log_e}")


    start_time = datetime.datetime.now()
    try:
        # <<< MODIFIED: Add stream=stream to request >>>
        # Using sync requests for now, async version would need aiohttp
        # TODO: Consider switching to aiohttp for full async compatibility
        response = requests.post(target_url, headers=headers, json=payload, timeout=300, stream=stream)
        response.raise_for_status() # Levanta erro para status HTTP 4xx/5xx

        if stream:
            llm_logger.info(f"LLM stream started. Duration to first byte: {(datetime.datetime.now() - start_time).total_seconds():.3f}s")
            # Handle streaming response (Server-Sent Events)
            full_response_text = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        try:
                            # Remove 'data: ' prefix and parse JSON
                            json_data = json.loads(decoded_line[6:])
                            # Extract content chunk (adjust based on actual server response structure)
                            if 'choices' in json_data and json_data['choices']:
                                delta = json_data['choices'][0].get('delta', {})
                                chunk = delta.get('content')
                                if chunk:
                                    full_response_text += chunk
                                    yield chunk # Yield the content chunk
                            elif 'content' in json_data: # Handle direct content if needed
                                chunk = json_data.get('content')
                                if chunk and isinstance(chunk, str):
                                     full_response_text += chunk
                                     yield chunk
                            # Handle potential stop reason if needed
                            # finish_reason = json_data['choices'][0].get('finish_reason')
                            # if finish_reason:
                            #    llm_logger.info(f"LLM stream finished with reason: {finish_reason}")
                            #    break
                        except json.JSONDecodeError:
                            llm_logger.warning(f"Failed to decode JSON chunk from stream: {decoded_line}")
                            continue # Skip malformed lines
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            llm_logger.info(f"LLM stream finished. Total duration: {duration:.3f}s")
            # For stream=True, we don't return, we just yield. 
            # We could potentially return the full concatenated text after the loop if needed,
            # but the generator pattern implies yielding is the primary output.
            # return full_response_text # Optional: return full text if needed after stream

        else: # Non-streaming mode (existing logic)
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            llm_logger.info(f"LLM call successful (non-streaming). Duration: {duration:.3f}s")
            try:
                response_data = response.json()
            except json.JSONDecodeError as json_err:
                llm_logger.error(f"Falha ao decodificar JSON da resposta LLM. Status: {response.status_code}, Raw Response: {response.text[:500]}...")
                raise ValueError(f"Resposta LLM não é JSON válido: {json_err}") from json_err

            if 'choices' in response_data and isinstance(response_data['choices'], list) and len(response_data['choices']) > 0:
                message = response_data['choices'][0].get('message', {})
                llm_content = message.get('content')
                if llm_content:
                    # In non-streaming, yield a single item (the full response) then stop
                    yield llm_content.strip()
                    return # Necessary to stop the generator
                else:
                    llm_logger.error(f"Estrutura de resposta LLM inesperada (sem 'content' em 'message'). Data: {response_data}")
                    raise ValueError("Resposta LLM não contém 'content' esperado.")
            elif 'content' in response_data:
                 llm_content = response_data.get('content')
                 if llm_content and isinstance(llm_content, str):
                     yield llm_content.strip()
                     return
                 else:
                     llm_logger.error(f"Estrutura de resposta LLM inesperada (campo 'content' não é string ou vazio). Data: {response_data}")
                     raise ValueError("Resposta LLM não contém 'content' string esperado.")
            else:
                llm_logger.error(f"Estrutura de resposta LLM inesperada (sem 'choices' ou 'content'). Data: {response_data}")
                raise ValueError("Resposta LLM não tem a estrutura esperada ('choices' ou 'content').")

    except requests.exceptions.Timeout as e:
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        llm_logger.error(f"LLM call timed out after {duration:.3f}s. URL: {target_url}")
        raise requests.exceptions.Timeout(f"Timeout na chamada LLM ({duration:.3f}s)") from e
    except requests.exceptions.RequestException as e:
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        llm_logger.exception(f"Erro na chamada HTTP para LLM ({duration:.3f}s). URL: {target_url}. Erro: {e}")
        # Logar a resposta se disponível
        if e.response is not None:
            llm_logger.error(f"LLM Error Response Status: {e.response.status_code}")
            llm_logger.error(f"LLM Error Response Body: {e.response.text[:500]}...") # Limita o tamanho do log
        raise e # Re-levanta a exceção original
    except Exception as e: # Captura outras exceções inesperadas
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        llm_logger.exception(f"Erro inesperado durante o processamento da chamada LLM ({duration:.3f}s). Erro: {e}")
        raise e # Re-levanta a exceção

