import logging
import requests
import json
import datetime
import os
from typing import List, Dict, Optional, AsyncGenerator
import time

# Initialize logger for this module *before* first use
llm_logger = logging.getLogger(__name__)

# Local imports (assuming config is accessible from core)
# <<< MODIFIED: Handle potential ImportError or missing LLAMA_SERVER_URL >>>
try:
    # from .config import LLAMA_DEFAULT_HEADERS
    from a3x.core.config import LLAMA_DEFAULT_HEADERS

    # Try importing LLAMA_SERVER_URL, but don't fail if it's missing/commented out
    # from .config import LLAMA_SERVER_URL as _CONFIG_LLAMA_URL
    from a3x.core.config import LLAMA_SERVER_URL as _CONFIG_LLAMA_URL
except ImportError:
    LLAMA_DEFAULT_HEADERS = {"Content-Type": "application/json"}  # Sensible fallback
    _CONFIG_LLAMA_URL = None
    llm_logger.warning(
        "Could not import config from core.config or LLAMA_SERVER_URL is missing. Using defaults."
    )

# Define a default URL within the module
_DEFAULT_LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

# <<< ADD REACT_SCHEMA Definition for testing >>>
REACT_SCHEMA = {
    "type": "object",
    "properties": {
        "Thought": {"type": "string"},
        "Action": {"type": "string"},
        "Action Input": {"type": "object"},  # Allow any object for input initially
    },
    "required": ["Thought", "Action"],
}

DEFAULT_TIMEOUT = 600 # Increased default timeout

def _determine_llm_url(provided_url: Optional[str]) -> str:
    """Determines the LLM URL to use, checking environment variables and defaults."""
    if provided_url:
        return provided_url
    env_url = os.getenv("LLM_API_URL")
    if env_url:
        return env_url
    return _DEFAULT_LLM_URL

# <<< MODIFIED: Make it an async generator and add stream param >>>
async def call_llm(
    messages: List[Dict[str, str]],
    llm_url: Optional[str] = None,
    stream: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> AsyncGenerator[str, None]:
    """
    Chama a API do LLM (compatível com OpenAI Chat Completions) de forma assíncrona.

    Args:
        messages: Lista de dicionários de mensagens (role/content).
        llm_url: URL da API do LLM (opcional, usa config/env/default).
        stream: Se True, retorna um gerador assíncrono para streaming.
                Se False, acumula a resposta e retorna como gerador de um único item.
        timeout: Timeout em segundos para a requisição.

    Yields:
        str: Chunks da resposta do LLM (se stream=True) ou a resposta completa (se stream=False).

    Raises:
        requests.exceptions.RequestException: Para erros de conexão/HTTP.
        requests.exceptions.Timeout: Se o timeout for atingido.
        Exception: Para outros erros inesperados.
    """
    target_url = _determine_llm_url(llm_url)
    headers = {"Content-Type": "application/json"}
    payload = {"messages": messages, "stream": stream}

    llm_logger.debug(f"Enviando para LLM. URL: {target_url}, Stream: {stream}")
    llm_logger.debug(f"Payload (sem stream key): { {k:v for k,v in payload.items() if k != 'stream'} }")

    full_response_content = ""
    start_time = time.time()
    try:
        response = requests.post(
            target_url,
            headers=headers,
            json=payload,
            stream=stream,
            timeout=timeout,
        )
        duration = time.time() - start_time
        response.raise_for_status()  # Levanta erro para status HTTP 4xx/5xx

        if stream:
            llm_logger.info(
                f"LLM stream started. Duration to first byte: {duration:.3f}s"
            )
            # Handle streaming response (Server-Sent Events)
            lines_processed = 0
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[len("data: ") :].strip()
                        if json_str == "[DONE]":
                            llm_logger.debug("LLM stream finished marker [DONE] received.")
                            break
                        try:
                            data = json.loads(json_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    lines_processed += 1
                                    yield content
                        except json.JSONDecodeError:
                            llm_logger.warning(
                                f"LLM stream: Failed to decode JSON line: {json_str}"
                            )
                    elif decoded_line.strip(): # Log other non-empty lines
                         llm_logger.warning(f"LLM stream: Received unexpected line: {decoded_line}")
                # Timeout check within stream loop could be added if needed
            duration = time.time() - start_time # Recalculate total duration
            llm_logger.info(f"LLM stream finished. Lines processed: {lines_processed}. Total duration: {duration:.3f}s")
            # For stream=True, we don't return, we just yield.

        else:  # Non-streaming mode
            llm_logger.info(
                f"LLM call successful (non-streaming). Duration: {duration:.3f}s"
            )
            try:
                response_data = response.json()
                choices = response_data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    full_response_content = message.get("content", "")
                    llm_logger.debug(f"Non-streaming response content length: {len(full_response_content)}")
                else:
                    llm_logger.warning("LLM non-streaming response missing 'choices'.")
            except json.JSONDecodeError as json_err:
                llm_logger.error(f"Failed to decode non-streaming JSON response: {json_err}")
                llm_logger.debug(f"Raw non-streaming response text: {response.text}")
                # Yield an error message or raise?
                yield f"[LLM Response Error: Failed to parse JSON - {json_err}]"
                return # Stop generation

            yield full_response_content # Yield the single full response

    except requests.exceptions.Timeout as e:
        duration = time.time() - start_time
        llm_logger.error(f"LLM call timed out after {duration:.3f}s. URL: {target_url}")
        raise requests.exceptions.Timeout(
            f"Timeout na chamada LLM ({duration:.3f}s)"
        ) from e
    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time
        llm_logger.exception(
            f"Erro na chamada HTTP para LLM ({duration:.3f}s). URL: {target_url}. Erro: {e}"
        )
        # Adicionar log do corpo da resposta, se disponível e útil
        try:
            if e.response is not None:
                 llm_logger.error(f"LLM Error Response Status: {e.response.status_code}")
                 llm_logger.error(f"LLM Error Response Body: {e.response.text[:500]}...") # Log first 500 chars
        except Exception as log_err:
             llm_logger.error(f"Failed to log error response body: {log_err}")

        raise e  # Re-levanta a exceção original
    except Exception as e:  # Captura outras exceções inesperadas
        duration = time.time() - start_time
        llm_logger.exception(
            f"Erro inesperado durante o processamento da chamada LLM ({duration:.3f}s). Erro: {e}"
        )
        raise e # Re-levanta para tratamento no nível superior
