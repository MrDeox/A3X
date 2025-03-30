import logging
import requests
import json
import datetime
import os
import sys
import traceback
from typing import List, Dict, Optional

# Local imports (assuming config is accessible from core)
from .config import LLAMA_SERVER_URL, LLAMA_DEFAULT_HEADERS

# Initialize logger for this module
llm_logger = logging.getLogger(__name__)

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

async def call_llm(messages: List[Dict[str, str]], llm_url: Optional[str] = None) -> str:
    """
    Chama a API do LLM (agora em Llama.cpp server) com a lista de mensagens.

    Args:
        messages: Lista de dicionários de mensagens (formato OpenAI).
        llm_url: URL opcional da API LLM (padrão para LLAMA_SERVER_URL de config).

    Returns:
        A resposta do LLM como string.

    Raises:
        requests.exceptions.RequestException: Se houver erro na chamada HTTP.
        ValueError: Se a resposta da API não for JSON válido ou não tiver o conteúdo esperado.
    """
    target_url = llm_url if llm_url else LLAMA_SERVER_URL
    if not target_url:
        llm_logger.error("LLM API URL não configurada (LLAMA_SERVER_URL).")
        raise ValueError("LLM API URL não está configurada.")

    payload = {
        "messages": messages,
        "temperature": 0.7, # Ajuste conforme necessário
        "max_tokens": 2048, # Ajuste conforme necessário
        # Adicione outros parâmetros suportados pelo servidor Llama.cpp se necessário
        # "stop": ["Observation:"] # Exemplo, se o servidor suportar
    }
    headers = LLAMA_DEFAULT_HEADERS

    llm_logger.debug(f"Enviando para LLM: URL={target_url}, Payload Messages Count={len(messages)}")
    if llm_logger.isEnabledFor(logging.DEBUG):
        try:
            # Log detalhado apenas se DEBUG estiver ativo
            detailed_payload_log = json.dumps(payload, indent=2, ensure_ascii=False)
            llm_logger.debug(f"Payload Detalhado:\n{detailed_payload_log}")
        except Exception as log_e:
            llm_logger.warning(f"Falha ao serializar payload para log detalhado: {log_e}")


    start_time = datetime.datetime.now()
    try:
        # Aumenta o timeout para 300 segundos (5 minutos)
        response = requests.post(target_url, headers=headers, json=payload, timeout=300)
        response.raise_for_status() # Levanta erro para status HTTP 4xx/5xx

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        llm_logger.info(f"LLM call successful. Duration: {duration:.3f}s")

        # Tentar decodificar JSON
        try:
            response_data = response.json()
        except json.JSONDecodeError as json_err:
            llm_logger.error(f"Falha ao decodificar JSON da resposta LLM. Status: {response.status_code}, Raw Response: {response.text[:500]}...")
            raise ValueError(f"Resposta LLM não é JSON válido: {json_err}") from json_err

        # Extrair conteúdo da resposta (ajustar conforme a estrutura da API do Llama.cpp server)
        # Exemplo: se a resposta for como OpenAI: response_data['choices'][0]['message']['content']
        # Exemplo: se for um campo 'content' direto: response_data['content']
        if 'choices' in response_data and isinstance(response_data['choices'], list) and len(response_data['choices']) > 0:
            message = response_data['choices'][0].get('message', {})
            llm_content = message.get('content')
            if llm_content:
                return llm_content.strip()
            else:
                llm_logger.error(f"Estrutura de resposta LLM inesperada (sem 'content' em 'message'). Data: {response_data}")
                raise ValueError("Resposta LLM não contém 'content' esperado.")
        elif 'content' in response_data: # Tenta um campo 'content' direto
             llm_content = response_data.get('content')
             if llm_content and isinstance(llm_content, str):
                 return llm_content.strip()
             else:
                 llm_logger.error(f"Estrutura de resposta LLM inesperada (campo 'content' não é string ou vazio). Data: {response_data}")
                 raise ValueError("Resposta LLM não contém 'content' string esperado.")
        else:
            llm_logger.error(f"Estrutura de resposta LLM inesperada (sem 'choices' ou 'content'). Data: {response_data}")
            raise ValueError("Resposta LLM não tem a estrutura esperada ('choices' ou 'content').")

    except requests.exceptions.Timeout:
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        llm_logger.error(f"LLM call timed out after {duration:.3f}s. URL: {target_url}")
        raise requests.exceptions.Timeout(f"Timeout na chamada LLM ({duration:.3f}s)")
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

