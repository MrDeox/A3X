import requests
import logging
import json
import re
from typing import Dict, Any

# Use configurações centralizadas
from core.config import LLAMA_SERVER_URL, LLAMA_DEFAULT_HEADERS

# Configure logger para esta skill
logger = logging.getLogger(__name__)

def skill_generate_code(action_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gera código usando o LLM com base em uma descrição (purpose).

    Args:
        action_input (Dict[str, Any]): Dicionário contendo:
            - purpose (str): Descrição do que o código deve fazer (obrigatório).
            - language (str, optional): Linguagem de programação. Padrão: 'python'.
            - construct_type (str, optional): Tipo de construção (ex: function, class). Padrão: 'function'.
            - context (str, optional): Contexto adicional ou código existente para referência.

    Returns:
        Dict[str, Any]: Dicionário com status, action, e data (contendo o código gerado).
    """
    logger.info(f"Executando skill_generate_code com input: {action_input}")

    # Extrair parâmetros
    purpose = action_input.get("purpose")
    language = action_input.get("language", "python")
    construct_type = action_input.get("construct_type", "function")
    context = action_input.get("context") # Novo parâmetro opcional para contexto

    # Validar parâmetros obrigatórios
    if not purpose:
        logger.error("Parâmetro obrigatório 'purpose' não fornecido.")
        return {
            "status": "error",
            "action": "code_generation_failed",
            "data": {"message": "Erro: O propósito do código (purpose) não foi especificado."}
        }

    try:
        # Construir o prompt para o LLM de geração de código
        prompt_lines = [
            f"Gere um código conciso e funcional em {language} que sirva para: {purpose}.",
            f"O código deve ser uma {construct_type}."
        ]
        if context:
            prompt_lines.append(f"\nConsidere o seguinte contexto ou código existente:\n```\n{context}\n```")

        prompt_lines.append("\nRetorne APENAS o bloco de código bruto na linguagem solicitada, sem nenhuma explicação, introdução, ou formatação extra como ```markdown. APENAS O CÓDIGO.")
        prompt = "\n".join(prompt_lines)

        logger.debug(f"Prompt para LLM de geração de código:\n{prompt}")

        # Preparar payload para o LLM (pode ser diferente do ReAct)
        # Usamos /v1/completions aqui? Ou /v1/chat/completions ainda serve?
        # Vamos manter /v1/chat/completions por simplicidade, mas talvez ajustar parâmetros.
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4, # Temperatura mais baixa para código mais determinístico
            "max_tokens": 1024, # Permitir código mais longo
            "stream": False,
            # "stop": [...] # Pode ser útil definir stops para evitar explicações
        }

        # Enviar para o LLM
        # Nota: Usando LLAMA_DEFAULT_HEADERS que pode conter Auth se configurado
        response = requests.post(
            LLAMA_SERVER_URL, # Assume que a URL base é a mesma
            headers=LLAMA_DEFAULT_HEADERS,
            json=payload,
            timeout=120 # Timeout para geração de código
        )

        response.raise_for_status() # Levanta erro para status >= 400

        # Extrair o código da resposta (assumindo formato chat/completions)
        response_data = response.json()
        code = "" # Inicializa code
        if 'choices' in response_data and response_data['choices']:
             message = response_data['choices'][0].get('message', {})
             raw_code = message.get('content', '').strip()
             if raw_code: # Procede somente se raw_code não for vazio
                 # Tenta remover ```<lang>...``` se o LLM ainda os adicionar
                 temp_code = re.sub(r"^```(?:[a-zA-Z]+)?\n?", "", raw_code, flags=re.MULTILINE)
                 code = re.sub(r"\n?```$", "", temp_code, flags=re.MULTILINE).strip()
                 logger.debug(f"Código após limpeza de markdown: {code[:100]}...")
             else:
                 logger.warning("Campo 'content' da resposta LLM está vazio.")
                 code = "" # Garante que code é vazio se content for vazio
        else:
             logger.error(f"Resposta inesperada do LLM (sem 'choices' válidos): {response_data}")
             raise ValueError("Formato de resposta inesperado do LLM para geração de código.")

        # Verifica se o código final está vazio APÓS o processamento
        if not code:
             logger.warning("LLM retornou código vazio ou a limpeza resultou em vazio.")
             return {
                "status": "error",
                "action": "code_generation_failed",
                "data": {"message": "LLM retornou uma resposta vazia ou inválida ao gerar o código."}
             }

        logger.info(f"Código gerado com sucesso em {language}.")
        return {
            "status": "success",
            "action": "code_generated",
            "data": {
                "code": code,
                "language": language,
                "construct_type": construct_type,
                "message": f"Código gerado com sucesso em {language}."
            }
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Erro de rede ao chamar LLM para gerar código: {e}")
        return {
            "status": "error",
            "action": "code_generation_failed",
            "data": {"message": f"Erro de comunicação com o servidor LLM: {e}"}
        }
    except Exception as e:
        logger.exception("Erro inesperado ao gerar código:") # Log com traceback
        return {
            "status": "error",
            "action": "code_generation_failed",
            "data": {"message": f"Erro inesperado durante a geração de código: {str(e)}"}
        } 