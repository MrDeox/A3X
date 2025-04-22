# skills/modify_code.py
import requests
import logging
import json
import re
import os
import ast
from typing import Dict, Any, Optional

# Import skill decorator and Context
from a3x.core.skills import skill
from a3x.core.context import Context

# from core.tools import skill # Remove old commented import
# from core.skills_utils import create_skill_response # Remove old commented import
# from core.code_safety import is_safe_ast # Import safety check # Keep commented for now if unused

# Use absolute import for config, handle potential ImportError
try:
    from a3x.core.config import LLAMA_SERVER_URL, LLAMA_DEFAULT_HEADERS
except ImportError:
    # logging.warning("Could not import config from core.config. Using default LLM URL.") # Logged below
    LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions" # Provide a default
    LLAMA_DEFAULT_HEADERS = { "Content-Type": "application/json" }

# Re-enable logger after potential config import
logger = logging.getLogger(__name__)
# Log warning here if import failed
if 'LLAMA_SERVER_URL' not in globals():
     logging.warning("Could not import config from a3x.core.config. Using default LLM URL.")

@skill(
    name="modify_code",
    description="Modifica um trecho de código com base em instruções fornecidas usando um LLM.",
    parameters={
        "type": "object",
        "properties": {
            "modification": {
                "type": "str",
                "description": "Descrição clara de como modificar o código."
            },
            "code_to_modify": {
                "type": "str",
                "description": "O código original completo a ser modificado."
            },
            "language": {
                "type": "str",
                "description": "Linguagem de programação (padrão: python).",
                "default": "python"
            }
        },
        "required": ["modification", "code_to_modify"]
    }
)
def modify_code(
    context: Context, # Add context argument
    modification: str,
    code_to_modify: str,
    language: str = "python"
    # Remove action_input: Dict[str, Any],
    # Remove agent_memory: dict = None,
    # Remove agent_history: list | None = None,
) -> Dict[str, Any]:
    """
    Modifica um bloco de código existente usando o LLM com base em instruções.

    Args:
        context (Context): O contexto de execução da skill.
        modification (str): Descrição clara de como modificar o código (obrigatório).
        code_to_modify (str): O código original completo a ser modificado (obrigatório).
        language (str, optional): Linguagem de programação. Padrão: 'python'.

    Returns:
        Dict[str, Any]: Dicionário com status, action, e data (contendo código original e modificado).
    """
    logger.info("Executando skill modify_code...")
    # Access parameters directly
    # logger.debug(f"Action Input: {action_input}") # Remove old debug log

    # Extrair parâmetros (já são argumentos da função)
    # modification = action_input.get("modification")
    # original_code = action_input.get("code_to_modify")
    # language = action_input.get("language", "python")
    original_code = code_to_modify # Use argument directly

    # Validar parâmetros obrigatórios (já tratados pelo schema)
    # if not modification: ... # Remove validation block
    # if (not original_code): ... # Remove validation block

    try:
        # Construir o prompt para o LLM de modificação de código
        prompt = f"""You are an expert code editor for the {language} language.
Apply the following modification to the provided code:
Modification: {modification}

Original Code:
```{language}
{original_code}
```

Return ONLY the complete, modified code block in {language}, without any explanations or markdown formatting."""
        logger.debug(f"Prompt para LLM de modificação de código:\n{prompt}")

        # Preparar payload para o LLM
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,  # Temperatura baixa para modificações precisas
            "max_tokens": 2048,  # Permitir modificações em códigos mais longos
            "stream": False,
        }

        # Enviar para o LLM
        logger.info(f"Enviando requisição para LLM em {LLAMA_SERVER_URL}")
        response = requests.post(
            LLAMA_SERVER_URL,
            headers=LLAMA_DEFAULT_HEADERS,
            json=payload,
            timeout=150,  # Timeout um pouco maior para modificação
        )
        response.raise_for_status()  # Levanta erro para status >= 400

        # Extrair o código modificado da resposta
        response_data = response.json()
        modified_code = ""  # Inicializa
        if "choices" in response_data and response_data["choices"]:
            message = response_data["choices"][0].get("message", {})
            raw_modified_code = message.get("content", "").strip()
            if raw_modified_code:
                # Limpeza de markdown (melhor esforço)
                temp_code = re.sub(
                    r"^```(?:[a-zA-Z]+)?\n?", "", raw_modified_code, flags=re.MULTILINE
                )
                modified_code = re.sub(
                    r"\n?```$", "", temp_code, flags=re.MULTILINE
                ).strip()
                logger.debug(
                    f"Código modificado após limpeza: {modified_code[:100]}..."
                )
            else:
                logger.warning(
                    "Campo 'content' da resposta LLM está vazio ao modificar código."
                )
                modified_code = ""  # Garante que é vazio se content for vazio
        else:
            logger.error(
                f"Resposta inesperada do LLM ao modificar código: {response_data}"
            )
            raise ValueError(
                "Formato de resposta inesperado do LLM para modificação de código."
            )

        # Verificar se o código retornado está vazio
        if not modified_code:
            logger.warning(
                "LLM retornou código modificado vazio ou limpeza resultou em vazio."
            )
            # Pode ser um erro ou o LLM deletou tudo? Retornar erro por segurança.
            return {
                "status": "error",
                "action": "modify_code_failed",
                "data": {
                    "message": "LLM retornou uma resposta vazia ou inválida ao modificar o código."
                },
            }

        # Verificar se o código realmente mudou (removendo espaços em branco para comparação)
        if modified_code.strip() == original_code.strip():
            logger.info("Código modificado retornado pelo LLM é idêntico ao original.")
            return {
                "status": "no_change",
                "action": "code_modification_no_change",
                "data": {
                    "original_code": original_code,
                    "modified_code": modified_code,  # Retorna o mesmo código
                    "language": language,
                    "message": "LLM não aplicou a modificação solicitada (código retornado é idêntico ao original).",
                },
            }
        else:
            logger.info(f"Código modificado com sucesso em {language}.")
            return {
                "status": "success",
                "action": "code_modified",
                "data": {
                    "original_code": original_code,
                    "modified_code": modified_code,
                    "language": language,
                    "message": "Código modificado com sucesso.",
                },
            }

    except requests.exceptions.RequestException as e:
        logger.error(f"Erro de rede ao chamar LLM para modificar código: {e}")
        return {
            "status": "error",
            "action": "modify_code_failed",
            "data": {"message": f"Erro de comunicação com o servidor LLM: {e}"},
        }
    except (
        json.JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
    ) as e:  # Captura erros de parsing e formato
        logger.error(
            f"Erro ao processar resposta do LLM: {e}. Resposta: {response_data if 'response_data' in locals() else 'N/A'}"
        )
        return {
            "status": "error",
            "action": "modify_code_failed",
            "data": {"message": f"Erro ao processar resposta do LLM: {str(e)}"},
        }
    except Exception as e:
        logger.exception("Erro inesperado ao modificar código:")  # Log com traceback
        return {
            "status": "error",
            "action": "modify_code_failed",
            "data": {
                "message": f"Erro inesperado durante a modificação de código: {str(e)}"
            },
        }
