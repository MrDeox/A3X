# skills/modify_code.py
import requests
import logging
import json
import re
from typing import Dict, Any

# from core.tools import skill
from a3x.core.tools import skill
# from core.skills_utils import create_skill_response
from a3x.core.skills_utils import create_skill_response
# from core.code_safety import is_safe_ast # Import safety check
from a3x.core.code_safety import is_safe_ast # Import safety check

# Use configurações centralizadas
try:
    from core.config import LLAMA_SERVER_URL, LLAMA_DEFAULT_HEADERS
except ImportError:
    # Fallback se rodando fora do contexto principal ou erro de importação
    logging.warning("Could not import config from core.config. Using default LLM URL.")
    LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
    LLAMA_DEFAULT_HEADERS = {"Content-Type": "application/json"}

# Configure logger para esta skill
logger = logging.getLogger(__name__)


def skill_modify_code(
    action_input: Dict[str, Any],
    agent_memory: dict = None,
    agent_history: list | None = None,
) -> Dict[str, Any]:
    """
    Modifica um bloco de código existente usando o LLM com base em instruções.

    Args:
        action_input (Dict[str, Any]): Dicionário contendo:
            - modification (str): Descrição clara de como modificar o código (obrigatório).
            - code_to_modify (str): O código original completo a ser modificado (obrigatório).
            - language (str, optional): Linguagem de programação. Padrão: 'python'.
        agent_memory (dict, optional): Memória do agente.
        agent_history (list | None, optional): Histórico do agente.

    Returns:
        Dict[str, Any]: Dicionário com status, action, e data (contendo código original e modificado).
    """
    logger.info("Executando skill_modify_code...")
    logger.debug(f"Action Input: {action_input}")

    # Extrair parâmetros
    modification = action_input.get("modification")
    original_code = action_input.get("code_to_modify")
    language = action_input.get("language", "python")

    # Validar parâmetros obrigatórios
    if not modification:
        logger.error("Parâmetro obrigatório 'modification' não fornecido.")
        return {
            "status": "error",
            "action": "modify_code_failed",
            "data": {
                "message": "Erro: A instrução de modificação (modification) não foi especificada."
            },
        }
    if (
        not original_code
    ):  # code_to_modify pode ser string vazia, mas não None ou ausente
        logger.error("Parâmetro obrigatório 'code_to_modify' não fornecido.")
        return {
            "status": "error",
            "action": "modify_code_failed",
            "data": {
                "message": "Erro: O código a ser modificado (code_to_modify) não foi fornecido."
            },
        }

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
