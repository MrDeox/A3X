# skills/final_answer.py
import logging
from typing import Dict, Any
from a3x.core.skills import skill  # Import the decorator

# from core.skills_utils import create_skill_response

logger = logging.getLogger(__name__)


@skill(
    name="final_answer",
    description="Provides the final answer to the user's request when the objective is complete.",
    parameters={"answer": (str, ...)},  # Parâmetro 'answer' é uma string obrigatória
)
def final_answer(answer: str) -> dict:
    """
    Processa a ação final do agente.

    Args:
        answer (str): A resposta final a ser fornecida.

    Returns:
        dict: Dicionário padronizado indicando sucesso e a resposta final.
              {"status": "success", "action": "final_answer_provided",
               "data": {"final_answer": "..."}}
    """
    logger.debug(f"Executing final_answer with answer: {str(answer)[:100]}...")

    # A validação de tipo já é feita pelo Pydantic através do schema no decorador
    # Não precisamos mais de action_input.get ou isinstance

    final_answer_text = answer  # Usar o parâmetro diretamente

    logger.info(f"Final Answer processed: {str(final_answer_text)[:100]}...")

    return {
        "status": "success",
        "action": "final_answer_provided",  # Indica que a resposta foi processada
        "data": {"final_answer": str(final_answer_text)},
    }


# Remover a função antiga se existir (opcional, mas limpa o código)
# del skill_final_answer
