# skills/final_answer.py
import logging

logger = logging.getLogger(__name__)

def skill_final_answer(action_input: dict) -> dict:
    """
    Processa a ação final do agente.

    Args:
        action_input (dict): Dicionário contendo a resposta final.
            Exemplo:
                {"action": "final_answer", "answer": "A tarefa foi concluída."}

    Returns:
        dict: Dicionário padronizado indicando sucesso e a resposta final.
              {"status": "success", "action": "final_answer_provided",
               "data": {"final_answer": "..."}}
    """
    logger.debug(f"Recebido action_input para final_answer: {action_input}")

    final_answer_text = action_input.get("answer", "N/A - Resposta final não fornecida no Action Input.")

    if not isinstance(final_answer_text, str):
         logger.warning(f"Formato inesperado para 'answer' em final_answer: {type(final_answer_text)}. Convertendo para string.")
         final_answer_text = str(final_answer_text)

    logger.info(f"Final Answer processado: {final_answer_text[:100]}...")

    return {
        "status": "success",
        "action": "final_answer_provided",
        "data": {"final_answer": final_answer_text}
    }

