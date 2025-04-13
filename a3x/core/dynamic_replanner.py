import logging
from typing import List, Dict, Any, Optional

from a3x.core.llm_interface import call_llm

logger = logging.getLogger(__name__)

def detect_plan_stuck(execution_results: List[Dict[str, Any]], max_repeats: int = 2) -> bool:
    """
    Detecta se o plano está travado por repetição cega de erro.
    """
    error_messages = [r.get("data", {}).get("message", "") for r in execution_results if r.get("status") == "error"]
    for msg in set(error_messages):
        if error_messages.count(msg) > max_repeats:
            logger.warning(f"[DynamicReplanner] Detected repeated error: '{msg}'")
            return True
    return False

async def dynamic_replan(
    current_plan: List[str],
    execution_results: List[Dict[str, Any]],
    heuristics: Optional[List[Dict[str, Any]]] = None,
    context: Optional[Dict[str, Any]] = None,
    llm_url: Optional[str] = None,
) -> List[str]:
    """
    Gera um novo subplano ou ajusta os passos restantes com base em falhas, heurísticas e contexto.
    """
    logger.info("[DynamicReplanner] Iniciando replanejamento dinâmico...")
    if not detect_plan_stuck(execution_results):
        logger.info("[DynamicReplanner] Plano não está travado. Nenhum replanejamento necessário.")
        return current_plan

    # Monta prompt para o LLM sugerir novo subplano
    prompt = [
        {
            "role": "system",
            "content": (
                "Você é um planejador cognitivo autônomo. O plano atual está travado ou ineficaz. "
                "Analise o histórico de execução, heurísticas e contexto, e gere um novo subplano para atingir o objetivo, "
                "evitando os erros anteriores. Responda apenas com uma lista de passos ReAct, um por linha."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Plano atual:\n{current_plan}\n\n"
                f"Resultados de execução:\n{execution_results}\n\n"
                f"Heurísticas relevantes:\n{heuristics}\n\n"
                f"Contexto:\n{context}\n"
            ),
        },
    ]
    new_plan = []
    try:
        logger.info("[DynamicReplanner] Chamando LLM para gerar novo subplano...")
        response = ""
        async for chunk in call_llm(prompt, llm_url=llm_url, stream=False):
            response += chunk
        logger.info(f"[DynamicReplanner] Resposta do LLM para novo plano:\n{response}")
        # Extrai passos do LLM (um por linha)
        for line in response.splitlines():
            line = line.strip()
            if line and not line.lower().startswith("thought"):
                new_plan.append(line)
        if not new_plan:
            logger.warning("[DynamicReplanner] LLM não retornou novos passos. Mantendo plano original.")
            return current_plan
        return new_plan
    except Exception as e:
        logger.error(f"[DynamicReplanner] Erro ao gerar novo plano: {e}")
        return current_plan