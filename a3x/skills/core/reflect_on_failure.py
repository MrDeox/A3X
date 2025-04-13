import logging
import json
import datetime
from typing import Dict, Any, List, Optional
from a3x.core.skills import skill
# Import the class and default URL, not the function
from a3x.core.llm_interface import LLMInterface, DEFAULT_LLM_URL
from a3x.core.config import LEARNING_LOGS_DIR, HEURISTIC_LOG_FILE
from a3x.core.context import Context
import os

reflect_logger = logging.getLogger(__name__)

HEURISTIC_LOG_PATH = os.path.join(LEARNING_LOGS_DIR, HEURISTIC_LOG_FILE)

async def _log_learned_heuristic(log_entry: Dict[str, Any]):
    # (Same logging helper function as in reflect_on_success)
    try:
        log_dir = os.path.dirname(HEURISTIC_LOG_PATH)
        os.makedirs(log_dir, exist_ok=True)
        with open(HEURISTIC_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        reflect_logger.info(f"Logged heuristic: {log_entry.get('heuristic', '')[:50]}...")
    except Exception as e:
        reflect_logger.exception("Failed to log learned heuristic:")

REFLECTION_PROMPT_TEMPLATE_FAILURE = """
# Reflexão Pós-Execução (Falha)

**Objetivo:** {objective}

**Plano Original:**
{plan_str}

**Erro(s) Encontrado(s):**
{errors_str}

**Resultado Final:** Falha

**Tarefa:** Analise a execução falha acima. Identifique a causa raiz do erro ou a sequência de ações que levou à falha. Com base nisso, formule UMA ÚNICA heurística NEGATIVA e ACIONÁVEL (uma regra geral ou dica sobre o que *evitar*) que possa ser usada para prevenir falhas semelhantes em situações futuras. A heurística deve ser concisa (1-2 frases).

**Heurística Gerada (O que evitar):**
"""

@skill(
    name="reflect_on_failure",
    description="Reflete sobre uma execução falha para extrair heurísticas preventivas.",
    parameters={
        "objective": (str, ...),
        "plan": (List[str], ...),
        "execution_results": (List[Dict[str, Any]], ...)
    }
)
async def reflect_on_failure(
    ctx: Context,
    objective: str,
    plan: List[str],
    execution_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Reflects on a failed execution to extract preventative heuristics."""
    reflect_logger.info(f"Reflecting on failure for objective: {objective[:100]}...")

    # Prepare input for the LLM
    plan_str = "\n".join([f"- {step}" for step in plan])
    # Focus on the errors
    errors_str = "\n".join([f"- Passo {i+1} ({res.get('action', 'N/A')}): Status={res.get('status')}, Erro={str(res.get('output', 'N/A'))[:100]}..." for i, res in enumerate(execution_results) if res.get('status') != 'success'])
    if not errors_str:
        errors_str = "(Nenhum erro detalhado registrado nos resultados)"

    prompt = REFLECTION_PROMPT_TEMPLATE_FAILURE.format(
        objective=objective,
        plan_str=plan_str,
        errors_str=errors_str
    )

    # Get LLMInterface instance from context or create a fallback
    if hasattr(ctx, 'llm_interface') and isinstance(ctx.llm_interface, LLMInterface):
        llm_interface = ctx.llm_interface
        reflect_logger.debug("Using LLMInterface from context for failure reflection.")
    else:
        llm_url = getattr(ctx, 'llm_url', DEFAULT_LLM_URL)
        reflect_logger.warning(f"LLMInterface not found in context. Creating temporary instance for failure reflection with URL: {llm_url}")
        llm_interface = LLMInterface(llm_url=llm_url)

    # Parameters for the call
    llm_call_params = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 100, # Heuristics should be concise
        # temperature can be added here if needed
    }

    reflect_logger.debug("Calling LLM for failed execution reflection...")
    heuristic_text = ""
    try:
        response_content = ""
        # Use the instance method to make the call
        async for chunk in llm_interface.call_llm(**llm_call_params):
            response_content += chunk
        heuristic_text = response_content.strip().strip('"\'\n ') # Clean up response

        if heuristic_text:
            reflect_logger.info(f"Generated heuristic from failure: {heuristic_text}")
            log_entry = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + 'Z',
                "objective": objective,
                "plan": plan,
                "results": execution_results,
                "heuristic": heuristic_text,
                "type": "failure" # Mark as failure heuristic
            }
            await _log_learned_heuristic(log_entry)
            return {"status": "success", "data": {"heuristic": heuristic_text}}
        else:
            reflect_logger.warning("LLM did not generate a heuristic for the failed execution.")
            return {"status": "warning", "data": {"message": "No heuristic generated."}}

    except Exception as e:
        reflect_logger.exception("Error during LLM call for failure reflection:")
        return {"status": "error", "data": {"message": f"LLM call failed: {e}"}}

# Example usage (for testing) can be added here if needed
# Similar to reflect_on_success, creating a MockContext and test data. 