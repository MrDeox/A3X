import logging
import json
import datetime
from typing import Dict, Any, List, Optional
from a3x.core.skills import skill
# Import the class and default URL, not the function
from a3x.core.llm_interface import LLMInterface, DEFAULT_LLM_URL
from a3x.core.config import LEARNING_LOGS_DIR, HEURISTIC_LOG_FILE, ERROR_LOG_FILE
from a3x.core.context import Context, SharedTaskContext
import os
from pathlib import Path

reflect_logger = logging.getLogger(__name__)

# Log file paths (ensure directory exists later)
HEURISTIC_LOG_PATH = Path(LEARNING_LOGS_DIR) / HEURISTIC_LOG_FILE
ERROR_LOG_PATH = Path(LEARNING_LOGS_DIR) / ERROR_LOG_FILE

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

**Contexto Compartilhado Final:**
{context_summary}

**Resultado Final:** Falha

**Tarefa:** Analise a execução falha acima, incluindo o estado final do Contexto Compartilhado. Identifique a causa raiz do erro ou a sequência de ações que levou à falha. Considere se o contexto compartilhado contém informações que poderiam ter ajudado a evitar a falha, ou se ele contribuiu para o erro. Com base nisso, formule UMA ÚNICA heurística NEGATIVA e ACIONÁVEL (uma regra geral ou dica sobre o que *evitar*) que possa ser usada para prevenir falhas semelhantes em situações futuras. A heurística deve ser concisa (1-2 frases).

**Heurística Gerada (O que evitar):**
"""

@skill(
    name="reflect_on_failure",
    description="Reflete sobre uma execução que falhou para extrair heurísticas de correção.",
    parameters={
        "objective": {"type": str, "description": "O objetivo geral da tarefa que falhou."},
        "failed_step": {"type": str, "description": "A descrição da ação/skill que falhou."},
        "error_message": {"type": str, "description": "A mensagem de erro ou o resultado detalhado da falha."},
        "plan_executed": {"type": List[str], "description": "A sequência de passos executados até a falha."},
        "final_task_context": {"type": Optional[SharedTaskContext], "description": "The final state of the shared task context at the time of failure.", "default": None}
    }
)
async def reflect_on_failure(
    ctx: Context,
    objective: str,
    failed_step: str,
    error_message: str,
    plan_executed: List[str],
    final_task_context: Optional[SharedTaskContext] = None
) -> Dict[str, Any]:
    """Reflects on a failed execution to extract corrective heuristics."""
    reflect_logger.info(f"Reflecting on failure for objective: {objective[:100]}...")

    # Prepare input for the LLM
    plan_str = "\n".join([f"- {step}" for step in plan_executed])
    # Focus on the errors
    errors_str = f"- Falha no passo: {failed_step}\n- Mensagem de erro: {error_message}"

    # Prepare shared context summary for prompt
    context_summary = "(No shared context provided or empty)"
    if final_task_context:
        all_context_entries = final_task_context.get_all_entries()
        if all_context_entries:
            summary_lines = ["Shared Context Snapshot:"]
            for key, entry in all_context_entries.items():
                value_str = str(entry.value)[:70] + ('...' if len(str(entry.value)) > 70 else '')
                summary_lines.append(f"  - {key}: Value='{value_str}', Source='{entry.source}', Tags={entry.tags}")
            context_summary = "\n".join(summary_lines)

    prompt = REFLECTION_PROMPT_TEMPLATE_FAILURE.format(
        objective=objective,
        plan_str=plan_str,
        errors_str=errors_str,
        context_summary=context_summary
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
                "plan": plan_executed,
                "results": [{'action': failed_step, 'status': 'failed', 'output': error_message}],
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