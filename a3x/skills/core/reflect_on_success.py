import logging
import json
import datetime
from typing import Dict, Any, List, Optional
from a3x.core.skills import skill
from a3x.core.llm_interface import LLMInterface, DEFAULT_LLM_URL
from a3x.core.config import LEARNING_LOGS_DIR, HEURISTIC_LOG_FILE
from a3x.core.context import Context, SharedTaskContext
import os
from pathlib import Path

# <<< ADDED Imports for type resolution >>>
from a3x.fragments.base import BaseFragment
from a3x.fragments.manager_fragment import ManagerFragment

reflect_logger = logging.getLogger(__name__)

HEURISTIC_LOG_PATH = os.path.join(LEARNING_LOGS_DIR, HEURISTIC_LOG_FILE)

async def _log_learned_heuristic(log_entry: Dict[str, Any]):
    """Appends a learned heuristic to the log file."""
    try:
        # Ensure directory exists
        log_dir = os.path.dirname(HEURISTIC_LOG_PATH)
        os.makedirs(log_dir, exist_ok=True)
        with open(HEURISTIC_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        reflect_logger.info(f"Logged heuristic: {log_entry.get('heuristic', '')[:50]}...")
    except Exception as e:
        reflect_logger.exception("Failed to log learned heuristic:")

REFLECTION_PROMPT_TEMPLATE_SUCCESS = """
# Reflexão Pós-Execução (Sucesso)

**Objetivo:** {objective}

**Plano Executado:**
{plan_str}

**Resultados Detalhados:**
{results_str}

**Contexto Compartilhado Final:**
{context_summary}

**Resultado Final:** Sucesso

**Tarefa:** Analise a execução bem-sucedida acima, incluindo o uso do Contexto Compartilhado. Identifique o fator chave ou a sequência de ações mais importante que levou ao sucesso. Considere como o contexto compartilhado ajudou (ou não). Com base nisso, formule UMA ÚNICA heurística POSITIVA e ACIONÁVEL (uma regra geral ou dica) que possa ser usada em situações futuras semelhantes. A heurística deve ser concisa (1-2 frases).

**Heurística Gerada:**
"""

@skill(
    name="reflect_on_success",
    description="Reflete sobre uma execução bem-sucedida para extrair heurísticas e aprendizados.",
    parameters={
        "objective": {"type": str, "description": "O objetivo geral da tarefa que foi executada."},
        "plan": {"type": List[str], "description": "A sequência de passos (ações/skills) que foram executadas."},
        "execution_results": {"type": List[Dict[str, Any]], "description": "Uma lista de dicionários, cada um representando o resultado de um passo do plano."},
    }
)
async def reflect_on_success(
    ctx: Context,
    objective: str,
    plan: List[str],
    execution_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Reflects on a successful execution to extract heuristics."""
    reflect_logger.info(f"Reflecting on success for objective: {objective[:100]}...")

    # Access SharedTaskContext from ctx
    shared_task_context = getattr(ctx, 'shared_task_context', None)

    # Prepare input for the LLM
    plan_str = "\n".join([f"- {step}" for step in plan])
    results_str = "\n".join([f"- Passo {i+1}: Status={res.get('status')}, Saída={str(res.get('output', 'N/A'))[:100]}..." for i, res in enumerate(execution_results)])

    # Prepare shared context summary for prompt
    context_summary = "(No shared context available or empty)"
    if shared_task_context:
        all_context_entries = shared_task_context.get_all_entries()
        if all_context_entries:
            summary_lines = ["Shared Context Snapshot:"]
            for key, entry in all_context_entries.items():
                # Basic summary: key, value (truncated), source, tags
                value_str = str(entry.value)[:70] + ('...' if len(str(entry.value)) > 70 else '')
                summary_lines.append(f"  - {key}: Value='{value_str}', Source='{entry.source}', Tags={entry.tags}")
            context_summary = "\n".join(summary_lines)

    prompt = REFLECTION_PROMPT_TEMPLATE_SUCCESS.format(
        objective=objective,
        plan_str=plan_str,
        results_str=results_str,
        context_summary=context_summary
    )

    # Get LLMInterface instance from context or create a fallback
    if hasattr(ctx, 'llm_interface') and isinstance(ctx.llm_interface, LLMInterface):
        llm_interface = ctx.llm_interface
        reflect_logger.debug("Using LLMInterface from context.")
    else:
        llm_url = getattr(ctx, 'llm_url', DEFAULT_LLM_URL)
        reflect_logger.warning(f"LLMInterface not found in context. Creating temporary instance with URL: {llm_url}")
        llm_interface = LLMInterface(llm_url=llm_url)

    # Parameters for the call (specific generation params)
    llm_call_params = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 100,
        # temperature could be added here if required by the LLM API and supported by the interface
    }

    reflect_logger.debug("Calling LLM for successful execution reflection...")
    heuristic_text = ""
    try:
        response_content = ""
        # Use the llm_interface instance method to make the call
        async for chunk in llm_interface.call_llm(**llm_call_params):
            response_content += chunk
        # Clean up response: remove leading/trailing whitespace, quotes, and newlines
        heuristic_text = response_content.strip().strip('"\'\n ')

        if heuristic_text:
            reflect_logger.info(f"Generated heuristic from success: {heuristic_text}")
            log_entry = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + 'Z',
                "objective": objective,
                "plan": plan,
                "results": execution_results, # Consider summarizing or truncating results if large
                "heuristic": heuristic_text,
                "type": "success"
            }
            await _log_learned_heuristic(log_entry)
            return {"status": "success", "data": {"heuristic": heuristic_text}}
        else:
            reflect_logger.warning("LLM did not generate a heuristic for the successful execution.")
            return {"status": "warning", "data": {"message": "No heuristic generated."}}

    except Exception as e:
        reflect_logger.exception("Error during LLM call for success reflection:")
        return {"status": "error", "data": {"message": f"LLM call failed: {e}"}}

# Example usage (for testing)
async def main_test():
    logging.basicConfig(level=logging.DEBUG)
    reflect_logger.info("Running reflect_on_success test...")

    # Mock Context
    class MockContext:
        def __init__(self):
            # Simulate having an llm_interface attribute
            self.llm_interface = LLMInterface(llm_url=os.getenv("LLM_API_URL", DEFAULT_LLM_URL))
            self.logger = reflect_logger
            self.workspace_root = Path(".")
            # Mock SharedTaskContext for testing
            self.shared_task_context = SharedTaskContext(task_id="test-task", initial_objective="test")
            self.shared_task_context.set("initial_data", {"value": 1}, source="setup")
            self.shared_task_context.set("processed_data", {"value": 2}, source="step1", tags=["intermediate"])

    mock_ctx = MockContext()

    # Mock data
    test_objective = "Write a simple greeting function in Python."
    test_plan = [
        "write_file(path='greeting.py', content='''def hello():\n  print(\"Hello\")''')",
        "execute_code(code='import greeting\ngreeting.hello()')"
    ]
    
    test_results = [
        {"status": "success", "output": "File written.", "action": "write_file"},
        {"status": "success", "output": "Hello", "action": "execute_code"}
    ]

    result = await reflect_on_success(
        ctx=mock_ctx,
        objective=test_objective,
        plan=test_plan,
        execution_results=test_results
    )

    print("\n--- Test Result ---")
    print(json.dumps(result, indent=2))
    print("-------------------")

if __name__ == "__main__":
    import asyncio
    from pathlib import Path
    asyncio.run(main_test()) 