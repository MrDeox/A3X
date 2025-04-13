import logging
import json
import asyncio
from typing import Dict, Any, List

# Core imports
from a3x.core.skills import skill
from a3x.core.learning_logs import load_recent_reflection_logs
from a3x.core.llm_interface import LLMInterface
from a3x.core.agent import _ToolExecutionContext

logger = logging.getLogger(__name__)

@skill(
    name="refine_decision_prompt",
    description="Refina o prompt da skill de simulação de decisões com base nos feedbacks registrados nos últimos logs.",
    parameters={} # No parameters needed besides context
)
async def refine_decision_prompt(ctx: _ToolExecutionContext) -> Dict[str, Any]:
    """
    Analyzes recent decision reflection logs and asks an LLM to refine the
    original prompt for the 'simulate_decision_reflection' skill based on feedback.

    Args:
        ctx: The skill execution context (provides logger, llm_interface).

    Returns:
        A dictionary containing the refined prompt or an error message.
    """
    ctx.logger.info("Executing refine_decision_prompt skill...")
    n_logs_to_load = 10

    # Get LLM interface from context
    llm_interface = ctx.llm_interface
    if not llm_interface:
        ctx.logger.error("LLMInterface not found in execution context.")
        return {"error": "Internal error: LLMInterface missing."}

    # 1. Load recent logs
    try:
        ctx.logger.info(f"Loading last {n_logs_to_load} decision reflection logs...")
        recent_logs = load_recent_reflection_logs(n=n_logs_to_load)
    except Exception as e:
        ctx.logger.exception("Failed to load decision reflection logs:")
        return {"error": f"Failed to load decision reflection logs: {e}"}

    if not recent_logs:
        ctx.logger.warning("No decision reflection logs found to analyze.")
        return {"error": "Nenhum log de reflexão de decisão encontrado para análise."}

    ctx.logger.info(f"Loaded {len(recent_logs)} reflection logs for analysis.")

    # 2. Construct the LLM prompt for refinement
    log_entries_str = []
    for i, log_entry in enumerate(reversed(recent_logs)): # Most recent first
        user_input = log_entry.get("user_input", "[Entrada não registrada]")
        sim_reflection = log_entry.get("simulated_reflection", "[Reflexão não registrada]")
        llm_feedback = log_entry.get("llm_feedback")
        if not llm_feedback:
            llm_feedback = "[Feedback não registrado ou vazio]"

        log_str = f"""--- LOG #{i + 1} ---
Entrada Original: {user_input}
Reflexão Simulada Gerada:
{sim_reflection}
Feedback Recebido sobre a Reflexão:
{llm_feedback}"""
        log_entries_str.append(log_str)

    # Join log entries *before* the f-string
    joined_logs_str = '\n\n'.join(log_entries_str)

    # Assume we need the original prompt text here.
    # Since we don't have it directly, we'll ask the LLM to infer improvements based on logs.
    # TODO: Ideally, fetch the actual current prompt for 'simulate_decision_reflection'
    #       and provide it in the prompt below for more accurate refinement.
    #       For now, the LLM must work based only on the logs provided.

    prompt = f"""Você é um especialista em engenharia de prompts. Sua tarefa é refinar o prompt de uma skill chamada 'simulate_decision_reflection'.
Abaixo estão exemplos recentes de uso dessa skill, incluindo a entrada do usuário, a reflexão simulada gerada pela skill e o feedback recebido de um LLM avaliador sobre essa reflexão.

Analise os logs fornecidos, prestando atenção especial aos feedbacks recebidos. Identifique temas recorrentes, sugestões de melhoria, pontos de ambiguidade ou partes que podem ser removidas do prompt original (mesmo que você não o veja).

Com base na sua análise, gere uma NOVA versão completa e refinada do prompt para a skill 'simulate_decision_reflection'. O novo prompt deve:
1. Ser mais claro e conciso.
2. Incorporar as sugestões implícitas ou explícitas nos feedbacks.
3. Manter o objetivo original de simular a reflexão de Arthur sobre uma decisão.
4. Instruir o modelo a gerar a reflexão no formato desejado (inferido dos exemplos, se possível).

Logs para Análise:
==================

[INÍCIO DOS LOGS]

{joined_logs_str}

[FINAL DOS LOGS]

Instruções de Saída:
- Responda APENAS com o texto completo do novo prompt refinado.
- Não inclua nenhuma explicação, introdução, cabeçalho, ou qualquer texto antes ou depois do prompt gerado.
- O prompt deve estar pronto para ser usado diretamente na definição da skill 'simulate_decision_reflection'.

Novo Prompt Refinado:"""

    ctx.logger.debug(f"Generated prompt for LLM refinement (length: {len(prompt)}). Sample: {prompt[:300]}...")

    # 3. Call LLM for refinement
    try:
        ctx.logger.info("Calling LLM to refine the decision prompt (streaming)...")
        refined_prompt_response = ""
        async for chunk in llm_interface.call_llm(
            messages=[{"role": "user", "content": prompt}], 
            stream=True
        ):
            refined_prompt_response += chunk

        if not refined_prompt_response or not refined_prompt_response.strip():
            ctx.logger.warning("LLM returned empty response for prompt refinement.")
            return {"error": "LLM refinement resulted in an empty response."}

        # Check for potential error markers (LLMInterface might handle this differently now)
        if refined_prompt_response.startswith("[LLM Error:"):
            ctx.logger.error(f"LLM call failed during prompt refinement: {refined_prompt_response}")
            return {"error": refined_prompt_response}

        ctx.logger.info("Successfully received refined prompt from LLM.")
        ctx.logger.debug(f"Refined Prompt (Raw):\n{refined_prompt_response}")

        # 4. Return the result
        # The LLM was instructed to return *only* the prompt text.
        return {"refined_prompt": refined_prompt_response.strip()}

    except Exception as e:
        ctx.logger.exception("Error during LLM call for prompt refinement:")
        return {"error": f"Failed to get refined prompt from LLM: {e}"}

# Example Usage (Conceptual - requires running within A3X context)
# async def example_run():
#     # Mock or provide a real SkillContext
#     class MockLogger:
#         def info(self, msg, *args): print(f"[INFO] {msg}" % args)
#         def warning(self, msg, *args): print(f"[WARN] {msg}" % args)
#         def error(self, msg, *args, exc_info=False): print(f"[ERROR] {msg}" % args)
#         def debug(self, msg, *args): print(f"[DEBUG] {msg}" % args)
#         def exception(self, msg, *args): print(f"[EXCEPTION] {msg}" % args)
#
#     async def mock_refine_llm_call(prompt: str):
#         print("\n--- Mock LLM Called for Refinement ---")
#         yield "Este é um prompt refinado simulado. "
#         yield "Ele deve ser claro, conciso e incorporar feedback. "
#         yield "Instrua o modelo sobre o formato de saída desejado."
#         print("--- Mock LLM Finished ---\n")
#
#     # Ensure the log directory and file exist for the example
#     # (Code similar to the one in learn_from_reflection_logs.py needed here
#     #  to create a dummy log file if it doesn't exist)
#
#     mock_ctx = SkillContext(
#         logger=MockLogger(),
#         llm_call=mock_refine_llm_call,
#         available_skills={}, memory=None, current_task="Test Refine Prompt", thought_process=[]
#     )
#     result = await refine_decision_prompt(mock_ctx)
#     print("\n--- Refinement Result ---")
#     if "error" in result:
#         print(f"Error: {result['error']}")
#     else:
#         print(f"Refined Prompt:\n{result['refined_prompt']}")
#     print("--- End Refinement Result ---")
#
# if __name__ == '__main__':
#     # Need to ensure logs exist before running
#     # Create dummy logs if necessary
#     log_dir = '../../memory/llm_logs' # Adjust path relative to this file
#     log_file = os.path.join(log_dir, 'decision_reflections.jsonl')
#     if not os.path.exists(log_file):
#          # Add code to create dummy log file here if needed for standalone testing
#          print(f"Warning: Log file {log_file} not found. Create dummy logs for testing.")
#          # ... (implementation to create dummy file) ...
#     # asyncio.run(example_run()) # Uncomment to run example if setup 