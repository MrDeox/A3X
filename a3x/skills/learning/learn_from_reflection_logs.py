import logging
import json
import asyncio
import os # Added for example usage path
import datetime # Added for example usage timestamp
from typing import Dict, Any, List, Optional
from pathlib import Path

# Core imports
from a3x.core.skills import skill
from a3x.core.learning_logs import load_recent_reflection_logs
from a3x.core.llm_interface import LLMInterface
from a3x.core.agent import _ToolExecutionContext
from a3x.core.config import LEARNING_LOGS_DIR, HEURISTIC_LOG_FILE, LLM_LOGS_DIR

logger = logging.getLogger(__name__)

# --- Constants ---
REFLECTION_LOG_PATH = Path(LLM_LOGS_DIR) / "decision_reflections.jsonl"
HEURISTIC_LOG_PATH = Path(LEARNING_LOGS_DIR) / HEURISTIC_LOG_FILE

@skill(
    name="learn_from_reflection_logs",
    description="Analisa logs da skill de reflexão para extrair aprendizados. Pode analisar os N mais recentes ou um log específico por offset.",
    parameters={
        "n_logs": (int, 10),  # Analisa os N últimos logs (se offset não for usado).
        "log_offset": (Optional[int], None) # Default to None to prioritize n_logs
    }
)
async def learn_from_reflection_logs(
    ctx: _ToolExecutionContext, 
    n_logs: int = 10, 
    log_offset: Optional[int] = None # Default None
) -> Dict[str, Any]:
    """
    Analyzes reflection logs to extract learning points for system evolution.
    Prioritizes log_offset over n_logs if both are provided.

    Args:
        ctx: The skill execution context (provides logger, llm_interface).
        n_logs: The number of recent reflection logs to analyze (ignored if log_offset is set).
        log_offset: The offset for the specific log to analyze (0 for the latest, 1 for the second latest, etc.).
                   If set, analyzes exactly one log.

    Returns:
        A dictionary containing the synthesized insights from the log(s) or an error.
    """
    ctx.logger.info(f"Executing learn_from_reflection_logs skill. Offset: {log_offset}, N_logs: {n_logs}")

    # Get LLM interface
    llm_interface = ctx.llm_interface
    if not llm_interface:
        ctx.logger.error("LLMInterface not found in execution context.")
        return {"error": "Internal error: LLMInterface missing."}

    logs_to_process: List[Dict[str, Any]] = []
    log_load_description = ""

    # Determine which log(s) to load based on parameters
    if log_offset is not None and log_offset >= 0:
        # Offset is prioritized
        target_offset = log_offset
        ctx.logger.info(f"Loading specific log with offset: {target_offset}")
        log_load_description = f"log at offset {target_offset}"
        try:
            # Load enough logs to potentially reach the offset
            # We load offset + 1 logs, and if successful, take the last one (which is at the desired offset from the end)
            all_logs_for_offset = load_recent_reflection_logs(n=target_offset + 1)
            if len(all_logs_for_offset) > target_offset:
                 # Python list index -1 is last, -2 second last...
                 # So offset 0 needs index -1, offset 1 needs index -2 etc.
                 target_index = -1 - target_offset
                 logs_to_process = [all_logs_for_offset[target_index]]
            else:
                 # Not enough logs were available
                 ctx.logger.error(f"Log offset {target_offset} exceeds the number of available logs ({len(all_logs_for_offset)})." )
                 return {"error": f"Offset {target_offset} excede número de logs disponíveis ({len(all_logs_for_offset)})."}

        except Exception as e:
            ctx.logger.exception(f"Failed to load reflection logs for offset {target_offset}:" )
            return {"error": f"Failed to load reflection logs for offset {target_offset}: {e}"}

    else:
        # Use n_logs (default behavior)
        ctx.logger.info(f"Loading last {n_logs} logs.")
        log_load_description = f"last {n_logs} logs"
        try:
            logs_to_process = load_recent_reflection_logs(n=n_logs)
        except Exception as e:
            ctx.logger.exception("Failed to load reflection logs:")
            return {"error": f"Failed to load reflection logs: {e}"}

    if not logs_to_process:
        ctx.logger.warning(f"No reflection logs found for the specified criteria ({log_load_description}).")
        return {"insights_from_logs": f"Nenhum log de reflexão encontrado para: {log_load_description}."}

    ctx.logger.info(f"Loaded {len(logs_to_process)} reflection log(s) for analysis ({log_load_description}).")

    # 2. Construct the LLM prompt (using logs_to_process)
    log_entries_str = []
    # Iterate showing most recent log first in the prompt (or the single selected log)
    for i, log_entry in enumerate(reversed(logs_to_process)): # Reversed works even for single item list
        user_input = log_entry.get("user_input", "[Entrada não registrada]")
        sim_reflection = log_entry.get("simulated_reflection", "[Reflexão não registrada]")
        # Handle potential None or empty string feedback
        llm_feedback = log_entry.get("llm_feedback", "[Feedback não registrado ou vazio]")

        # Use index relative to the *loaded* logs (will be #1 if offset is used)
        log_str = f"""--- LOG #{i + 1} ---
Entrada: {user_input}

Reflexão Simulada:
{sim_reflection}

Feedback do LLM:
{llm_feedback}"""
        log_entries_str.append(log_str)

    # Join log entries *before* the f-string
    joined_logs_str = '\n\n'.join(log_entries_str)

    ctx.logger.info(f"<<< PREPARANDO PARA CONSTRUIR PROMPT para {log_load_description} >>>")
    prompt = f"""Você é um analista de aprendizagem cognitiva. Abaixo estão registros de decisões simuladas feitas por Arthur. O campo 'Feedback do LLM' pode estar vazio ou não registrado; IGNORE este campo se estiver vazio e foque sua análise APENAS na 'Reflexão Simulada'.

Sua tarefa é analisar APENAS as 'Reflexões Simuladas' para gerar aprendizados que possam ser aplicados para melhorar o desempenho futuro da IA de Arthur.

Para cada Reflexão Simulada analisada, identifique:
1. Padrões de raciocínio recorrentes (heurísticas, estilo de decisão, princípios mencionados).
2. Possíveis refatorações no prompt original da skill 'simulate_decision_reflection' que podem ser inferidas a partir do estilo e conteúdo da reflexão gerada.
3. Ideias para LoRAs conceituais que poderiam ser treinadas com base nos temas das reflexões (ex: "Arthur em decisões de risco", "Arthur em temas existenciais", etc).

Registros de decisões simuladas:
=============================

[INÍCIO DOS LOGS]

{joined_logs_str}

[FINAL DOS LOGS]

Agora, sintetize os aprendizados da análise das REFLEXÕES SIMULADAS acima em formato estruturado.
Responda SOMENTE com o seguinte formato:
- Heurísticas Relevantes:
- Padrões ou Temas nas Reflexões:
- Melhorias de Prompt Sugeridas (baseadas nas reflexões):
- Ideias de LoRAs Temáticas:

Seja direto e objetivo. Evite repetições. Foque em como o sistema pode evoluir com base nas reflexões.
Não adicione nenhuma introdução ou comentário antes ou depois da lista estruturada.
"""

    ctx.logger.debug(f"Constructed prompt for {log_load_description}. Length: {len(prompt)}")

    # Refined prompt sent to LLM for analysis
    analysis_prompt = prompt

    ctx.logger.debug(f"Generated prompt for LLM analysis (length: {len(analysis_prompt)}). Sample: {analysis_prompt[:300]}...")

    # 3. Call LLM for analysis
    try:
        ctx.logger.info(f"Calling LLM for reflection log analysis ({log_load_description})...")
        analysis_response = ""
        async for chunk in llm_interface.call_llm(messages=[{"role": "user", "content": analysis_prompt}], stream=True):
            analysis_response += chunk

        if not analysis_response or not analysis_response.strip():
            # Handle empty response from LLM
            ctx.logger.warning(f"Received empty analysis from LLM for {log_load_description}.")
            # Return insights indicating empty analysis, not necessarily an error
            return {"insights_from_logs": f"Análise do LLM resultou vazia para {log_load_description}."}

        ctx.logger.info(f"Successfully received analysis from LLM for {log_load_description}.")

        ctx.logger.debug(f"Raw Analysis:\n{analysis_response}")

        # 4. Return the result
        return {
            "status": "success",
            "data": {"insights_from_logs": analysis_response.strip()}
        }

    except Exception as e:
        ctx.logger.exception(f"Error during LLM call or processing for {log_load_description}:" )
        return {"error": f"Failed to get analysis from LLM for {log_load_description}: {e}"}

# Example Usage (if run directly, requires async context and logs file)
# Commented out by default
# if __name__ == '__main__':
#     import logging
#     # Need SkillContext for example
#     # Need actual call_llm or a mock

#     logging.basicConfig(level=logging.INFO)

#     # Ensure the log directory and file exist for the example
#     log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'memory', 'llm_logs')
#     log_file = os.path.join(log_dir, 'decision_reflections.jsonl')
#     os.makedirs(log_dir, exist_ok=True)
#     if not os.path.exists(log_file):
#         print(f"Creating dummy log file: {log_file}")
#         with open(log_file, 'w', encoding='utf-8') as f:
#             # Create a simple dummy log entry
#             dummy_log = {
#                 "timestamp": datetime.datetime.now().isoformat(),
#                 "user_input": "Qual o sentido da vida?",
#                 "simulated_reflection": "Reflexão simulada sobre o sentido da vida...",
#                 "llm_feedback": "Feedback do LLM sobre a reflexão..."
#             }
#             f.write(json.dumps(dummy_log) + '\n')

#     # Example mock context (replace llm_call with actual implementation if needed)
#     class MockLogger:
#         def info(self, msg, *args, **kwargs): print(f"[INFO] {msg}" % args)
#         def warning(self, msg, *args, **kwargs): print(f"[WARN] {msg}" % args)
#         def error(self, msg, *args, exc_info=False, **kwargs): print(f"[ERROR] {msg}" % args)
#         def debug(self, msg, *args, **kwargs): print(f"[DEBUG] {msg}" % args)
#         def exception(self, msg, *args, **kwargs): print(f"[EXCEPTION] {msg}" % args)

#     async def mock_analyze_llm_call(prompt: str):
#         print("--- Mock LLM Called for Analysis ---")
#         # Simulate receiving the structured response
#         yield "- Heurísticas Relevantes: Prioriza aprendizado e adaptabilidade."
#         yield "\n- Pontos Repetitivos no Feedback: Reflexão pode ser mais concisa; feedback pede mais exemplos concretos."
#         yield "\n- Melhorias de Prompt Sugeridas: Adicionar instrução para focar em concisão na seção 'Justificativa Final'."
#         yield "\n- Ideias de LoRAs Temáticas: Arthur_Decision_Making_Philosophy, Arthur_Feedback_Integration"
#         print("--- Mock LLM Finished ---")

#     async def run_example():
#         # Assuming llm_call is available or mocked in the actual context
#         mock_ctx = SkillContext(
#             logger=MockLogger(),
#             llm_call=mock_analyze_llm_call,
#             # Add other necessary context attributes if your SkillContext requires them
#             # Example: available_skills={}, memory=None, etc.
#             available_skills={},
#             memory=None,
#             current_task="Testing learn_from_reflection_logs",
#             thought_process=[]
#         )
#         result = await learn_from_reflection_logs(mock_ctx, n_logs=5)
#         print("\n--- Analysis Result ---")
#         if "error" in result:
#             print(f"Error: {result['error']}")
#         else:
#             print(result['insights_from_logs'])
#         print("--- End Analysis Result ---")

#     # Run the async example
#     asyncio.run(run_example()) 