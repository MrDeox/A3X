import logging
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import re # For simple keyword matching

from a3x.core.skills import skill
from a3x.core.context import Context

logger = logging.getLogger(__name__)

# Import constants from config
from a3x.core.config import HEURISTIC_LOG_FILE, HEURISTIC_LOG_CONSOLIDATED_FILE

MAX_HEURISTICS_TO_CONSIDER = 100 # Limit how many recent consolidated logs to check
MIN_KEYWORD_MATCH_SCORE = 1 # Minimum overlap to consider a heuristic relevant

# <<< RESTORED Decorator >>>
@skill(
    name="consult_learned_heuristics",
    description="Consulta o log CONSOLIDADO de heurísticas aprendidas (representativas/únicas) para encontrar regras relevantes.",
    parameters={
        "objective": {"type": str, "description": "The objective to find relevant heuristics for."},
        "top_k": {"type": int, "description": "The maximum number of heuristics of each type (success/failure) to return.", "default": 3}
    }
)
# <<< RESTORED async def and ctx parameter >>>
async def consult_learned_heuristics(ctx: Context, objective: str, top_k: int = 3) -> Dict[str, Any]:
    """Busca heurísticas relevantes (sucesso e falha) no log consolidado."""

    log_prefix = "[ConsultHeuristics Skill]"
    logger.info(f"{log_prefix} Consultando heurísticas CONSOLIDADAS para objetivo '{objective[:50]}...'")

    success_heuristics = []
    failure_heuristics = []

    try:
        # Use the imported constants
        log_file_path = HEURISTIC_LOG_CONSOLIDATED_FILE

        if not log_file_path.exists():
            logger.warning(f"{log_prefix} Log de heurísticas CONSOLIDADAS não encontrado em {log_file_path}. Verificando log original como fallback...")
            # Fallback to original log if consolidated doesn't exist yet
            log_file_path = HEURISTIC_LOG_FILE
            if not log_file_path.exists():
                 logger.warning(f"{log_prefix} Log original também não encontrado em {log_file_path}. Nenhuma heurística será retornada.")
                 return {"status": "success", "data": {"success": [], "failure": [], "message": "Nenhum log de aprendizado encontrado."}}
            else:
                 logger.info(f"{log_prefix} Usando log original {ORIGINAL_HEURISTIC_LOG_FILE} como fallback.")

        # --- Keyword Matching --- 
        objective_keywords = set(re.findall(r'\b\w+\b', objective.lower()))
        if not objective_keywords:
             logger.warning(f"{log_prefix} Não foi possível extrair palavras-chave do objetivo: '{objective}'.")
             return {"status": "success", "data": {"success": [], "failure": [], "message": "Não foi possível processar o objetivo para busca."}}

        logger.debug(f"{log_prefix} Palavras-chave do objetivo: {objective_keywords}")

        matched_entries = [] # List of tuples: (score, type, heuristic_text)
        lines_read = 0

        with open(log_file_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            lines_to_process = all_lines[-MAX_HEURISTICS_TO_CONSIDER:]

            for line in reversed(lines_to_process):
                lines_read += 1
                try:
                    log_entry = json.loads(line.strip())
                    
                    # <<< ADDED: Filter based on consolidation status >>>
                    entry_status = log_entry.get("status")
                    # Process if status is representative, unique, or if status field doesn't exist (fallback for old format)
                    if entry_status is None or entry_status in ["representative", "unique"]:
                        heuristic_type = log_entry.get("type", "unknown") # Get type
                        heuristic_text = log_entry.get("heuristic", "")
                        context_snapshot = log_entry.get("context_snapshot", {})
                        objective_summary = context_snapshot.get("objective_summary", "").lower()
                        
                        if not heuristic_text: continue # Skip entries without heuristic text

                        # Combine text sources for keyword matching
                        text_to_check = f"{heuristic_text.lower()} {objective_summary}"
                        entry_keywords = set(re.findall(r'\b\w+\b', text_to_check))
                        score = len(objective_keywords.intersection(entry_keywords))

                        if score >= MIN_KEYWORD_MATCH_SCORE:
                            # Store the original type (success/failure), not the consolidation status
                            original_type = heuristic_type if heuristic_type in ["success", "failure"] else "unknown"
                            matched_entries.append((score, original_type, heuristic_text))
                            logger.debug(f"{log_prefix} Heurística '{original_type}' (Status: {entry_status or 'N/A'}) correspondente (score {score}): {heuristic_text[:60]}...")
                    # else: # Log skipped redundant/other status if needed for debugging
                    #    logger.debug(f"{log_prefix} Skipping heuristic with status '{entry_status}'.")

                except json.JSONDecodeError:
                    logger.warning(f"{log_prefix} Linha inválida (JSON) encontrada no log: {line.strip()}")
                except Exception as parse_err:
                     logger.warning(f"{log_prefix} Erro ao processar linha do log '{line.strip()}': {parse_err}")

        # Sort all matches by score
        matched_entries.sort(key=lambda x: x[0], reverse=True)

        # Populate separate lists based on original type, up to top_k each
        count_success = 0
        count_failure = 0
        for score, type, text in matched_entries:
            if type == "success" and count_success < top_k:
                success_heuristics.append(text)
                count_success += 1
            elif type == "failure" and count_failure < top_k:
                failure_heuristics.append(text)
                count_failure += 1
            if count_success >= top_k and count_failure >= top_k:
                break

        logger.info(f"{log_prefix} Consulta ao log CONSOLIDADO concluída. Encontradas {len(success_heuristics)} heurísticas de sucesso e {len(failure_heuristics)} de falha (de {len(matched_entries)} correspondências em {lines_read} logs recentes)." )

        return {
            "status": "success",
            "data": {
                "success": success_heuristics,
                "failure": failure_heuristics
            }
        }

    except Exception as e:
        logger.exception(f"{log_prefix} Erro ao consultar log de heurísticas CONSOLIDADAS:")
        return {
            "status": "error",
            "data": {"message": f"Erro ao consultar log de heurísticas consolidadas: {e}"}
        }

# Example usage (for testing - might need adjustments)
if __name__ == '__main__':
    import asyncio
    logging.basicConfig(level=logging.DEBUG)
    # Create dummy log file for testing
    if not os.path.exists(LEARNING_LOG_DIR):
        os.makedirs(LEARNING_LOG_DIR)
    from a3x.core.learning_logs import log_heuristic_with_traceability
    import uuid
    # Limpa o arquivo antes de inserir exemplos
    if os.path.exists(HEURISTIC_LOG_FILE):
        os.remove(HEURISTIC_LOG_FILE)
    # Exemplo de heurísticas seed
    seeds = [
        {"timestamp": "", "type": "failure", "heuristic": "Ao usar 'edit_file', sempre forneça contexto claro.", "context_snapshot": {"objective_summary": "editar arquivo de configuração"}},
        {"timestamp": "", "type": "success", "heuristic": "Para listar diretórios, 'list_dir' é direto.", "context_snapshot": {"objective_summary": "listar arquivos python"}},
        {"timestamp": "", "type": "success", "heuristic": "Usar 'run_terminal_cmd' com 'mkdir -p' cria diretórios aninhados.", "context_snapshot": {"objective_summary": "criar estrutura de pastas"}},
        {"timestamp": "", "type": "failure", "heuristic": "Não assuma que o arquivo existe antes de ler.", "context_snapshot": {"objective_summary": "ler conteúdo de arquivo"}},
    ]
    for i, h in enumerate(seeds):
        plan_id = f"seed-plan-{i+1}"
        execution_id = f"seed-exec-{i+1}"
        log_heuristic_with_traceability(h, plan_id, execution_id, validation_status="seed")

    async def run_test():
        test_objective = "Preciso editar o arquivo principal do projeto."
        result = await consult_learned_heuristics(ctx=None, objective=test_objective, top_k=2)
        print("\n--- Consultation Result ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(run_test())
