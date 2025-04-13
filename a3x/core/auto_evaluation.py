import datetime
import json
from typing import List, Dict, Any

AUTOEVAL_LOG = "memory/learning_logs/auto_evaluation.jsonl"
SELF_EVAL_LOG = "memory/self_evaluation_log.jsonl"

def auto_evaluate_task(objective: str, plan: List[str], execution_results: List[Dict[str, Any]], heuristics_used: List[str], start_time: float, end_time: float):
    """
    Avalia o desempenho do Arthur ao final de uma tarefa.
    Registra métricas quantitativas e qualitativas para loops de melhoria contínua.
    """
    success = any(res.get("status") == "success" for res in execution_results)
    num_steps = len(plan)
    num_failures = sum(1 for res in execution_results if res.get("status") == "error")
    heuristics_applied = heuristics_used or []
    duration = end_time - start_time
    reuse_index = len(set(heuristics_applied)) / (len(heuristics_applied) or 1)
    # Passos corrigidos: passos que falharam e foram repetidos com sucesso
    corrected_steps = [i for i, res in enumerate(execution_results) if res.get("status") == "success" and i > 0 and execution_results[i-1].get("status") == "error"]
    # Erros evitados: heurísticas aplicadas que impediram repetição de erro conhecido
    errors_evited = [h for h in heuristics_applied if "evitar" in h.lower() or "não repetir" in h.lower()]
    # Tempo/etapas poupados: estimativa (pode ser aprimorada)
    steps_saved = max(0, num_steps - len(execution_results))

    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "objective": objective,
        "plan_steps": num_steps,
        "success": success,
        "num_failures": num_failures,
        "heuristics_applied": heuristics_applied,
        "reuse_index_last_10": reuse_index,
        "duration_seconds": duration,
        "corrected_steps": corrected_steps,
        "errors_evited": errors_evited,
        "steps_saved": steps_saved,
        "final_status": execution_results[-1].get("status") if execution_results else "unknown",
        "final_message": execution_results[-1].get("data", {}).get("message", "") if execution_results else "",
        "observations": {
            "insights": "TODO: Adicionar análise qualitativa automática.",
            "difficulties": "TODO: Detectar dificuldades recorrentes.",
        }
    }
    # Salva em ambos os logs
    with open(AUTOEVAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(SELF_EVAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry