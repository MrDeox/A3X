"""
a3x/core/learning_logs.py

Funções utilitárias para registro rastreável e validação de heurísticas e aprendizados do Arthur (A³X).
Pode substituir o logger atual em qualquer ponto de registro de heurística.
"""

import os
import json
import uuid
import datetime

LOG_PATH = "a3x/memory/learning_logs/heuristics_traceable.jsonl"

def log_heuristic_with_traceability(heuristic: dict, plan_id: str, execution_id: str, validation_status: str = "pending"):
    """
    Registra uma heurística com rastreabilidade total: origem, plano, execução, status de validação.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    entry = {
        "heuristic_id": str(uuid.uuid4()),
        "plan_id": plan_id,
        "execution_id": execution_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "origin": heuristic.get("origin", {}),
        "heuristic": heuristic,
        "validation_status": validation_status
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry

# Exemplo de uso:
if __name__ == "__main__":
    h = {"type": "failure_pattern", "trigger": "web_search", "recommendation": "Use proxy", "origin": {"task_id": "t1"}}
    print(log_heuristic_with_traceability(h, "plan-123", "exec-456"))

def load_recent_reflection_logs(limit: int = 10):
    """
    Stub: Retorna os logs de reflexão mais recentes.
    No momento, retorna uma lista vazia ou lê de um arquivo se implementado.
    """
    # TODO: Implementar leitura real dos logs de reflexão se necessário
    return []
