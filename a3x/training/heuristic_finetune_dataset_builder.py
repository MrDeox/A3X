"""
a3x/training/heuristic_finetune_dataset_builder.py

Módulo para estruturar heurísticas em datasets prontos para fine-tuning incremental do LLM ou módulos de decisão.
Inclui função para adicionar heurísticas ao dataset e trigger para acionar fine-tuning.

Ponto de integração: Chamar ao registrar heurística relevante em learning_cycle.py ou watchers.
"""

import os
import json

DATASET_PATH = "a3x/training/heuristic_finetune_dataset.jsonl"

from a3x.core.learning_logs import log_heuristic_with_traceability

def append_heuristic_to_finetune_dataset(heuristic: dict, plan_id: str = None, execution_id: str = None, validation_status: str = "pending"):
    """
    Adiciona uma heurística ao dataset de fine-tuning incremental e registra com rastreabilidade total.
    """
    # Gera plan_id e execution_id se não fornecidos
    import uuid
    if plan_id is None:
        plan_id = f"finetune-{uuid.uuid4()}"
    if execution_id is None:
        execution_id = f"finetune-{uuid.uuid4()}"

    # Registra no log centralizado
    log_heuristic_with_traceability(heuristic, plan_id, execution_id, validation_status=validation_status)

    # (Opcional) Mantém compatibilidade: também adiciona ao dataset puro, se necessário para pipeline legado
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    with open(DATASET_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(heuristic, ensure_ascii=False) + "\n")

def should_trigger_finetune(threshold: int = 50) -> bool:
    """
    Retorna True se o número de heurísticas no dataset atingir o threshold.
    """
    if not os.path.exists(DATASET_PATH):
        return False
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return len(lines) >= threshold

def launch_finetune_job():
    """
    Mock: dispara o processo de fine-tuning (substitua por integração real).
    """
    print("[Fine-tune Triggered] Iniciando fine-tuning com heurísticas acumuladas...")

# Exemplo de uso:
if __name__ == "__main__":
    h = {"input": "web_search falhou sem proxy", "output": "adicionar proxy_config antes"}
    append_heuristic_to_finetune_dataset(h)
    if should_trigger_finetune(1):
        launch_finetune_job()