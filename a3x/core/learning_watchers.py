"""
a3x/core/learning_watchers.py

Módulo de triggers e watchers para automação de aprendizado, autogeração de skills e disparo de fine-tuning no Arthur (A³X).
"""

import logging
import time
import json
import os
import asyncio
from typing import Dict, Any, Optional
from collections import deque
from a3x.skills.auto_generated.skill_autogenerator import propose_skill_from_gap

logger = logging.getLogger(__name__)

LOG_PATH = "a3x/memory/learning_logs/heuristics_traceable.jsonl"

def load_heuristics():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def count_redundant_heuristics(trigger: str) -> int:
    heuristics = load_heuristics()
    return sum(1 for h in heuristics if h["heuristic"].get("trigger") == trigger)

def count_pending_skills() -> int:
    # Conta skills auto-geradas pendentes de validação
    auto_dir = "a3x/skills/auto_generated/"
    if not os.path.exists(auto_dir):
        return 0
    return len([f for f in os.listdir(auto_dir) if f.endswith(".py") and f.startswith("auto_")])

def notify_human_curator():
    print("[NOTIFY] Existem skills pendentes de validação. Curador humano deve revisar.")

def fine_tune_needed() -> bool:
    # This function is not provided in the original file or the new file
    # It's assumed to exist as it's called in the watch_heuristics_and_trigger_events function
    pass

def watch_heuristics_and_trigger_events():
    # Exemplo: se houver muitas heurísticas redundantes, sugerir skill robusta
    if count_redundant_heuristics("web_search") > 5:
        context = {"reason": "Muitas falhas em web_search"}
        propose_skill_from_gap("robust_web_search", context)
    if count_pending_skills() > 3:
        notify_human_curator()
    if fine_tune_needed():
        # This function is not provided in the original file or the new file
        # It's assumed to exist as it's called in the watch_heuristics_and_trigger_events function
        pass

# Exemplo de uso:
if __name__ == "__main__":
    watch_heuristics_and_trigger_events()