"""
a3x/core/learning_dashboard.py

Mockup de dashboard para visualização do progresso de aprendizado, heatmaps de skills, timeline de eventos e curadoria de heurísticas do Arthur (A³X).
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime

LOG_PATH = "a3x/memory/learning_logs/heuristics_traceable.jsonl"

def load_heuristics():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def render_heatmap_skills(heuristics):
    # Contagem simples: skill x tipo de heurística
    matrix = defaultdict(Counter)
    for h in heuristics:
        skill = h["heuristic"].get("trigger", "unknown")
        htype = h["heuristic"].get("type", "other")
        matrix[skill][htype] += 1
    print("HEATMAP: Skills x Heurística")
    print("Skill".ljust(20), "Failure".ljust(8), "Success".ljust(8), "Optimization".ljust(12))
    for skill, counts in matrix.items():
        print(skill.ljust(20), str(counts.get("failure_pattern",0)).ljust(8), str(counts.get("success_pattern",0)).ljust(8), str(counts.get("optimization",0)).ljust(12))

def render_timeline(heuristics):
    print("\nTIMELINE: Eventos de aprendizado")
    events = sorted(heuristics, key=lambda h: h["timestamp"])
    for h in events:
        t = h["timestamp"]
        skill = h["heuristic"].get("trigger", "unknown")
        v = h.get("validation_status", "pending")
        print(f"{t} | Skill: {skill} | Status: {v}")

def render_pending_heuristics(heuristics):
    print("\nHeurísticas pendentes de validação:")
    for h in heuristics:
        if h.get("validation_status") == "pending":
            print(f"- {h['heuristic'].get('trigger','?')} | {h['heuristic'].get('recommendation','?')}")

def render_dashboard():
    heuristics = load_heuristics()
    render_heatmap_skills(heuristics)
    render_timeline(heuristics)
    render_pending_heuristics(heuristics)

# Exemplo de uso:
if __name__ == "__main__":
    render_dashboard()