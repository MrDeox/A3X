"""
a3x/core/heuristic_planner_middleware.py

Middleware para integração ativa de heurísticas no planejamento do Arthur (A³X).
Permite injetar recomendações, restrições e otimizações baseadas em heurísticas aprendidas diretamente no plano gerado.

Ponto de integração: Chamar no final de generate_plan, antes de retornar o plano.
"""

from typing import List, Dict, Any

# Estrutura de dados de heurística rastreável
Heuristic = Dict[str, Any]
"""
Exemplo:
{
    "id": "heuristic-uuid",
    "type": "failure_pattern" | "success_pattern" | "optimization",
    "trigger": "web_search_without_proxy",
    "recommendation": "Always configure proxy before web_search",
    "context": {...},
    "impact": {"success_rate_delta": 0.12},
    "origin": {"task_id": "...", "timestamp": "..."}
}
"""

def inject_heuristics_into_plan(plan: List[str], heuristics: List[Heuristic]) -> List[str]:
    """
    Analisa o plano e as heurísticas aprendidas, injetando recomendações, restrições ou passos extras conforme necessário.
    Retorna o plano modificado.
    """
    # Exemplo: se heurística recomenda proxy antes de web_search, insere passo extra
    modified_plan = []
    for step in plan:
        for h in heuristics:
            if h.get("trigger") in step and h.get("recommendation"):
                # Exemplo simples: inserir recomendação antes do passo
                modified_plan.append(f"[HEURISTIC] {h['recommendation']}")
        modified_plan.append(step)
    return modified_plan

# Exemplo de uso:
if __name__ == "__main__":
    plan = [
        "Use the web_search tool to find the current price of Bitcoin.",
        "Use the write_file tool to save the found price into 'btc_price.txt'.",
        "Use the final_answer tool to confirm the price has been saved."
    ]
    heuristics = [
        {
            "id": "h1",
            "type": "failure_pattern",
            "trigger": "web_search",
            "recommendation": "Configure proxy before web_search",
            "context": {},
            "impact": {"success_rate_delta": 0.12},
            "origin": {"task_id": "t1", "timestamp": "2025-04-12T19:00:00Z"}
        }
    ]
    print(inject_heuristics_into_plan(plan, heuristics))