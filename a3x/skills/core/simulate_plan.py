from a3x.core.skills import skill
from a3x.core.simulation import simulate_plan_execution
from typing import List, Dict, Optional

@skill(
    name="simulate_plan",
    description="Simula a execução de um plano passo-a-passo via LLM, sem afetar o mundo real. Retorna resultados simulados.",
    parameters={
        "plan": {"type": List[Dict], "description": "Lista de passos do plano a ser simulado."},
        "context": {"type": Optional[Dict], "description": "Contexto inicial para a simulação."},
        "heuristics": {"type": Optional[List[str]], "description": "Lista de heurísticas a serem consideradas na simulação."},
        "llm_url": {"type": Optional[str], "description": "URL do LLM a ser usado (sobrescreve a configuração padrão)."},
    },
)
async def simulate_plan_skill(
    plan: List[Dict],
    context: Optional[Dict] = None,
    heuristics: Optional[List[str]] = None,
    llm_url: Optional[str] = None,
):
    results = await simulate_plan_execution(plan, context, heuristics, llm_url)
    return {"status": "success", "data": {"simulation_results": results}}