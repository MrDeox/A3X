from a3x.core.skills import skill
from a3x.core.simulation import simulate_plan_execution

@skill(
    name="simulate_plan",
    description="Simula a execução de um plano passo-a-passo via LLM, sem afetar o mundo real. Retorna resultados simulados.",
    parameters={
        "plan": (list, ...),
        "context": (dict, None),
        "heuristics": (list, None),
        "llm_url": (str, None),
    },
)
async def simulate_plan_skill(
    plan,
    context=None,
    heuristics=None,
    llm_url=None,
):
    results = await simulate_plan_execution(plan, context, heuristics, llm_url)
    return {"status": "success", "data": {"simulation_results": results}}