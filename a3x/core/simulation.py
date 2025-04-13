import logging
from typing import List, Dict, Any, Optional

from a3x.core.llm_interface import call_llm

logger = logging.getLogger(__name__)

async def simulate_plan_execution(
    plan: List[str],
    context: Optional[Dict[str, Any]] = None,
    heuristics: Optional[List[Dict[str, Any]]] = None,
    llm_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Simula a execução de um plano passo-a-passo, sem afetar o mundo real.
    Retorna uma lista de resultados simulados (sucesso/falha, mensagens, heurísticas).
    """
    logger.info("[Simulation] Iniciando simulação de execução de plano...")
    simulation_results = []
    for i, step in enumerate(plan):
        prompt = [
            {
                "role": "system",
                "content": (
                    "Você é um simulador cognitivo. Dado um passo de plano, contexto e heurísticas, "
                    "simule o resultado provável da execução (sucesso, falha, mensagem de erro, aprendizado esperado). "
                    "Responda em JSON: {\"status\": \"success|error\", \"message\": \"...\", \"heuristic\": \"...\"}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Passo do plano: {step}\n"
                    f"Contexto: {context}\n"
                    f"Heurísticas relevantes: {heuristics}\n"
                ),
            },
        ]
        try:
            logger.info(f"[Simulation] Simulando passo {i+1}/{len(plan)}: {step}")
            response = ""
            async for chunk in call_llm(prompt, llm_url=llm_url, stream=False):
                response += chunk
            logger.info(f"[Simulation] Resposta do LLM para simulação:\n{response}")
            # Tenta extrair JSON da resposta
            import json
            result = json.loads(response.strip().split("```")[-1] if "```" in response else response)
            simulation_results.append(result)
        except Exception as e:
            logger.error(f"[Simulation] Erro ao simular passo '{step}': {e}")
            simulation_results.append({"status": "error", "message": f"Erro na simulação: {e}", "heuristic": None})
    return simulation_results

async def auto_evaluate_agent(
    benchmark_plans: List[List[str]],
    context: Optional[Dict[str, Any]] = None,
    heuristics: Optional[List[Dict[str, Any]]] = None,
    llm_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executa autoavaliação do agente rodando benchmarks simulados e sumarizando resultados.
    """
    logger.info("[Simulation] Iniciando autoavaliação do agente via simulação.")
    all_results = []
    for plan in benchmark_plans:
        sim_results = await simulate_plan_execution(plan, context, heuristics, llm_url)
        all_results.append(sim_results)
    # Sumariza resultados
    num_success = sum(1 for plan in all_results for r in plan if r.get("status") == "success")
    num_fail = sum(1 for plan in all_results for r in plan if r.get("status") == "error")
    return {
        "total_plans": len(benchmark_plans),
        "total_steps": sum(len(plan) for plan in benchmark_plans),
        "successes": num_success,
        "failures": num_fail,
        "details": all_results,
    }