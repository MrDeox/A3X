import logging
from typing import List, Dict, Any, Optional

from a3x.core.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

async def simulate_plan_execution(
    plan: List[str],
    llm_interface: LLMInterface,
    context: Optional[Dict[str, Any]] = None,
    heuristics: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Simula a execução de um plano passo-a-passo, sem afetar o mundo real.
    Retorna uma lista de resultados simulados (sucesso/falha, mensagens, heurísticas).
    """
    logger.info("[Simulation] Iniciando simulação de execução de plano...")
    simulation_results = []
    for i, step in enumerate(plan):
        prompt_messages = [
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
            async for chunk in llm_interface.call_llm(
                messages=prompt_messages,
                stream=False
            ):
                response += chunk
            logger.info(f"[Simulation] Resposta do LLM para simulação:\n{response}")
            
            import json
            try:
                result = json.loads(response.strip())
            except json.JSONDecodeError:
                logger.warning("[Simulation] Direct JSON parse failed, attempting markdown extraction.")
                start = response.find("```json")
                if start != -1:
                    start += len("```json")
                    end = response.find("```", start)
                    if end != -1:
                        json_str = response[start:end].strip()
                    else:
                         json_str = response[start:].strip()
                else:
                    start = response.find("```")
                    if start != -1:
                         start += 3
                         end = response.find("```", start)
                         if end != -1:
                              json_str = response[start:end].strip()
                         else:
                              json_str = response[start:].strip()
                    else:
                         json_str = response.strip()
                
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as final_err:
                    logger.error(f"[Simulation] Failed to parse JSON even after extraction: {final_err}. Raw response: {response}")
                    raise final_err

            simulation_results.append(result)
        except Exception as e:
            logger.error(f"[Simulation] Erro ao simular passo '{step}': {e}", exc_info=True)
            simulation_results.append({"status": "error", "message": f"Erro na simulação: {e}", "heuristic": None})
    return simulation_results

async def auto_evaluate_agent(
    benchmark_plans: List[List[str]],
    llm_interface: LLMInterface,
    context: Optional[Dict[str, Any]] = None,
    heuristics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Executa autoavaliação do agente rodando benchmarks simulados e sumarizando resultados.
    """
    logger.info("[Simulation] Iniciando autoavaliação do agente via simulação.")
    all_results = []
    for plan in benchmark_plans:
        sim_results = await simulate_plan_execution(
            plan,
            llm_interface=llm_interface,
            context=context,
            heuristics=heuristics
        )
        all_results.append(sim_results)
    
    num_success = sum(1 for plan_res in all_results for r in plan_res if r.get("status") == "success")
    num_fail = sum(1 for plan_res in all_results for r in plan_res if r.get("status") == "error")
    total_steps = sum(len(plan) for plan in benchmark_plans)
    
    logger.info(f"[Simulation] Auto-evaluation completed. Success: {num_success}, Fail: {num_fail}, Total Steps: {total_steps}")
    
    return {
        "total_plans": len(benchmark_plans),
        "total_steps": total_steps,
        "successes": num_success,
        "failures": num_fail,
        "details": all_results,
    }