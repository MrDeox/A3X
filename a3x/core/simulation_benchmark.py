import logging
from typing import List, Dict, Any, Optional
from a3x.core.simulation import simulate_plan_execution

logger = logging.getLogger(__name__)

class SimulationBenchmark:
    """
    Infraestrutura para simulação massiva e benchmarking interno do A³X.
    Permite rodar milhares de simulações e benchmarks em paralelo para testar hipóteses, estratégias e novas skills.
    """

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    async def run_benchmarks(self, plans: List[List[str]], context: Optional[Dict[str, Any]] = None, llm_url: Optional[str] = None):
        """
        Executa simulações em lote para múltiplos planos.
        """
        logger.info(f"[SimulationBenchmark] Rodando {len(plans)} benchmarks de simulação...")
        self.results = []
        for i, plan in enumerate(plans):
            sim_result = await simulate_plan_execution(plan, context=context, llm_url=llm_url)
            self.results.append({"plan": plan, "simulation": sim_result})
            logger.info(f"[SimulationBenchmark] Benchmark {i+1}/{len(plans)} concluído.")
        return self.results

    def summarize_results(self) -> Dict[str, Any]:
        """
        Sumariza resultados dos benchmarks: taxa de sucesso, falha, heurísticas aprendidas, etc.
        """
        total = len(self.results)
        success = sum(1 for r in self.results for s in r["simulation"] if s.get("status") == "success")
        fail = sum(1 for r in self.results for s in r["simulation"] if s.get("status") == "error")
        return {
            "total_benchmarks": total,
            "total_success": success,
            "total_fail": fail,
            "details": self.results,
        }