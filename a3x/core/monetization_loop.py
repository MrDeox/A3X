import logging
from typing import List, Dict, Any, Optional
from a3x.core.llm_interface import call_llm
from a3x.core.learning_logs import log_heuristic_with_traceability
from a3x.core.simulation import simulate_plan_execution

logger = logging.getLogger(__name__)

class MonetizationLoop:
    """
    Ciclo de monetização autônoma para o A³X: busca, avaliação, execução e aprendizado de oportunidades de geração de valor.
    Garante autonomia absoluta, flexibilidade ilimitada e inteligência máxima via LLM, simulação e heurísticas.
    """

    def __init__(self, llm_url: Optional[str] = None):
        self.llm_url = llm_url or ""
        self.history: List[Dict[str, Any]] = []
        self.heuristics: List[Dict[str, Any]] = []

    async def discover_opportunities(self, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Usa o LLM para buscar e propor oportunidades de geração de valor (tarefas, automações, bots, microserviços, etc).
        """
        prompt = [
            {
                "role": "system",
                "content": (
                    "Você é um agente autônomo de geração de valor. Liste oportunidades concretas e viáveis para gerar valor econômico, "
                    "considerando contexto, recursos disponíveis e tendências atuais. Seja criativo, prático e priorize impacto."
                ),
            },
            {
                "role": "user",
                "content": f"Contexto: {context}\n",
            },
        ]
        response = ""
        try:
            async for chunk in call_llm(prompt, llm_url=self.llm_url, stream=False):
                response += chunk
            logger.info(f"[MonetizationLoop] Oportunidades sugeridas:\n{response}")
            # Extrai oportunidades (um por linha, formato livre)
            opportunities = []
            for line in response.splitlines():
                line = line.strip("-* \n")
                if line:
                    opportunities.append({"description": line})
            return opportunities
        except Exception as e:
            logger.error(f"[MonetizationLoop] Erro ao buscar oportunidades: {e}")
            return []

    async def evaluate_opportunity(self, opportunity: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Usa o LLM e simulação para avaliar potencial de retorno, risco e viabilidade de uma oportunidade.
        """
        prompt = [
            {
                "role": "system",
                "content": (
                    "Você é um avaliador autônomo de oportunidades econômicas. Analise a oportunidade abaixo e atribua notas de potencial de retorno, risco e viabilidade (0-10), justificando cada uma. "
                    "Sugira estratégias para maximizar o sucesso."
                ),
            },
            {
                "role": "user",
                "content": f"Oportunidade: {opportunity}\nContexto: {context}\n",
            },
        ]
        response = ""
        try:
            async for chunk in call_llm(prompt, llm_url=self.llm_url, stream=False):
                response += chunk
            logger.info(f"[MonetizationLoop] Avaliação da oportunidade:\n{response}")
            # Extrai notas e sugestões do texto (pode ser aprimorado)
            import re
            retorno = int(re.search(r"retorno.*?(\d+)", response, re.IGNORECASE).group(1)) if re.search(r"retorno.*?(\d+)", response, re.IGNORECASE) else 0
            risco = int(re.search(r"risco.*?(\d+)", response, re.IGNORECASE).group(1)) if re.search(r"risco.*?(\d+)", response, re.IGNORECASE) else 0
            viabilidade = int(re.search(r"viabilidade.*?(\d+)", response, re.IGNORECASE).group(1)) if re.search(r"viabilidade.*?(\d+)", response, re.IGNORECASE) else 0
            sugestoes = re.findall(r"Sugest[aã]o.*?:\s*(.*)", response, re.IGNORECASE)
            return {
                "retorno": retorno,
                "risco": risco,
                "viabilidade": viabilidade,
                "sugestoes": sugestoes,
                "raw": response,
            }
        except Exception as e:
            logger.error(f"[MonetizationLoop] Erro ao avaliar oportunidade: {e}")
            return {"retorno": 0, "risco": 0, "viabilidade": 0, "sugestoes": [], "raw": ""}

    async def execute_opportunity(self, opportunity: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Gera um plano de execução para a oportunidade, simula e executa os passos, registra resultados e heurísticas.
        """
        # Gera plano via LLM
        prompt = [
            {
                "role": "system",
                "content": (
                    "Você é um executor autônomo. Gere um plano ReAct detalhado para executar a oportunidade abaixo, maximizando retorno e minimizando risco. "
                    "Responda apenas com uma lista de passos ReAct, um por linha."
                ),
            },
            {
                "role": "user",
                "content": f"Oportunidade: {opportunity}\nContexto: {context}\n",
            },
        ]
        plan = []
        try:
            response = ""
            async for chunk in call_llm(prompt, llm_url=self.llm_url, stream=False):
                response += chunk
            logger.info(f"[MonetizationLoop] Plano gerado:\n{response}")
            for line in response.splitlines():
                line = line.strip()
                if line and not line.lower().startswith("thought"):
                    plan.append(line)
        except Exception as e:
            logger.error(f"[MonetizationLoop] Erro ao gerar plano: {e}")
            return {"status": "error", "message": f"Erro ao gerar plano: {e}"}

        # Simula execução do plano
        sim_results = await simulate_plan_execution(plan, context, heuristics=None, llm_url=self.llm_url)
        logger.info(f"[MonetizationLoop] Resultados da simulação: {sim_results}")

        # (Opcional) Executa de fato os passos aprovados (pode ser integrado ao ciclo principal)
        # Aqui, apenas registra heurísticas e resultados
        try:
            plan_id = f"plan-monetization-{opportunity.get('description', '')[:20]}"
            execution_id = f"exec-monetization-{opportunity.get('description', '')[:20]}"
            heuristic = {
                "type": "monetization_attempt",
                "opportunity": opportunity,
                "plan": plan,
                "simulation_results": sim_results,
            }
            log_heuristic_with_traceability(heuristic, plan_id, execution_id, validation_status="pending_manual")
        except Exception as log_err:
            logger.warning(f"[MonetizationLoop] Falha ao registrar heurística de monetização: {log_err}")

        return {
            "status": "success",
            "plan": plan,
            "simulation_results": sim_results,
        }

    async def run(self, context: Optional[Dict[str, Any]] = None, max_opportunities: int = 3):
        """
        Loop principal: busca, avalia, executa e aprende com oportunidades de valor.
        """
        logger.info("[MonetizationLoop] Iniciando ciclo de monetização autônoma.")
        opportunities = await self.discover_opportunities(context)
        for opp in opportunities[:max_opportunities]:
            eval_result = await self.evaluate_opportunity(opp, context)
            logger.info(f"[MonetizationLoop] Avaliação: {eval_result}")
            if eval_result["retorno"] >= 7 and eval_result["viabilidade"] >= 7 and eval_result["risco"] <= 4:
                exec_result = await self.execute_opportunity(opp, context)
                logger.info(f"[MonetizationLoop] Execução: {exec_result}")
            else:
                logger.info(f"[MonetizationLoop] Oportunidade descartada por avaliação insuficiente: {opp}")