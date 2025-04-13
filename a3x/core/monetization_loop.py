import logging
from typing import List, Dict, Any, Optional
from a3x.core.llm_interface import LLMInterface
from a3x.core.learning_logs import log_heuristic_with_traceability
from a3x.core.simulation import simulate_plan_execution

logger = logging.getLogger(__name__)

class MonetizationLoop:
    """
    Ciclo de monetização autônoma para o A³X: busca, avaliação, execução e aprendizado de oportunidades de geração de valor.
    Garante autonomia absoluta, flexibilidade ilimitada e inteligência máxima via LLM, simulação e heurísticas.
    """

    def __init__(self, llm_interface: LLMInterface):
        if not llm_interface:
             raise ValueError("LLMInterface instance is required for MonetizationLoop")
        self.llm_interface = llm_interface
        self.history: List[Dict[str, Any]] = []
        self.heuristics: List[Dict[str, Any]] = []

    async def discover_opportunities(self, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Usa o LLM para buscar e propor oportunidades de geração de valor (tarefas, automações, bots, microserviços, etc).
        """
        prompt_messages = [
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
            async for chunk in self.llm_interface.call_llm(messages=prompt_messages, stream=False):
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
            logger.error(f"[MonetizationLoop] Erro ao buscar oportunidades: {e}", exc_info=True)
            return []

    async def evaluate_opportunity(self, opportunity: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Usa o LLM e simulação para avaliar potencial de retorno, risco e viabilidade de uma oportunidade.
        """
        prompt_messages = [
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
            async for chunk in self.llm_interface.call_llm(messages=prompt_messages, stream=False):
                response += chunk
            logger.info(f"[MonetizationLoop] Avaliação da oportunidade:\n{response}")
            # Extrai notas e sugestões do texto (pode ser aprimorado)
            import re
            retorno_match = re.search(r"retorno.*?(\d+)", response, re.IGNORECASE)
            risco_match = re.search(r"risco.*?(\d+)", response, re.IGNORECASE)
            viabilidade_match = re.search(r"viabilidade.*?(\d+)", response, re.IGNORECASE)
            
            retorno = int(retorno_match.group(1)) if retorno_match else 0
            risco = int(risco_match.group(1)) if risco_match else 0
            viabilidade = int(viabilidade_match.group(1)) if viabilidade_match else 0
            
            sugestoes = re.findall(r"Sugest[aã]o.*?:\s*(.*)", response, re.IGNORECASE)
            return {
                "retorno": retorno,
                "risco": risco,
                "viabilidade": viabilidade,
                "sugestoes": sugestoes,
                "raw": response,
            }
        except Exception as e:
            logger.error(f"[MonetizationLoop] Erro ao avaliar oportunidade: {e}", exc_info=True)
            return {"retorno": 0, "risco": 0, "viabilidade": 0, "sugestoes": [], "raw": ""}

    async def execute_opportunity(self, opportunity: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Gera um plano de execução para a oportunidade, simula e executa os passos, registra resultados e heurísticas.
        """
        # Gera plano via LLM
        prompt_messages = [
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
            async for chunk in self.llm_interface.call_llm(messages=prompt_messages, stream=False):
                response += chunk
            logger.info(f"[MonetizationLoop] Plano gerado:\n{response}")
            for line in response.splitlines():
                line = line.strip()
                if line and not line.lower().startswith("thought"):
                    plan.append(line)
            if not plan:
                 logger.warning("[MonetizationLoop] LLM did not generate a plan.")
                 # Decide how to handle no plan: error or empty plan?
                 # return {"status": "error", "message": "Failed to generate execution plan"}
        except Exception as e:
            logger.error(f"[MonetizationLoop] Erro ao gerar plano: {e}", exc_info=True)
            return {"status": "error", "message": f"Erro ao gerar plano: {e}"}

        # Simula execução do plano
        sim_results = await simulate_plan_execution(
             plan,
             llm_interface=self.llm_interface,
             context=context,
             heuristics=None
        )
        logger.info(f"[MonetizationLoop] Resultados da simulação: {sim_results}")

        # (Opcional) Executa de fato os passos aprovados (pode ser integrado ao ciclo principal)
        # Aqui, apenas registra heurísticas e resultados
        try:
            # Sanitize description for IDs
            desc_snippet = "".join(c if c.isalnum() else '_' for c in opportunity.get('description', '')[:20])
            plan_id = f"plan-monetization-{desc_snippet}"
            execution_id = f"exec-monetization-{desc_snippet}"
            heuristic_data = {
                "type": "monetization_attempt",
                "opportunity": opportunity,
                "plan": plan,
                "simulation_results": sim_results,
            }
            log_heuristic_with_traceability(heuristic_data, plan_id, execution_id, validation_status="pending_manual")
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
            if eval_result["retorno"] >= 5 and eval_result["viabilidade"] >= 5 and eval_result["risco"] <= 6:
                exec_result = await self.execute_opportunity(opp, context)
                logger.info(f"[MonetizationLoop] Execução simulada: {exec_result.get('status')}")
            else:
                logger.info(f"[MonetizationLoop] Oportunidade descartada por avaliação insuficiente: {opp['description']}")