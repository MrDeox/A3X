from typing import List, Dict, Any
import logging

class PlanningStrategy:
    """
    Interface para estratégias de planejamento.
    """
    def plan(self, perception: Dict[str, Any], context: Dict[str, Any], agent_logger: logging.Logger) -> List[str]:
        raise NotImplementedError

class SimpleListFilesStrategy(PlanningStrategy):
    """
    Estratégia para tarefas simples de listagem de arquivos.
    """
    def plan(self, perception: Dict[str, Any], context: Dict[str, Any], agent_logger: logging.Logger) -> List[str]:
        objective = perception.get("processed", "")
        agent_logger.info("[SimpleListFilesStrategy] Gerando plano simples para listagem de arquivos.")
        return [
            f"Use the list_directory tool for the objective: '{objective}'",
            "Use the final_answer tool to provide the list of files.",
        ]

class LLMPlannerStrategy(PlanningStrategy):
    """
    Estratégia para planejamento complexo usando LLM.
    """
    def __init__(self, generate_plan_func):
        self.generate_plan_func = generate_plan_func

    async def plan(self, perception: Dict[str, Any], context: Dict[str, Any], agent_logger: logging.Logger, tool_desc, heuristics_context=None) -> List[str]:
        objective = perception.get("processed", "")
        agent_logger.info("[LLMPlannerStrategy] Gerando plano via LLM para objetivo complexo.")
        plan = await self.generate_plan_func(
            objective=objective,
            tool_descriptions=tool_desc,
            agent_logger=agent_logger,
            heuristics_context=heuristics_context,
            context=context,
        )
        return plan