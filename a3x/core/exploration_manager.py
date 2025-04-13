import logging
import random
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ExplorationManager:
    """
    Gerenciador de exploração ativa e curiosidade artificial do A³X.
    Permite ao agente buscar ativamente situações novas, desafios e gaps de conhecimento.
    """

    def __init__(self, exploration_rate: float = 0.1):
        self.exploration_rate = exploration_rate
        self.exploration_log: List[Dict[str, Any]] = []

    def should_explore(self) -> bool:
        """
        Decide se o agente deve explorar (em vez de apenas explorar/explorar).
        """
        return random.random() < self.exploration_rate

    def propose_exploration(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Propõe uma ação exploratória: buscar novo domínio, testar skill desconhecida, simular cenário inédito, etc.
        """
        # Exemplos de ações exploratórias
        actions = [
            {"type": "test_new_skill", "skill": "auto_generated"},
            {"type": "simulate_unseen_scenario", "scenario": "edge_case"},
            {"type": "search_web", "query": "novas tendências em IA"},
            {"type": "analyze_gap", "gap": "skill_ausente"},
            {"type": "benchmark", "plan": ["Use the study skill to learn a new domain", "Test application in real task"]},
        ]
        action = random.choice(actions)
        entry = {"exploration": action, "context": context or {}}
        self.exploration_log.append(entry)
        logger.info(f"[ExplorationManager] Ação exploratória proposta: {action}")
        return action

    def log_exploration_result(self, action: Dict[str, Any], result: Any):
        """
        Registra o resultado de uma ação exploratória.
        """
        entry = {"action": action, "result": result}
        self.exploration_log.append(entry)
        logger.info(f"[ExplorationManager] Resultado da exploração registrado: {result}")