import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class MetaLearningManager:
    """
    Gerenciador de meta-aprendizado e autoajuste cognitivo do A³X.
    Permite o agente ajustar seus próprios parâmetros de aprendizado, replanejamento, simulação e monetização.
    """

    def __init__(self):
        self.meta_params: Dict[str, Any] = {
            "finetune_trigger_threshold": 50,
            "simulation_batch_size": 10,
            "replan_error_repeats": 2,
            "monetization_eval_threshold": 7,
            "exploration_rate": 0.1,
        }
        self.history: List[Dict[str, Any]] = []

    def update_param(self, param: str, value: Any):
        """
        Atualiza um parâmetro de meta-aprendizado.
        """
        logger.info(f"[MetaLearning] Atualizando parâmetro '{param}' para {value}")
        self.meta_params[param] = value

    def auto_adjust(self, metrics: Dict[str, Any]):
        """
        Ajusta automaticamente parâmetros com base em métricas de desempenho.
        """
        logger.info(f"[MetaLearning] Autoajustando parâmetros com base em métricas: {metrics}")
        # Exemplo: se taxa de falha aumentar, diminui threshold de replanejamento
        if metrics.get("fail_rate", 0) > 0.3:
            self.meta_params["replan_error_repeats"] = max(1, self.meta_params["replan_error_repeats"] - 1)
        # Se sucesso em monetização for baixo, aumenta exploração
        if metrics.get("monetization_success", 0) < 0.5:
            self.meta_params["exploration_rate"] = min(1.0, self.meta_params["exploration_rate"] + 0.05)
        # Outros ajustes podem ser implementados conforme necessário

    def log_meta_decision(self, decision: str, context: Optional[Dict[str, Any]] = None):
        """
        Registra decisões de meta-aprendizado para rastreabilidade.
        """
        entry = {"decision": decision, "context": context or {}}
        self.history.append(entry)
        logger.info(f"[MetaLearning] Decisão registrada: {decision} | Contexto: {context}")