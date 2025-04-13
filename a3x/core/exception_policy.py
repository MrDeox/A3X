from typing import Type, Dict, Any, Callable, Optional
import logging

class ExceptionPolicy:
    """
    Define uma política configurável para propagação ou tratamento de exceções.
    Pode ser usada para controlar, em tempo de execução, se exceções devem ser propagadas ou apenas logadas.
    """

    def __init__(self, policy: Optional[Dict[Type[BaseException], str]] = None, default: str = "log"):
        """
        policy: dict {ExceptionType: "raise"|"log"|"ignore"}
        default: ação padrão se exceção não estiver no dicionário
        """
        self.policy = policy or {}
        self.default = default
        self.logger = logging.getLogger(__name__)

    def handle(self, exc: BaseException, context: str = ""):
        action = self.policy.get(type(exc), self.default)
        if action == "raise":
            raise exc
        elif action == "log":
            self.logger.error(f"[ExceptionPolicy] {context}: {exc}")
        elif action == "ignore":
            pass
        else:
            self.logger.warning(f"[ExceptionPolicy] Ação desconhecida '{action}' para {type(exc)}. Logando por padrão.")
            self.logger.error(f"[ExceptionPolicy] {context}: {exc}")

    def set_policy(self, exc_type: Type[BaseException], action: str):
        self.policy[exc_type] = action

    def set_default(self, action: str):
        self.default = action