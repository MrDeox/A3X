import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class FederatedLearningManager:
    """
    Gerenciador de aprendizado federado/colaborativo do A³X.
    Permite compartilhamento, validação cruzada e consolidação de heurísticas, skills e experiências entre múltiplos agentes.
    """

    def __init__(self, peers: Optional[List[str]] = None):
        self.peers = peers or []  # Lista de endereços/IDs de outros agentes
        self.shared_heuristics: List[Dict[str, Any]] = []
        self.shared_skills: List[str] = []
        self.validation_log: List[Dict[str, Any]] = []

    def share_heuristics(self, heuristics: List[Dict[str, Any]]):
        """
        Compartilha heurísticas com os peers (placeholder: simula broadcast).
        """
        for peer in self.peers:
            logger.info(f"[FederatedLearning] Enviando heurísticas para peer {peer} (simulado).")
        self.shared_heuristics.extend(heuristics)

    def receive_heuristics(self, heuristics: List[Dict[str, Any]]):
        """
        Recebe heurísticas de outros agentes.
        """
        logger.info(f"[FederatedLearning] Recebendo {len(heuristics)} heurísticas de peers.")
        self.shared_heuristics.extend(heuristics)

    def validate_and_consolidate(self):
        """
        Valida heurísticas recebidas via consenso/validação cruzada e consolida as robustas.
        """
        # Placeholder: aceita heurísticas que aparecem em pelo menos 2 peers
        from collections import Counter
        all_heuristics = [json.dumps(h, sort_keys=True) for h in self.shared_heuristics]
        freq = Counter(all_heuristics)
        consolidated = [json.loads(h) for h, c in freq.items() if c >= 2]
        self.validation_log.append({"consolidated": consolidated, "total": len(self.shared_heuristics)})
        logger.info(f"[FederatedLearning] Heurísticas consolidadas: {len(consolidated)}")
        return consolidated

    def share_skills(self, skills: List[str]):
        """
        Compartilha nomes/códigos de skills com peers.
        """
        for peer in self.peers:
            logger.info(f"[FederatedLearning] Enviando skills para peer {peer} (simulado).")
        self.shared_skills.extend(skills)

    def receive_skills(self, skills: List[str]):
        """
        Recebe skills de outros agentes.
        """
        logger.info(f"[FederatedLearning] Recebendo {len(skills)} skills de peers.")
        self.shared_skills.extend(skills)