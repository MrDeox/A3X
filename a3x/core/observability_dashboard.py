import logging
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
from a3x.core.config import LEARNING_LOGS_DIR, HEURISTIC_LOG_FILE # Import constants

logger = logging.getLogger(__name__)

class ObservabilityDashboard:
    """
    Dashboard cognitivo para visualização, análise e otimização dos processos mentais do A³X.
    Permite auditoria, colaboração e tomada de decisão baseada em dados.
    """

    def __init__(self, logs_dir: Path = LEARNING_LOGS_DIR): # Use constant for default logs_dir
        self.logs_dir = logs_dir

    def load_heuristics(self) -> List[Dict[str, Any]]:
        """
        Carrega heurísticas do log.
        """
        # Use the imported constant for the specific file
        file_path = HEURISTIC_LOG_FILE
        if not file_path.exists():
            return []
        heuristics = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    heuristics.append(json.loads(line))
                except Exception as e:
                    logger.warning(f"[ObservabilityDashboard] Falha ao carregar heurística: {e}")
        return heuristics

    def summarize_heuristics(self) -> Dict[str, Any]:
        """
        Sumariza heurísticas por tipo, frequência e impacto.
        """
        heuristics = self.load_heuristics()
        from collections import Counter
        types = Counter(h.get("type", "unknown") for h in heuristics)
        return {
            "total": len(heuristics),
            "by_type": dict(types),
        }

    def list_recent_failures(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Lista as últimas falhas registradas.
        """
        heuristics = self.load_heuristics()
        failures = [h for h in heuristics if "fail" in h.get("type", "")]
        return failures[-n:]

    def list_recent_successes(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Lista os últimos sucessos registrados.
        """
        heuristics = self.load_heuristics()
        successes = [h for h in heuristics if "success" in h.get("type", "")]
        return successes[-n:]

    def export_dashboard_json(self, output_path: Path):
        """
        Exporta um snapshot do dashboard em JSON para visualização externa.
        """
        dashboard = {
            "summary": self.summarize_heuristics(),
            "recent_failures": self.list_recent_failures(),
            "recent_successes": self.list_recent_successes(),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dashboard, f, ensure_ascii=False, indent=2)
        logger.info(f"[ObservabilityDashboard] Dashboard exportado para {output_path}")
