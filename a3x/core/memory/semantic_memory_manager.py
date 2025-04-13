import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from a3x.core.config import SEMANTIC_INDEX_PATH, HEURISTIC_LOG_FILE # Import constants

logger = logging.getLogger(__name__)

class SemanticMemoryManager:
    """
    Gerenciador de memória semântica e episódica para o A³X.
    Permite indexação, busca, analogia e generalização de experiências.
    """

    def __init__(self, index_path: Path = Path(SEMANTIC_INDEX_PATH)): # Use constant for default index path
        self.index_path = index_path
        self.memory_log = HEURISTIC_LOG_FILE # Use constant for memory log path
        # Placeholder para FAISS ou outro backend de embeddings
        self.embeddings = []
        self.metadata = []

    def load_memory(self):
        """
        Carrega experiências e heurísticas do log para memória semântica.
        """
        if not self.memory_log.exists():
            logger.warning(f"[SemanticMemory] Log {self.memory_log} não encontrado.")
            return
        self.embeddings = []
        self.metadata = []
        with open(self.memory_log, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Suporte a diferentes tipos de experiência
                    if "heuristic" in entry or "success" in entry.get("type", "") or "failure" in entry.get("type", ""):
                        # Placeholder: gere embedding do texto da heurística/experiência
                        text = entry.get("heuristic") or entry.get("message") or str(entry)
                        embedding = self._embed_text(text)
                        self.embeddings.append(embedding)
                        self.metadata.append(entry)
                except Exception as e:
                    logger.warning(f"[SemanticMemory] Falha ao carregar linha: {e}")

    def _embed_text(self, text: str) -> List[float]:
        """
        Gera embedding do texto (placeholder, substitua por chamada real ao modelo de embeddings).
        """
        # Exemplo: retorna vetor dummy baseado no hash do texto
        return [float(ord(c)) % 1.0 for c in text[:32]]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Busca experiências/heurísticas semanticamente similares ao query.
        """
        query_emb = self._embed_text(query)
        results = []
        for emb, meta in zip(self.embeddings, self.metadata):
            # Similaridade simples (dot product, pode ser substituído por FAISS real)
            sim = sum(a * b for a, b in zip(query_emb, emb)) / (len(query_emb) or 1)
            results.append((sim, meta))
        results.sort(reverse=True, key=lambda x: x[0])
        return [meta for sim, meta in results[:top_k]]

    def find_analogies(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Busca experiências análogas ao query para transfer learning.
        """
        # Pode ser aprimorado com clustering, analogia vetorial, etc.
        return self.search(query, top_k=top_k)

    def generalize(self) -> List[str]:
        """
        Gera generalizações e regras a partir das experiências armazenadas.
        """
        # Placeholder: sumariza heurísticas mais frequentes
        from collections import Counter
        heuristics = [meta.get("heuristic", "") for meta in self.metadata if meta.get("heuristic")]
        freq = Counter(heuristics)
        generalizations = [f"Regra frequente: {h} (ocorrências: {c})" for h, c in freq.most_common(5)]
        logger.info(f"[SemanticMemory] Generalizações extraídas: {generalizations}")
        return generalizations
