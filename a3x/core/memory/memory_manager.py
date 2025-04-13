import logging
from typing import List, Dict, Any, Optional

class MemoryManager:
    """
    Abstrai o acesso à memória semântica (FAISS) e episódica (SQLite).
    Permite futura substituição de backends sem alterar o código do agente.
    """

    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        # Importações locais para facilitar substituição futura
        from a3x.core.embeddings import get_embedding
        from a3x.core.semantic_memory_backend import search_index
        from a3x.core.db_utils import (
            add_episodic_record,
            retrieve_recent_episodes,
        )
        self.get_embedding = get_embedding
        self.search_index = search_index
        self.add_episodic_record = add_episodic_record
        self.retrieve_recent_episodes = retrieve_recent_episodes

        self.semantic_index_path = config.get("SEMANTIC_INDEX_PATH")
        self.semantic_top_k = config.get("SEMANTIC_SEARCH_TOP_K", 5)
        self.episodic_limit = config.get("EPISODIC_RETRIEVAL_LIMIT", 5)

    # --- SEMANTIC MEMORY ---

    def search_semantic_memory(self, query: str) -> List[Dict[str, Any]]:
        """
        Busca na memória semântica (FAISS) usando o embedding do texto.
        """
        self.logger.info(f"Buscando memória semântica para: '{query[:50]}...'")
        embedding = self.get_embedding(query)
        if embedding is None:
            self.logger.error("Falha ao gerar embedding para busca semântica.")
            return []
        try:
            results = self.search_index(
                index_path_base=self.semantic_index_path,
                query_embedding=embedding,
                top_k=self.semantic_top_k,
            )
            return results or []
        except Exception as e:
            self.logger.exception("Erro ao buscar na memória semântica:")
            return []

    # --- EPISODIC MEMORY ---

    def get_recent_episodes(self) -> List[Dict[str, Any]]:
        """
        Recupera os episódios mais recentes da memória episódica.
        """
        try:
            episodes = self.retrieve_recent_episodes(limit=self.episodic_limit)
            return [dict(row) for row in episodes] if episodes else []
        except Exception as e:
            self.logger.exception("Erro ao recuperar episódios recentes:")
            return []

    def record_episodic_event(self, context: str, action: str, outcome: str, metadata: Optional[dict] = None):
        """
        Registra um evento na memória episódica.
        """
        try:
            self.add_episodic_record(context, action, outcome, metadata)
        except Exception as e:
            self.logger.exception("Erro ao registrar evento episódico:")

    # --- EXTENSÃO FUTURA: Métodos para outros tipos de memória/backends podem ser adicionados aqui ---