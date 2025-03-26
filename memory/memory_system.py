"""
Sistema de memória do A³X.
Gerencia diferentes tipos de memória (episódica, semântica e procedural).
"""

import logging
from typing import Optional
from .memory_models import (
    EpisodicMemoryEntry,
    SemanticMemoryEntry,
    ProceduralMemoryEntry
)

class MemorySystem:
    """
    Sistema de memória do A³X que gerencia diferentes tipos de memória.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Inicializa o sistema de memória.
        
        Args:
            logger: Logger opcional para registrar operações
        """
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Inicializando sistema de memória")
        
    def add_episodic_memory(self, entry: EpisodicMemoryEntry) -> None:
        """
        Adiciona uma memória episódica ao sistema.
        
        Args:
            entry: Entrada de memória episódica a ser adicionada
        """
        self.logger.info(f"Adicionando memória episódica: {entry.timestamp}")
        print(f"Adicionando memória episódica:")
        print(entry.model_dump_json(indent=2))
        # TODO: Implementar lógica de armazenamento (FAISS + SQLite)
        
    def add_semantic_memory(self, entry: SemanticMemoryEntry) -> None:
        """
        Adiciona uma memória semântica ao sistema.
        
        Args:
            entry: Entrada de memória semântica a ser adicionada
        """
        self.logger.info(f"Adicionando memória semântica: {entry.concept_id}")
        print(f"Adicionando memória semântica:")
        print(entry.model_dump_json(indent=2))
        # TODO: Implementar lógica de armazenamento (FAISS + SQLite)
        
    def add_procedural_memory(self, entry: ProceduralMemoryEntry) -> None:
        """
        Adiciona uma memória procedural ao sistema.
        
        Args:
            entry: Entrada de memória procedural a ser adicionada
        """
        self.logger.info(f"Adicionando memória procedural: {entry.skill_id}")
        print(f"Adicionando memória procedural:")
        print(entry.model_dump_json(indent=2))
        # TODO: Implementar lógica de armazenamento (FAISS + SQLite) 