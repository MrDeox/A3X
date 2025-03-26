"""
Testes do sistema de memória do A³X.
"""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from memory.models import (
    EpisodicMemoryEntry,
    SemanticMemoryEntry,
    ProceduralMemoryEntry
)
from memory.system import MemorySystem

@pytest.fixture
def memory_system():
    """Cria um sistema de memória temporário para testes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        system = MemorySystem(db_path)
        yield system

def test_episodic_memory(memory_system):
    """Testa operações com memória episódica."""
    # Cria uma entrada
    entry = EpisodicMemoryEntry(
        id="ep_001",
        event="Teste de memória episódica",
        context={"test": True},
        emotions=["curiosity"],
        importance=0.8
    )
    
    # Adiciona a entrada
    memory_system.add_episodic_memory(entry)
    
    # Recupera a entrada
    retrieved = memory_system.get_episodic_memory("ep_001")
    assert retrieved is not None
    assert retrieved.id == entry.id
    assert retrieved.event == entry.event
    assert retrieved.context == entry.context
    assert retrieved.emotions == entry.emotions
    assert retrieved.importance == entry.importance
    
    # Testa busca
    results = memory_system.search_episodic_memory("teste")
    assert len(results) > 0
    assert results[0].id == entry.id

def test_semantic_memory(memory_system):
    """Testa operações com memória semântica."""
    # Cria uma entrada
    entry = SemanticMemoryEntry(
        id="sem_001",
        concept="Teste",
        description="Conceito de teste",
        relations=["test", "memory"],
        confidence=0.9
    )
    
    # Adiciona a entrada
    memory_system.add_semantic_memory(entry)
    
    # Recupera a entrada
    retrieved = memory_system.get_semantic_memory("sem_001")
    assert retrieved is not None
    assert retrieved.id == entry.id
    assert retrieved.concept == entry.concept
    assert retrieved.description == entry.description
    assert retrieved.relations == entry.relations
    assert retrieved.confidence == entry.confidence
    
    # Testa busca
    results = memory_system.search_semantic_memory("teste")
    assert len(results) > 0
    assert results[0].id == entry.id

def test_procedural_memory(memory_system):
    """Testa operações com memória procedural."""
    # Cria uma entrada
    entry = ProceduralMemoryEntry(
        id="proc_001",
        skill="test_skill",
        steps=["step1", "step2"],
        success_rate=0.95,
        parameters={"param1": "value1"}
    )
    
    # Adiciona a entrada
    memory_system.add_procedural_memory(entry)
    
    # Recupera a entrada
    retrieved = memory_system.get_procedural_memory("proc_001")
    assert retrieved is not None
    assert retrieved.id == entry.id
    assert retrieved.skill == entry.skill
    assert retrieved.steps == entry.steps
    assert retrieved.success_rate == entry.success_rate
    assert retrieved.parameters == entry.parameters
    
    # Testa busca
    results = memory_system.search_procedural_memory("test")
    assert len(results) > 0
    assert results[0].id == entry.id

def test_memory_not_found(memory_system):
    """Testa recuperação de memórias inexistentes."""
    assert memory_system.get_episodic_memory("nonexistent") is None
    assert memory_system.get_semantic_memory("nonexistent") is None
    assert memory_system.get_procedural_memory("nonexistent") is None

def test_memory_update(memory_system):
    """Testa atualização de memórias existentes."""
    # Cria uma entrada
    entry = EpisodicMemoryEntry(
        id="ep_001",
        event="Evento original",
        context={"test": True},
        emotions=["curiosity"],
        importance=0.8
    )
    
    # Adiciona a entrada
    memory_system.add_episodic_memory(entry)
    
    # Atualiza a entrada
    entry.event = "Evento atualizado"
    entry.importance = 0.9
    memory_system.add_episodic_memory(entry)
    
    # Verifica a atualização
    retrieved = memory_system.get_episodic_memory("ep_001")
    assert retrieved is not None
    assert retrieved.event == "Evento atualizado"
    assert retrieved.importance == 0.9 