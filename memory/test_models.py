"""
Testes para os modelos de memória do A³X.
"""

from datetime import datetime
from memory_models import (
    EpisodicMemoryEntry,
    SemanticMemoryEntry,
    ProceduralMemoryEntry
)

def test_episodic_memory():
    """Testa a criação de uma memória episódica."""
    memory = EpisodicMemoryEntry(
        trigger_input="Usuário pediu para lembrar sua GPU",
        goal="Armazenar informação sobre a GPU do usuário",
        outcome="Informação armazenada com sucesso",
        reflection="O usuário tem uma RX 6400"
    )
    print("\nMemória Episódica:")
    print(memory.model_dump_json(indent=2))

def test_semantic_memory():
    """Testa a criação de uma memória semântica."""
    memory = SemanticMemoryEntry(
        concept_id="gpu_rx6400",
        content="GPU AMD Radeon RX 6400",
        metadata={
            "source": "user_input",
            "confidence": 1.0,
            "related_concepts": ["gpu", "amd", "radeon"]
        }
    )
    print("\nMemória Semântica:")
    print(memory.model_dump_json(indent=2))

def test_procedural_memory():
    """Testa a criação de uma memória procedural."""
    memory = ProceduralMemoryEntry(
        skill_id="store_gpu_info",
        trigger_description="Quando o usuário quer armazenar informações sobre sua GPU",
        steps=[
            {"action": "identify_gpu_info", "description": "Identificar informações da GPU"},
            {"action": "store_info", "description": "Armazenar informações no banco de dados"},
            {"action": "confirm", "description": "Confirmar armazenamento"}
        ],
        usage_count=1,
        success_rate=1.0,
        last_used=datetime.now()
    )
    print("\nMemória Procedural:")
    print(memory.model_dump_json(indent=2))

if __name__ == "__main__":
    test_episodic_memory()
    test_semantic_memory()
    test_procedural_memory() 