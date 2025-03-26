"""
Testes para o sistema de memória do A³X.
"""

import logging
from datetime import datetime
from memory_models import (
    EpisodicMemoryEntry,
    SemanticMemoryEntry,
    ProceduralMemoryEntry
)
from memory_system import MemorySystem

# Configurar logging para os testes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_memory_system():
    """Testa a funcionalidade do sistema de memória."""
    # Criar instância do sistema
    memory_system = MemorySystem(logger)
    
    # Testar memória episódica
    episodic = EpisodicMemoryEntry(
        trigger_input="Usuário pediu para lembrar sua GPU",
        goal="Armazenar informação sobre a GPU do usuário",
        outcome="Informação armazenada com sucesso",
        reflection="O usuário tem uma RX 6400"
    )
    memory_system.add_episodic_memory(episodic)
    
    # Testar memória semântica
    semantic = SemanticMemoryEntry(
        concept_id="gpu_rx6400",
        content="GPU AMD Radeon RX 6400",
        metadata={
            "source": "user_input",
            "confidence": 1.0,
            "related_concepts": ["gpu", "amd", "radeon"]
        }
    )
    memory_system.add_semantic_memory(semantic)
    
    # Testar memória procedural
    procedural = ProceduralMemoryEntry(
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
    memory_system.add_procedural_memory(procedural)

if __name__ == "__main__":
    test_memory_system() 