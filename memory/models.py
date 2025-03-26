"""
Modelos de dados para o sistema de memória do A³X.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class MemoryEntry(BaseModel):
    """Classe base para entradas de memória."""
    key: str = Field(..., description="Chave única para identificar a entrada")
    value: str = Field(..., description="Valor armazenado")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp da criação")
    last_accessed: Optional[datetime] = Field(None, description="Timestamp do último acesso")

class EpisodicMemoryEntry(MemoryEntry):
    """
    Entrada de memória episódica.
    Armazena eventos e experiências específicas com contexto temporal.
    """
    context: dict = Field(default_factory=dict, description="Contexto do episódio")
    duration: Optional[float] = Field(None, description="Duração do episódio em segundos")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "ep_001",
                "timestamp": "2024-01-01T12:00:00",
                "event": "Primeiro encontro com o usuário",
                "context": {"location": "terminal", "user": "arthur"},
                "emotions": ["curiosity", "excitement"],
                "importance": 0.8
            }
        }

class SemanticMemoryEntry(MemoryEntry):
    """
    Entrada de memória semântica.
    Armazena fatos, conceitos e relações entre informações.
    """
    relations: List[str] = Field(default_factory=list, description="Lista de relações com outros conceitos")
    confidence: float = Field(default=1.0, description="Nível de confiança na informação")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "sem_001",
                "concept": "Python",
                "description": "Linguagem de programação interpretada de alto nível",
                "relations": ["programming", "language", "development"],
                "confidence": 0.9,
                "last_accessed": "2024-01-01T12:00:00"
            }
        }

class ProceduralMemoryEntry(MemoryEntry):
    """
    Entrada de memória procedural.
    Armazena habilidades, rotinas e procedimentos.
    """
    steps: List[str] = Field(default_factory=list, description="Lista de passos do procedimento")
    success_rate: float = Field(default=1.0, description="Taxa de sucesso do procedimento")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "proc_001",
                "skill": "executar_comando_terminal",
                "steps": [
                    "validar_comando",
                    "verificar_permissões",
                    "executar_comando",
                    "capturar_saida"
                ],
                "success_rate": 0.95,
                "last_practiced": "2024-01-01T12:00:00",
                "parameters": {
                    "command": "string",
                    "timeout": "float"
                }
            }
        } 