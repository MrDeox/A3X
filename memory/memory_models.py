"""
Modelos de dados para o sistema de memória do A³X.
Define as estruturas de dados para memória episódica, semântica e procedural.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, confloat

class EpisodicMemoryEntry(BaseModel):
    """
    Representa uma memória episódica - um evento ou experiência específica.
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    trigger_input: Optional[str] = None
    goal: Optional[str] = None
    plan: Optional[List[Dict[str, Any]]] = None
    action_executed: Optional[Dict[str, Any]] = None
    outcome: str
    reflection: Optional[str] = None
    embedding: Optional[List[float]] = None

class SemanticMemoryEntry(BaseModel):
    """
    Representa uma memória semântica - conhecimento factual ou conceitual.
    """
    concept_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    last_accessed: datetime = Field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None

class ProceduralMemoryEntry(BaseModel):
    """
    Representa uma memória procedural - habilidade ou sequência de ações aprendida.
    """
    skill_id: str
    trigger_description: str
    steps: List[Dict[str, Any]]
    usage_count: int = 0
    success_rate: confloat(ge=0.0, le=1.0) = 0.5
    last_used: Optional[datetime] = None
    embedding: Optional[List[float]] = None 