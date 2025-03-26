"""
Sistema de memória do A³X.
Responsável pelo armazenamento e recuperação de informações.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from .models import MemoryEntry, SemanticMemoryEntry, EpisodicMemoryEntry, ProceduralMemoryEntry

# Configurar logger
logger = logging.getLogger(__name__)

class MemorySystem:
    """
    Sistema de memória do A³X.
    Gerencia diferentes tipos de memória: episódica, semântica e procedural.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Inicializa o sistema de memória.
        
        Args:
            db_path: Caminho para o arquivo do banco de dados SQLite.
                    Se None, usa o arquivo padrão memory.db no diretório atual.
        """
        self.db_path = db_path or str(Path.cwd() / 'memory.db')
        self._init_db()
        logger.info(f"Sistema de memória inicializado com banco em {self.db_path}")
    
    def _init_db(self):
        """Inicializa o banco de dados com as tabelas necessárias."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabela para memória semântica
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    last_accessed DATETIME,
                    relations TEXT,
                    confidence REAL DEFAULT 1.0
                )
            """)
            
            # Tabela para memória episódica
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    last_accessed DATETIME,
                    context TEXT,
                    duration REAL
                )
            """)
            
            # Tabela para memória procedural
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS procedural_memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    last_accessed DATETIME,
                    steps TEXT,
                    success_rate REAL DEFAULT 1.0
                )
            """)
            
            conn.commit()
    
    def store(self, key: str, entry: Union[MemoryEntry, str]) -> None:
        """
        Armazena uma entrada na memória.
        
        Args:
            key: Chave para identificar a entrada
            entry: Entrada de memória ou string para armazenar
        """
        if isinstance(entry, str):
            entry = SemanticMemoryEntry(
                key=key,
                value=entry,
                timestamp=datetime.now()
            )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if isinstance(entry, SemanticMemoryEntry):
                cursor.execute("""
                    INSERT OR REPLACE INTO semantic_memory
                    (key, value, timestamp, last_accessed, relations, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    entry.key,
                    entry.value,
                    entry.timestamp,
                    entry.last_accessed,
                    ','.join(entry.relations),
                    entry.confidence
                ))
            
            elif isinstance(entry, EpisodicMemoryEntry):
                cursor.execute("""
                    INSERT OR REPLACE INTO episodic_memory
                    (key, value, timestamp, last_accessed, context, duration)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    entry.key,
                    entry.value,
                    entry.timestamp,
                    entry.last_accessed,
                    str(entry.context),
                    entry.duration
                ))
            
            elif isinstance(entry, ProceduralMemoryEntry):
                cursor.execute("""
                    INSERT OR REPLACE INTO procedural_memory
                    (key, value, timestamp, last_accessed, steps, success_rate)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    entry.key,
                    entry.value,
                    entry.timestamp,
                    entry.last_accessed,
                    ','.join(entry.steps),
                    entry.success_rate
                ))
            
            conn.commit()
            logger.info(f"Entrada armazenada com chave {key}")
    
    def retrieve(self, key: str) -> Optional[str]:
        """
        Recupera uma entrada da memória.
        
        Args:
            key: Chave da entrada a ser recuperada
            
        Returns:
            Optional[str]: Valor armazenado ou None se não encontrado
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tenta recuperar da memória semântica
            cursor.execute("""
                SELECT value, last_accessed FROM semantic_memory WHERE key = ?
            """, (key,))
            result = cursor.fetchone()
            
            if result:
                value, last_accessed = result
                # Atualiza último acesso
                cursor.execute("""
                    UPDATE semantic_memory
                    SET last_accessed = ?
                    WHERE key = ?
                """, (datetime.now(), key))
                conn.commit()
                return value
            
            # Tenta recuperar da memória episódica
            cursor.execute("""
                SELECT value, last_accessed FROM episodic_memory WHERE key = ?
            """, (key,))
            result = cursor.fetchone()
            
            if result:
                value, last_accessed = result
                # Atualiza último acesso
                cursor.execute("""
                    UPDATE episodic_memory
                    SET last_accessed = ?
                    WHERE key = ?
                """, (datetime.now(), key))
                conn.commit()
                return value
            
            # Tenta recuperar da memória procedural
            cursor.execute("""
                SELECT value, last_accessed FROM procedural_memory WHERE key = ?
            """, (key,))
            result = cursor.fetchone()
            
            if result:
                value, last_accessed = result
                # Atualiza último acesso
                cursor.execute("""
                    UPDATE procedural_memory
                    SET last_accessed = ?
                    WHERE key = ?
                """, (datetime.now(), key))
                conn.commit()
                return value
            
            logger.warning(f"Entrada não encontrada para chave {key}")
            return None
            
    def get_episodic_memory(self, memory_id: str) -> Optional[EpisodicMemoryEntry]:
        """
        Recupera uma entrada da memória episódica.
        
        Args:
            memory_id: ID da entrada
            
        Returns:
            Optional[EpisodicMemoryEntry]: Entrada encontrada ou None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT key, value, timestamp, last_accessed, context, duration
                FROM episodic_memory
                WHERE key = ?
            ''', (memory_id,))
            row = cursor.fetchone()
            
            if row:
                return EpisodicMemoryEntry(
                    key=row[0],
                    value=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    last_accessed=datetime.fromisoformat(row[3]),
                    context=row[4],
                    duration=row[5]
                )
            return None
            
    def get_semantic_memory(self, memory_id: str) -> Optional[SemanticMemoryEntry]:
        """
        Recupera uma entrada da memória semântica.
        
        Args:
            memory_id: ID da entrada
            
        Returns:
            Optional[SemanticMemoryEntry]: Entrada encontrada ou None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT key, value, timestamp, last_accessed, relations, confidence
                FROM semantic_memory
                WHERE key = ?
            ''', (memory_id,))
            row = cursor.fetchone()
            
            if row:
                return SemanticMemoryEntry(
                    key=row[0],
                    value=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    last_accessed=datetime.fromisoformat(row[3]),
                    relations=row[4].split(',') if row[4] else [],
                    confidence=row[5]
                )
            return None
            
    def get_procedural_memory(self, memory_id: str) -> Optional[ProceduralMemoryEntry]:
        """
        Recupera uma entrada da memória procedural.
        
        Args:
            memory_id: ID da entrada
            
        Returns:
            Optional[ProceduralMemoryEntry]: Entrada encontrada ou None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT key, value, timestamp, last_accessed, steps, success_rate
                FROM procedural_memory
                WHERE key = ?
            ''', (memory_id,))
            row = cursor.fetchone()
            
            if row:
                return ProceduralMemoryEntry(
                    key=row[0],
                    value=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    last_accessed=datetime.fromisoformat(row[3]),
                    steps=row[4].split(',') if row[4] else [],
                    success_rate=row[5]
                )
            return None
            
    def search_episodic_memory(self, query: str) -> List[EpisodicMemoryEntry]:
        """
        Busca entradas na memória episódica.
        
        Args:
            query: Texto para busca
            
        Returns:
            List[EpisodicMemoryEntry]: Lista de entradas encontradas
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT key, value, timestamp, last_accessed, context, duration
                FROM episodic_memory
                WHERE value LIKE ? OR context LIKE ?
            ''', (f'%{query}%', f'%{query}%'))
            rows = cursor.fetchall()
            
            return [
                EpisodicMemoryEntry(
                    key=row[0],
                    value=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    last_accessed=datetime.fromisoformat(row[3]),
                    context=row[4],
                    duration=row[5]
                )
                for row in rows
            ]
            
    def search_semantic_memory(self, query: str) -> List[SemanticMemoryEntry]:
        """
        Busca entradas na memória semântica.
        
        Args:
            query: Texto para busca
            
        Returns:
            List[SemanticMemoryEntry]: Lista de entradas encontradas
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT key, value, timestamp, last_accessed, relations, confidence
                FROM semantic_memory
                WHERE value LIKE ? OR relations LIKE ?
            ''', (f'%{query}%', f'%{query}%'))
            rows = cursor.fetchall()
            
            return [
                SemanticMemoryEntry(
                    key=row[0],
                    value=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    last_accessed=datetime.fromisoformat(row[3]),
                    relations=row[4].split(',') if row[4] else [],
                    confidence=row[5]
                )
                for row in rows
            ]
            
    def search_procedural_memory(self, query: str) -> List[ProceduralMemoryEntry]:
        """
        Busca entradas na memória procedural.
        
        Args:
            query: Texto para busca
            
        Returns:
            List[ProceduralMemoryEntry]: Lista de entradas encontradas
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT key, value, timestamp, last_accessed, steps, success_rate
                FROM procedural_memory
                WHERE value LIKE ? OR steps LIKE ?
            ''', (f'%{query}%', f'%{query}%'))
            rows = cursor.fetchall()
            
            return [
                ProceduralMemoryEntry(
                    key=row[0],
                    value=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    last_accessed=datetime.fromisoformat(row[3]),
                    steps=row[4].split(',') if row[4] else [],
                    success_rate=row[5]
                )
                for row in rows
            ] 