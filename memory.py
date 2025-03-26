#!/usr/bin/env python3
"""
Módulo de Memória do A³X - Armazenamento persistente de contexto.
"""

import sqlite3
import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configuração de logging
logging.basicConfig(
    filename='logs/memory.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constantes
MAX_KEY_LENGTH = 64
MAX_VALUE_SIZE = 10 * 1024  # 10KB em bytes
DB_PATH = Path('data/memory.db')

def _validate_key(key: str) -> bool:
    """
    Valida o formato da chave.
    
    Args:
        key: Chave a ser validada
        
    Returns:
        bool: True se válida, False caso contrário
    """
    if not key or len(key) > MAX_KEY_LENGTH:
        return False
        
    # Apenas letras, números e underscores
    pattern = r'^[a-zA-Z0-9_]+$'
    return bool(re.match(pattern, key))

def _validate_value(value: str) -> bool:
    """
    Valida o formato do valor.
    
    Args:
        value: Valor a ser validado
        
    Returns:
        bool: True se válido, False caso contrário
    """
    if not isinstance(value, str):
        return False
        
    # Verifica tamanho máximo
    return len(value.encode('utf-8')) <= MAX_VALUE_SIZE

def _init_db() -> None:
    """
    Inicializa o banco de dados SQLite.
    """
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Cria tabela se não existir
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logging.info("Banco de dados inicializado com sucesso")
        
    except Exception as e:
        logging.error(f"Erro ao inicializar banco de dados: {e}")
        raise

def store(key: str, value: str, overwrite: bool = False) -> None:
    """
    Armazena um par chave/valor no banco de dados.
    
    Args:
        key: Chave única
        value: Valor a ser armazenado
        overwrite: Se True, permite sobrescrever chave existente
        
    Raises:
        ValueError: Se chave ou valor forem inválidos
        KeyError: Se chave existir e overwrite=False
    """
    # Validações
    if not _validate_key(key):
        raise ValueError("Chave inválida")
    if not _validate_value(value):
        raise ValueError("Valor inválido")
        
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Verifica se chave existe
        cursor.execute("SELECT 1 FROM memory WHERE key = ?", (key,))
        exists = cursor.fetchone() is not None
        
        if exists and not overwrite:
            raise KeyError(f"Chave '{key}' já existe")
            
        # Insere ou atualiza
        if exists:
            cursor.execute(
                "UPDATE memory SET value = ?, created_at = CURRENT_TIMESTAMP WHERE key = ?",
                (value, key)
            )
        else:
            cursor.execute(
                "INSERT INTO memory (key, value) VALUES (?, ?)",
                (key, value)
            )
            
        conn.commit()
        logging.info(f"Valor armazenado para chave: {key}")
        
    except Exception as e:
        logging.error(f"Erro ao armazenar valor: {str(e)}")
        raise
        
    finally:
        conn.close()

def retrieve(key: str) -> Optional[str]:
    """
    Recupera um valor do banco de dados.
    
    Args:
        key: Chave do valor desejado
        
    Returns:
        Optional[str]: Valor armazenado ou None se não encontrado
    """
    if not _validate_key(key):
        raise ValueError("Chave inválida")
        
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("SELECT value FROM memory WHERE key = ?", (key,))
        result = cursor.fetchone()
        
        if result:
            logging.info(f"Valor recuperado para chave: {key}")
            return result[0]
        else:
            logging.warning(f"Chave não encontrada: {key}")
            return None
            
    except Exception as e:
        logging.error(f"Erro ao recuperar valor: {str(e)}")
        raise
        
    finally:
        conn.close()

# Inicializa o banco de dados
_init_db()

# Interface pública
__all__ = ['store', 'retrieve'] 