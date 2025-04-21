import sqlite3
import asyncio
import json
import logging
from typing import Protocol, Any, Optional, List, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Define o diretório do banco de dados (pode ser configurável)
DB_DIRECTORY = Path(__file__).parent.parent / "data"
DB_DIRECTORY.mkdir(parents=True, exist_ok=True)

class ContextStore(Protocol):
    """Define a interface para armazenamento de contexto persistente."""

    async def initialize(self) -> None:
        """Inicializa a conexão e a estrutura do banco de dados."""
        ...

    async def set(self, key: str, value: Any) -> None:
        """Define/sobrescreve um valor para uma chave."""
        ...

    async def get(self, key: str, default: Any = None) -> Any:
        """Obtém o valor associado a uma chave."""
        ...
        
    async def delete(self, key: str) -> None:
        """Remove uma chave e seu valor."""
        ...

    async def push(self, key: str, item: Any) -> None:
        """Adiciona um item a uma lista associada a uma chave."""
        ...
        
    async def pop_all(self, key: str) -> List[Any]:
        """Obtém todos os itens de uma lista e a limpa."""
        ...

    async def scan(self, prefix: str) -> Dict[str, Any]:
        """Encontra todas as chaves que começam com um prefixo e retorna seus valores."""
        ...

    async def close(self) -> None:
        """Fecha a conexão com o banco de dados."""
        ...


class SQLiteContextStore:
    """Implementação de ContextStore usando SQLite."""

    def __init__(self, db_name: str = "a3net_context.sqlite"):
        self.db_path = DB_DIRECTORY / db_name
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock() # Para proteger o acesso à conexão
        logger.info(f"SQLiteContextStore initialized for db: {self.db_path}")

    async def initialize(self) -> None:
        async with self._lock:
            if self._conn is None:
                try:
                    self._conn = await asyncio.to_thread(
                        sqlite3.connect, self.db_path, check_same_thread=False
                    )
                    # Criar tabela se não existir
                    await asyncio.to_thread(self._create_table)
                    logger.info(f"Database connection established and table verified for {self.db_path}")
                except sqlite3.Error as e:
                    logger.error(f"Failed to initialize database connection for {self.db_path}: {e}", exc_info=True)
                    self._conn = None # Garante que conn é None em caso de falha
                    raise # Re-lança a exceção

    def _create_table(self):
        """Cria a tabela key_value se ela não existir (executado em thread)."""
        if not self._conn: return
        try:
            with self._conn: # Usa a conexão como gerenciador de contexto para commit/rollback
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS key_value_store (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        type TEXT NOT NULL DEFAULT 'json' -- 'json' or 'list'
                    )
                """)
        except sqlite3.Error as e:
            logger.error(f"Failed to create table 'key_value_store': {e}", exc_info=True)
            raise

    def _serialize(self, value: Any) -> str:
        return json.dumps(value)

    def _deserialize(self, value: Optional[str]) -> Any:
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(f"Failed to deserialize value: {value[:100]}... Returning raw string.")
            return value # Retorna a string bruta se não for JSON válido

    async def set(self, key: str, value: Any) -> None:
        if self._conn is None: await self.initialize()
        if self._conn is None: raise ConnectionError("Database connection not established.")
        
        serialized_value = self._serialize(value)
        value_type = 'list' if isinstance(value, list) else 'json'
        
        async with self._lock:
            try:
                await asyncio.to_thread(
                    self._set_sync, key, serialized_value, value_type
                )
                logger.debug(f"[ContextStore] Set key='{key}', type='{value_type}'")
            except sqlite3.Error as e:
                logger.error(f"Failed to set key '{key}': {e}", exc_info=True)
                raise

    def _set_sync(self, key: str, serialized_value: str, value_type: str):
        if not self._conn: return
        with self._conn:
            self._conn.execute("""
                INSERT OR REPLACE INTO key_value_store (key, value, type)
                VALUES (?, ?, ?)
            """, (key, serialized_value, value_type))

    async def get(self, key: str, default: Any = None) -> Any:
        if self._conn is None: await self.initialize()
        if self._conn is None: return default # Retorna default se conexão falhou

        async with self._lock:
            try:
                result = await asyncio.to_thread(self._get_sync, key)
                if result is None:
                    logger.debug(f"[ContextStore] Get key='{key}' -> Not Found (returning default)")
                    return default
                else:
                    value = self._deserialize(result[0])
                    logger.debug(f"[ContextStore] Get key='{key}' -> Found (type: {result[1]})")
                    return value
            except sqlite3.Error as e:
                logger.error(f"Failed to get key '{key}': {e}", exc_info=True)
                return default

    def _get_sync(self, key: str) -> Optional[Tuple[str, str]]:
        if not self._conn: return None
        cursor = self._conn.cursor()
        cursor.execute("SELECT value, type FROM key_value_store WHERE key = ?", (key,))
        return cursor.fetchone()

    async def delete(self, key: str) -> None:
        if self._conn is None: await self.initialize()
        if self._conn is None: raise ConnectionError("Database connection not established.")

        async with self._lock:
            try:
                await asyncio.to_thread(self._delete_sync, key)
                logger.debug(f"[ContextStore] Deleted key='{key}'")
            except sqlite3.Error as e:
                logger.error(f"Failed to delete key '{key}': {e}", exc_info=True)
                raise
                
    def _delete_sync(self, key: str):
         if not self._conn: return
         with self._conn:
             self._conn.execute("DELETE FROM key_value_store WHERE key = ?", (key,))

    async def push(self, key: str, item: Any) -> None:
        """Adiciona um item a uma lista (cria a lista se não existir)."""
        if self._conn is None: await self.initialize()
        if self._conn is None: raise ConnectionError("Database connection not established.")

        async with self._lock:
            try:
                await asyncio.to_thread(self._push_sync, key, item)
                logger.debug(f"[ContextStore] Pushed item to key='{key}'")
            except sqlite3.Error as e:
                logger.error(f"Failed to push to key '{key}': {e}", exc_info=True)
                raise

    def _push_sync(self, key: str, item: Any):
         if not self._conn: return
         with self._conn:
             cursor = self._conn.cursor()
             cursor.execute("SELECT value, type FROM key_value_store WHERE key = ?", (key,))
             result = cursor.fetchone()
             
             current_list = []
             if result:
                 if result[1] == 'list':
                     current_list = self._deserialize(result[0])
                     if not isinstance(current_list, list):
                         logger.warning(f"Data corruption for list key '{key}'. Resetting to empty list.")
                         current_list = []
                 else:
                      logger.warning(f"Trying to push to non-list key '{key}'. Overwriting existing value.")
                      current_list = []
             
             current_list.append(item)
             serialized_list = self._serialize(current_list)
             
             cursor.execute("""
                 INSERT OR REPLACE INTO key_value_store (key, value, type)
                 VALUES (?, ?, 'list')
             """, (key, serialized_list))

    async def pop_all(self, key: str) -> List[Any]:
        """Obtém todos os itens de uma lista e a limpa."""
        if self._conn is None: await self.initialize()
        if self._conn is None: return []

        async with self._lock:
            try:
                items = await asyncio.to_thread(self._pop_all_sync, key)
                logger.debug(f"[ContextStore] Popped {len(items)} items from key='{key}'")
                return items
            except sqlite3.Error as e:
                logger.error(f"Failed to pop_all from key '{key}': {e}", exc_info=True)
                return [] # Retorna lista vazia em caso de erro

    def _pop_all_sync(self, key: str) -> List[Any]:
        if not self._conn: return []
        items = []
        with self._conn: # Transação
             cursor = self._conn.cursor()
             cursor.execute("SELECT value, type FROM key_value_store WHERE key = ?", (key,))
             result = cursor.fetchone()
             
             if result and result[1] == 'list':
                 items = self._deserialize(result[0])
                 if not isinstance(items, list):
                     logger.warning(f"Data corruption for list key '{key}' during pop_all. Returning empty list.")
                     items = []
                 # Remove a chave após ler
                 cursor.execute("DELETE FROM key_value_store WHERE key = ?", (key,))
             elif result:
                 logger.warning(f"Trying to pop_all from non-list key '{key}'. Key not cleared.")
                 # Não remove a chave se não for lista
             # Se a chave não existe, 'items' permanece []
             
        return items

    async def scan(self, prefix: str) -> Dict[str, Any]:
        """Encontra todas as chaves que começam com um prefixo e retorna seus valores."""
        if self._conn is None: await self.initialize()
        if self._conn is None: return {}

        async with self._lock:
            try:
                results = await asyncio.to_thread(self._scan_sync, prefix)
                # Deserialize values
                deserialized_results = {
                    key: self._deserialize(value) for key, value in results
                }
                logger.debug(f"[ContextStore] Scanned prefix='{prefix}', found {len(deserialized_results)} items.")
                return deserialized_results
            except sqlite3.Error as e:
                logger.error(f"Failed to scan prefix '{prefix}': {e}", exc_info=True)
                return {}

    def _scan_sync(self, prefix: str) -> List[Tuple[str, str]]:
        """Executa a consulta SCAN na thread."""
        if not self._conn: return []
        cursor = self._conn.cursor()
        # Use the LIKE operator with a wildcard
        # Ensure the prefix is escaped properly if needed, though LIKE doesn't treat
        # many characters specially by default unless ESCAPE is used.
        query_prefix = f"{prefix}%"
        cursor.execute("SELECT key, value FROM key_value_store WHERE key LIKE ?", (query_prefix,))
        return cursor.fetchall()

    async def close(self) -> None:
        async with self._lock:
            if self._conn:
                try:
                    await asyncio.to_thread(self._conn.close)
                    self._conn = None
                    logger.info(f"Database connection closed for {self.db_path}")
                except sqlite3.Error as e:
                    logger.error(f"Failed to close database connection {self.db_path}: {e}", exc_info=True)