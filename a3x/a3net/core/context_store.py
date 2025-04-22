import sqlite3
import asyncio
import json
import logging
from typing import Protocol, Any, Optional, List, Dict, Tuple, runtime_checkable
from pathlib import Path
import aiosqlite

logger = logging.getLogger(__name__)

# Define o diretório do banco de dados (pode ser configurável)
DB_DIRECTORY = Path(__file__).parent.parent / "data"
DB_DIRECTORY.mkdir(parents=True, exist_ok=True)

@runtime_checkable
class ContextStore(Protocol):
    """Interface defining required methods for a context storage mechanism."""
    async def initialize(self): ...
    async def set(self, key: str, value: Any, tags: Optional[List[str]] = None): ...
    async def get(self, key: str) -> Optional[Any]: ...
    async def delete(self, key: str): ...
    async def find_keys_by_tag(self, tag: str, limit: Optional[int] = None) -> List[str]: ...
    async def get_all_keys(self) -> List[str]: ...
    async def close(self): ...

class SQLiteContextStore:
    """SQLite implementation of the ContextStore interface."""
    def __init__(self, db_name: str = "a3net_context.sqlite"):
        self.db_path = DB_DIRECTORY / db_name
        self.conn = None
        self._lock = asyncio.Lock() # Lock for async operations
        logger.info(f"SQLiteContextStore initialized for db: {self.db_path}")

    async def initialize(self) -> None:
        async with self._lock:
            try:
                # Use detect_types to handle basic types like lists/dicts via JSON
                self.conn = await aiosqlite.connect(self.db_path)
                await self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS store (
                        key TEXT PRIMARY KEY,
                        value TEXT, -- Store complex types as JSON strings
                        tags TEXT    -- Store tags as a JSON list string
                    )
                """)
                # Optional: Create index on tags if querying by tags becomes frequent
                # await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tags ON store (tags);")
                await self.conn.commit()
                logger.info(f"Database connection established and table verified for {self.db_path}")
            except Exception as e:
                logger.error(f"Failed to initialize SQLite database at {self.db_path}: {e}", exc_info=True)
                raise # Re-raise to signal failure

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

    async def set(self, key: str, value: Any, tags: Optional[List[str]] = None):
        """Sets a key-value pair, optionally with tags. Complex values are stored as JSON."""
        async with self._lock:
            if self.conn is None: await self.initialize() # Ensure connection
            try:
                # Serialize non-basic types to JSON
                serialized_value = self._serialize(value)
                serialized_tags = json.dumps(tags) if tags else None

                await self.conn.execute(
                    "INSERT OR REPLACE INTO store (key, value, tags) VALUES (?, ?, ?)",
                    (key, serialized_value, serialized_tags)
                )
                await self.conn.commit()
                logger.debug(f"Set key='{key}', tags={tags}")
            except Exception as e:
                logger.error(f"Failed to set key '{key}' in SQLite store: {e}", exc_info=True)

    async def get(self, key: str) -> Optional[Any]:
        """Gets the value for a key, deserializing JSON if necessary."""
        async with self._lock:
            if self.conn is None: await self.initialize()
            try:
                async with self.conn.execute("SELECT value FROM store WHERE key = ?", (key,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        try:
                            # Attempt to deserialize JSON
                            return self._deserialize(row[0])
                        except (json.JSONDecodeError, TypeError):
                             # Return as string if not valid JSON
                            return row[0] 
                    else:
                        return None
            except Exception as e:
                logger.error(f"Failed to get key '{key}' from SQLite store: {e}", exc_info=True)
                return None

    async def delete(self, key: str):
        """Deletes a key."""
        async with self._lock:
            if self.conn is None: await self.initialize()
            try:
                await self.conn.execute("DELETE FROM store WHERE key = ?", (key,))
                await self.conn.commit()
                logger.debug(f"Deleted key='{key}'")
            except Exception as e:
                logger.error(f"Failed to delete key '{key}' from SQLite store: {e}", exc_info=True)

    async def find_keys_by_tag(self, tag: str, limit: Optional[int] = None) -> List[str]:
        """Finds keys associated with a specific tag using JSON search (inefficient for large datasets)."""
        # Note: This implementation uses JSON string matching, which is not indexed and
        # can be slow. For high performance, consider normalizing tags into a separate table.
        async with self._lock:
            if self.conn is None: await self.initialize()
            try:
                # Using json_each to potentially search within the JSON array
                # This is still potentially slow.
                # We construct a LIKE pattern to find the tag within the JSON list string.
                # This is a simplification and might match unintended keys if tag names overlap.
                # Example: searching for 'tag' might match 'new_tag'. Needs refinement for robustness.
                like_pattern = f'%"{tag}"%' # Look for the tag quoted within the JSON string list
                
                sql = "SELECT key FROM store WHERE tags IS NOT NULL AND json_valid(tags) AND tags LIKE ?"
                params = [like_pattern]
                
                if limit is not None and isinstance(limit, int) and limit > 0:
                    sql += " LIMIT ?"
                    params.append(limit)
                    
                async with self.conn.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()
                    return [row[0] for row in rows]
            except Exception as e:
                 logger.error(f"Failed to find keys by tag '{tag}' using LIKE: {e}", exc_info=True)
                 # Fallback or alternative strategy might be needed here.
                 # For now, return empty list on error.
                 return []

    async def get_all_keys(self) -> List[str]:
         """Gets all keys (potentially slow for very large stores)."""
         async with self._lock:
             if self.conn is None: await self.initialize()
             try:
                 async with self.conn.execute("SELECT key FROM store") as cursor:
                     rows = await cursor.fetchall()
                     return [row[0] for row in rows]
             except Exception as e:
                 logger.error(f"Failed to get all keys from SQLite store: {e}", exc_info=True)
                 return []

    async def close(self) -> None:
        async with self._lock:
            if self.conn:
                try:
                    await self.conn.close()
                    self.conn = None
                    logger.info(f"Database connection closed for {self.db_path}")
                except Exception as e:
                    logger.error(f"Failed to close SQLite connection: {e}", exc_info=True)