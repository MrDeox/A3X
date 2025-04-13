# a3x/core/context.py
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

class Context:
    """
    Context object passed to skills, providing access to shared resources.
    """
    def __init__(self,
                 logger: Optional[logging.Logger] = None,
                 workspace_root: Optional[Path] = None,
                 mem: Optional[Dict[str, Any]] = None, # Simple memory store
                 llm_url: Optional[str] = None,
                 tools: Optional[Dict[str, Any]] = None, # Available tools
                 # Add other necessary attributes as identified
                 ):
        self.logger = logger or logging.getLogger(__name__)
        self.workspace_root = workspace_root or Path('.')
        self.mem = mem if mem is not None else {}
        self.llm_url = llm_url
        self.tools = tools if tools is not None else {}
        # Potentially add execute_tool method reference if needed by skills

    # Add methods if Context needs specific functionality 

logger = logging.getLogger(__name__)

class ContextEntry:
    """Represents a single entry within the SharedTaskContext.

    Stores the actual value along with associated metadata like the source,
    timestamp, tags, and other custom metadata fields.
    """
    def __init__(self, value: Any, source: Optional[str] = None, timestamp: Optional[float] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Initializes a ContextEntry.

        Args:
            value: The data value being stored.
            source: Identifier of the fragment/skill that created/updated this entry.
            timestamp: Time of creation/update (defaults to current time).
            tags: List of string tags for categorization.
            metadata: Dictionary for additional metadata (e.g., confidence, priority).
        """
        self.value = value
        self.source = source # Fragmento/Skill que criou/modificou
        self.timestamp = timestamp or time.time()
        self.tags = tags or []
        self.metadata = metadata or {} # Para outros metadados: confidence, priority, etc.

    def __repr__(self):
        return f"ContextEntry(value={str(self.value)[:50]}..., source={self.source}, tags={self.tags}, meta_keys={list(self.metadata.keys())})"

class SharedTaskContext:
    """
    Manages shared state and intermediate results for a single agent task execution,
    now with support for metadata and tagging.
    
    This object acts like a temporary workspace or "whiteboard" for fragments
    collaborating on the same high-level objective within a single `run_task` call.
    It allows fragments to pass structured data and status updates to each other
    indirectly, coordinated by the Orchestrator's flow.

    The context is typically created at the start of an agent task and discarded
    once the task concludes (successfully or not).
    """
    def __init__(self, task_id: str, initial_objective: str):
        """
        Initializes the shared context for a specific task.

        Args:
            task_id: A unique identifier for the current task run.
            initial_objective: The initial objective provided to the agent.
        """
        self._task_id = task_id
        self._objective = initial_objective
        # Alterado para armazenar ContextEntry
        self._context_data: Dict[str, ContextEntry] = {}
        logger.debug(f"[SharedTaskContext:{self._task_id}] Initialized for objective: '{initial_objective}'")

    def set(self,
            key: str, 
            value: Any,
            source: Optional[str] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None):
        """
        Sets or updates a value in the shared context, storing it as a ContextEntry
        with associated metadata.

        Args:
            key: The unique key for the data entry.
            value: The value to store.
            source: Optional identifier for the fragment/skill setting the value.
            tags: Optional list of string tags for categorization/querying.
            metadata: Optional dictionary for additional metadata (e.g., priority, confidence).
        """
        entry = ContextEntry(value, source=source, tags=tags, metadata=metadata)
        logger.debug(f"[SharedTaskContext:{self._task_id}] Setting '{key}' = {entry!r}")
        self._context_data[key] = entry

    def get_entry(self, key: str) -> Optional[ContextEntry]:
        """Retrieves the full ContextEntry object associated with a key.

        This allows access to the value as well as all associated metadata (source,
        timestamp, tags, etc.).

        Args:
            key: The key of the entry to retrieve.

        Returns:
            The ContextEntry object, or None if the key is not found.
        """
        return self._context_data.get(key)

    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Retrieves only the underlying value associated with a key from the shared context.

        Args:
            key: The key of the data entry to retrieve.
            default: The value to return if the key is not found (defaults to None).

        Returns:
            The stored value associated with the key, or the default value.
        """
        entry = self.get_entry(key)
        value = entry.value if entry else default
        logger.debug(f"[SharedTaskContext:{self._task_id}] Getting value for '{key}'. Found: {entry is not None}")
        return value

    def get_by_tag(self, tag: str) -> Dict[str, ContextEntry]:
        """Retrieves all context entries that have a specific tag.

        Allows filtering the context based on assigned tags.

        Args:
            tag: The tag to search for.

        Returns:
            A dictionary mapping keys to their corresponding ContextEntry objects
            that contain the specified tag.
        """
        logger.debug(f"[SharedTaskContext:{self._task_id}] Getting entries tagged with '{tag}'")
        return {k: entry for k, entry in self._context_data.items() if tag in entry.tags}

    def get_by_source(self, source: str) -> Dict[str, ContextEntry]:
        """Retrieves all context entries originating from a specific source.

        Allows filtering the context based on which fragment/skill created the entry.

        Args:
            source: The source identifier to search for.

        Returns:
            A dictionary mapping keys to their corresponding ContextEntry objects
            originating from the specified source.
        """
        logger.debug(f"[SharedTaskContext:{self._task_id}] Getting entries from source '{source}'")
        return {k: entry for k, entry in self._context_data.items() if entry.source == source}

    # Manter update e get_all_data (adaptar se necessário)
    def update(self, data: Dict[str, Any], source: Optional[str] = None):
         """
        Updates the context with multiple key-value pairs from a dictionary.
        Note: This currently creates basic ContextEntry objects without rich tags/metadata
        for the bulk update. Enhance if needed.

        Args:
            data: A dictionary containing key-value pairs to merge.
            source: Optional source identifier applied to all updated entries.
        """
         logger.debug(f"[SharedTaskContext:{self._task_id}] Updating context with {len(data)} keys from source '{source}'.")
         for key, value in data.items():
             # Creates simple entries; enhance if needed
             self.set(key, value, source=source)

    # <<< RENAMED from get_all_data >>>
    def get_all_entries(self) -> Dict[str, ContextEntry]:
        """Returns a copy of the entire context data dictionary containing ContextEntry objects.

        Provides a snapshot of the full shared context, including all values and metadata.

        Returns:
            A dictionary mapping keys to their corresponding ContextEntry objects.
        """
        return self._context_data.copy()

    # Adaptar __str__ e __repr__
    def __str__(self) -> str:
        return f"SharedTaskContext(task_id={self._task_id}, objective='{self._objective}', data_keys={list(self._context_data.keys())})"

    def __repr__(self) -> str:
        return f"<SharedTaskContext task_id={self._task_id} data_keys={list(self._context_data.keys())}>" 