# a3x/core/context.py
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple, NamedTuple, TYPE_CHECKING, Type, Callable, Coroutine
import asyncio
from dataclasses import dataclass, field
from collections import namedtuple # Ensure namedtuple is imported
import uuid # <<< ADDED import >>>

# Import the new types
from a3x.core.types import ActionIntent, PendingRequest # <<< ADDED IMPORT >>>

# <<< ADDED Forward references if needed by FragmentContext >>>
if TYPE_CHECKING:
    from a3x.core.llm_interface import LLMInterface
    from a3x.core.tool_registry import ToolRegistry
    from a3x.fragments.registry import FragmentRegistry, FragmentDef # Assuming FragmentRegistry lives here
    from a3x.core.memory.memory_manager import MemoryManager # Added
    from a3x.fragments.base import BaseFragment # <<< ADDED for active_fragments typing >>>

# Moved from tool_executor.py
_ToolExecutionContext = namedtuple("ToolExecutionContext", [
    "logger", 
    "workspace_root", 
    "llm_url", 
    "tools_dict", 
    "llm_interface",
    "fragment_registry",
    "shared_task_context",
    "allowed_skills",
    "skill_instance",
    "memory_manager"
])

# <<< ADDED: Fragment Definition type hint >>>
if TYPE_CHECKING:
    from a3x.fragments.registry import FragmentDef # Assuming FragmentDef is defined here

# Updated FragmentContext definition and __new__ method
class FragmentContext(namedtuple('FragmentContext', [
    'logger', 
    'llm_interface', 
    'tool_registry', 
    'fragment_registry',
    'shared_task_context',
    'workspace_root',
    'memory_manager',
    # <<< ADDED Fields >>>
    'fragment_id', 
    'fragment_name', 
    'fragment_class', 
    'fragment_def', 
    'config' 
])):
    # Adding __new__ with default for optional field and new fields
    def __new__(cls, logger: logging.Logger, llm_interface: 'LLMInterface', 
                tool_registry: 'ToolRegistry', fragment_registry: 'FragmentRegistry', 
                shared_task_context: Optional[Dict[str, Any]], workspace_root: Path,
                # <<< ADDED Parameters >>>
                fragment_id: str, 
                fragment_name: str, 
                fragment_class: Type, 
                fragment_def: 'FragmentDef', 
                # <<< ADDED config parameter >>>
                config: Dict[str, Any], # Assuming config is a required Dict
                memory_manager: Optional['MemoryManager'] = None): # Default to None
        return super().__new__(cls, logger, llm_interface, tool_registry, 
                               fragment_registry, shared_task_context, workspace_root,
                               memory_manager, 
                               # <<< ADDED Arguments >>>
                               fragment_id, fragment_name, fragment_class, fragment_def,
                               # <<< ADDED config argument >>>
                               config
                               )

class Context:
    """
    Context object passed to skills, providing access to shared resources.
    Now includes direct access to LLMInterface and ToolExecutor.
    """
    def __init__(self,
                 logger: Optional[logging.Logger] = None,
                 workspace_root: Optional[Path] = None,
                 mem: Optional[Dict[str, Any]] = None, # Simple memory store
                 llm_url: Optional[str] = None, # Can be derived from llm_interface if provided
                 llm_interface: Optional['LLMInterface'] = None, # <<< ADDED >>>
                 tools: Optional[Dict[str, Any]] = None, # Available tools
                 tool_executor: Optional[Any] = None, # Explicit ToolExecutor instance
                 # Add other necessary attributes as identified
                 ):
        self.logger = logger or logging.getLogger(__name__)
        self.workspace_root = workspace_root or Path('.')
        self.mem = mem if mem is not None else {}
        self.llm_interface = llm_interface # <<< ADDED >>>
        # Derive llm_url from interface if not provided directly
        self.llm_url = llm_url or (llm_interface.llm_url if llm_interface else None)
        self.tools = tools if tools is not None else {}
        self.tool_executor = tool_executor # <<< ADDED >>>
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

@dataclass
class SharedTaskContext:
    """
    Contexto compartilhado de task entre fragments, com suporte robusto a execução simbólica real.
    Inclui propagação automática do ToolExecutor real para comandos simbólicos.
    """
    # --- Campos obrigatórios ---
    task_id: str
    initial_objective: Optional[str] = None
    task_data: Dict[str, Any] = field(default_factory=dict)
    execution_history: List[Tuple[str, str]] = field(default_factory=list)
    # --- Chat and Active Fragment Fields ---
    internal_chat_queue: asyncio.Queue = field(default_factory=asyncio.Queue) # <<< CHANGED to Queue >>>
    active_fragments: Dict[str, 'BaseFragment'] = field(default_factory=dict) # <<< ADDED active fragments tracker >>>
    # --- ADDED Fields for structured requests and actions ---
    pending_request: Optional[PendingRequest] = None
    action_intent: Optional[ActionIntent] = None
    # --- End ADDED Fields ---
    # --- Other Fields ---
    sandbox_dialogue_queue: List[Tuple[str, Dict]] = field(default_factory=list) # Kept for sandbox mode
    error_info: Optional[Any] = None
    result: Optional[Any] = None
    # --- Campo opcional para execução simbólica real ---
    tool_executor: Optional[Any] = None  # Executor real de skills/tools
    async def run_symbolic_command(self, command: str, **kwargs):
        """
        Executa um comando simbólico real usando o executor do A3X, com contexto persistente e recursos reais.
        Propaga automaticamente o ToolExecutor real se estiver presente.
        """
        from a3x.core.mem import execute as symbolic_execute
        from a3x.core.context import Context
        context = Context(
            logger=getattr(self, 'logger', None),
            workspace_root=getattr(self, 'workspace_root', None),
            mem=getattr(self, 'task_data', {}),
            llm_interface=getattr(self, 'llm_interface', None),
            tools=getattr(self, 'tools', None),
            tool_executor=self.tool_executor if hasattr(self, 'tool_executor') else None
        )
        return await symbolic_execute(command, context=context, mode="symbolic")

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
    task_id: str
    initial_objective: Optional[str] = None
    task_data: Dict[str, Any] = field(default_factory=dict)
    execution_history: List[Tuple[str, str]] = field(default_factory=list)
    # --- Chat and Active Fragment Fields ---
    internal_chat_queue: asyncio.Queue = field(default_factory=asyncio.Queue) # <<< CHANGED to Queue >>>
    active_fragments: Dict[str, 'BaseFragment'] = field(default_factory=dict) # <<< ADDED active fragments tracker >>>
    # --- ADDED Fields for structured requests and actions ---
    pending_request: Optional[PendingRequest] = None
    action_intent: Optional[ActionIntent] = None
    # --- End ADDED Fields ---
    # --- Other Fields ---
    sandbox_dialogue_queue: List[Tuple[str, Dict]] = field(default_factory=list) # Kept for sandbox mode
    _context_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    error_info: Optional[Dict[str, Any]] = None # <<< ADDED field for error info >>>
    result: Any = None # <<< ADDED field for final result >>>

    async def update_data(self, key: str, value: Any):
        async with self._context_lock:
            self.task_data[key] = value

    async def get_data(self, key: str, default: Any = None) -> Any:
        async with self._context_lock:
            return self.task_data.get(key, default)

    async def get_task_data_copy(self) -> Dict[str, Any]:
        """Returns a copy of the entire task_data dictionary."""
        async with self._context_lock:
            return dict(self.task_data) # Return a copy

    async def add_history(self, fragment_name: str, result_summary: str):
        async with self._context_lock:
            self.execution_history.append((fragment_name, result_summary))

    async def get_history(self) -> List[Tuple[str, str]]:
        async with self._context_lock:
            return list(self.execution_history)

    async def add_sandbox_message(self, fragment_name: str, message: Dict):
        """Adds a message to the shared sandbox dialogue queue."""
        async with self._context_lock:
            self.sandbox_dialogue_queue.append((fragment_name, message))

    async def get_sandbox_messages(self, since_index: int = 0) -> List[Tuple[str, Dict]]:
        """Gets messages from the sandbox dialogue queue after a given index."""
        async with self._context_lock:
            return list(self.sandbox_dialogue_queue[since_index:])

    # <<< MODIFIED Chat Methods for Queue >>>
    def add_chat_message(self, fragment_name: str, message_type: str, message_content: Dict) -> str:
        """Adds a message to the internal chat queue (non-blocking)."""
        # No lock needed for asyncio.Queue put_nowait
        message_id = f"chat-{uuid.uuid4()}"
        timestamp = time.time()
        chat_entry = {
            "timestamp": timestamp,
            "sender": fragment_name,
            "type": message_type.upper(), # Standardize type to uppercase
            "content": message_content,
            "message_id": message_id
        }
        try:
            self.internal_chat_queue.put_nowait(chat_entry)
            logger.debug(f"Added chat message from {fragment_name} (ID: {message_id}) to queue.")
            return message_id
        except asyncio.QueueFull:
            logger.error(f"Internal chat queue is full! Could not add message from {fragment_name}.")
            # Handle queue full? Maybe log and drop, or wait? For now, log and drop.
            return "error-queue-full"
        # Removed get_chat_messages - consumption happens via queue.get()
    # <<< END MODIFIED Chat Methods >>>

    # <<< ADDED Active Fragment Management Methods >>>
    async def register_active_fragment(self, name: str, instance: 'BaseFragment'):
        async with self._context_lock: # Protect dictionary access
            self.active_fragments[name] = instance
            logger.debug(f"Registered active fragment: {name}")

    async def unregister_active_fragment(self, name: str):
        async with self._context_lock:
            if name in self.active_fragments:
                del self.active_fragments[name]
                logger.debug(f"Unregistered active fragment: {name}")

    async def get_active_fragment(self, name: str) -> Optional['BaseFragment']:
        async with self._context_lock:
            return self.active_fragments.get(name)
            
    async def get_all_active_fragment_names(self) -> List[str]:
        async with self._context_lock:
            return list(self.active_fragments.keys())
    # <<< END ADDED Active Fragment Methods >>>

    def __str__(self) -> str:
        # <<< UPDATED __str__ for Queue and Active Fragments >>>
        qsize = self.internal_chat_queue.qsize() # Non-blocking size check
        active_names = list(self.active_fragments.keys()) # Get keys without lock (dict keys are atomic)
        data_keys = list(self.task_data.keys()) # Get data keys
        return f"SharedTaskContext(task_id={self.task_id}, objective='{self.initial_objective}', data_keys={data_keys}, chat_qsize={qsize}, active={active_names})"

    def __repr__(self) -> str:
        # <<< UPDATED __repr__ for Queue and Active Fragments >>>
        qsize = self.internal_chat_queue.qsize()
        active_names = list(self.active_fragments.keys())
        data_keys = list(self.task_data.keys())
        return f"<SharedTaskContext task_id={self.task_id} data_keys={data_keys} chat_qsize={qsize} active_count={len(active_names)}>"

    def to_dict(self) -> Dict[str, Any]:
        """Returns a serializable dictionary representation of the context."""
        # Note: asyncio.Lock is not serializable, so we exclude it.
        # ActionIntent and PendingRequest are dataclasses and should be serializable
        # if their contents are.
        # We also might need to handle non-serializable items in task_data if they exist.
        return {
            "task_id": self.task_id,
            "initial_objective": self.initial_objective,
            "task_data": self.task_data, # Consider deep copying or filtering non-serializable items
            "execution_history": self.execution_history,
            "pending_request": self.pending_request, # Added
            "action_intent": self.action_intent,   # Added
            "error_info": self.error_info,       # Added
            "result": self.result,             # Added
            "sandbox_dialogue_queue": self.sandbox_dialogue_queue,
            # Relevant info might be extracted if needed for specific logging/saving
        }

# Ensure ToolRegistry and FragmentRegistry are imported if needed for type hints elsewhere,
# or handle potential circular imports carefully.
# from .tool_registry import ToolRegistry
# from a3x.fragments.registry import FragmentRegistry 