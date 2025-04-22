# a3x/cli/agent_utils.py
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

from rich.console import Console

# Import type only for type checking
if TYPE_CHECKING:
    from a3x.core.cerebrumx import CerebrumXAgent

# Import core components needed for runtime
try:
    # Runtime import still needed for the actual class usage - aliased
    from a3x.core.cerebrumx import CerebrumXAgent as RuntimeCerebrumXAgent
    from a3x.fragments.registry import FragmentRegistry
    from a3x.core.tool_registry import ToolRegistry
    from a3x.core.memory.memory_manager import MemoryManager
    from a3x.core.config import PROJECT_ROOT, MAX_REACT_ITERATIONS
except ImportError as e:
    print(f"[CLI Agent Utils Error] Failed to import core modules: {e}")
    # Fallbacks for runtime - aliased
    RuntimeCerebrumXAgent = None
    FragmentRegistry = None
    ToolRegistry = None
    MemoryManager = None
    PROJECT_ROOT = Path(".")
    MAX_REACT_ITERATIONS = 10

logger = logging.getLogger(__name__)
console = Console()

def initialize_agent(
    system_prompt: str,
    tool_registry: ToolRegistry,
    llm_url_override: Optional[str] = None,
    max_steps: Optional[int] = None
) -> Optional["CerebrumXAgent"]:
    """Initializes the CerebrumX agent instance."""
    # Use the runtime import/fallback (aliased) for checks and instantiation
    if not all([RuntimeCerebrumXAgent, FragmentRegistry, ToolRegistry, MemoryManager]):
        logger.error("Core agent components (Agent, Registries, Manager) not available due to import error.")
        return None
    
    effective_max_steps = max_steps if max_steps is not None else MAX_REACT_ITERATIONS

    try:
        agent_id = f"cli-agent-{uuid.uuid4().hex[:8]}"
        workspace_root = Path(PROJECT_ROOT).resolve()
        
        fragment_registry = FragmentRegistry()

        memory_config = {
            "SEMANTIC_INDEX_PATH": "data/indexes/semantic_memory",
            "DATABASE_PATH": "data/a3x_main.db",
        }
        memory_manager = MemoryManager(config=memory_config) 

        # Instantiate using the runtime import/fallback (aliased)
        agent = RuntimeCerebrumXAgent(
            agent_id=agent_id,
            system_prompt=system_prompt,
            tool_registry=tool_registry,
            fragment_registry=fragment_registry,
            memory_manager=memory_manager,
            workspace_root=workspace_root,
            llm_url=llm_url_override,
            max_iterations=effective_max_steps
        )
        logger.info(f"Agent initialized successfully (ID: {agent_id}, Max Steps: {effective_max_steps}).")
        return agent
    except Exception as e:
        logger.exception("Failed to initialize agent:")
        console.print(f"[bold red]Error initializing agent:[/bold red] {e}")
        return None 