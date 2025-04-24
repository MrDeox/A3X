# a3x/core/mem.py
import logging
from typing import Any, Dict, Optional, List, Union

# Imports for context creation and execution
from a3x.core.context import Context
from a3x.core.tool_executor import ToolExecutor
from a3x.core.registry_instance import SKILL_REGISTRY
from a3x.core.skills import discover_skills
from a3x.core.llm_interface import LLMInterface # To instantiate if context is None
from a3x.core.config import PROJECT_ROOT # Default workspace root
# Corrigido: não existem 'Config' nem 'get_config' no config.py
# from a3x.core.config import Config, get_config
from a3x.core.logging_config import setup_logging
from a3x.core.tool_registry import ToolRegistry # Import ToolRegistry class
from a3x.core.db_utils import (
    initialize_database,
    add_episodic_record,
    retrieve_recent_episodes
)

# Embeddings/FAISS related (make optional?)
logger = logging.getLogger(__name__)

try:
    from a3x.core.embeddings import get_embedding # Function
    from a3x.core.semantic_memory_backend import (
        initialize_faiss_index,
        add_to_faiss_index,
        search_index
    )
    FAISS_ENABLED = True
except ImportError:
    logger.warning("FAISS or sentence-transformers not found. Semantic search functionality will be disabled.")
    FAISS_ENABLED = False
    # Define dummy functions if FAISS is not enabled
    def initialize_faiss_index(*args, **kwargs):
        pass
    def add_to_faiss_index(*args, **kwargs):
        pass
    def search_index(*args, **kwargs):
        return []

# Placeholder for potential future interpreter selection logic
# DEFAULT_INTERPRETER = "a3lang" 

# Default execution mode
DEFAULT_MODE = "symbolic"

async def execute(command: str, context: Optional[Context] = None, mode: str = DEFAULT_MODE) -> Dict[str, Any]:
    """
    Executes a command string using the appropriate interpreter based on the mode.

    Args:
        command: The command string to execute (e.g., A³L command).
        context: Optional context object. If None in symbolic mode, a basic one will be created.
        mode: The execution mode ('symbolic' or 'neural'). Defaults to symbolic.

    Returns:
        A dictionary containing the execution result.
    """
    logger.info(f"MEM Execute received command: '{command}' (Mode: {mode})")
    
    if mode == "symbolic":
        context_to_pass = context # Start with the provided context

        # If no context is provided, ensure skills are loaded for the interpreter to use.
        if context_to_pass is None:
            logger.info("No context provided for symbolic execution. Ensuring skills are loaded...")
            # Ensure skills are loaded into the global registry
            if not SKILL_REGISTRY.list_tools():
                logger.info("Skill registry empty, loading default skill packages...")
                try:
                    default_skill_packages = ['a3x.skills.core', 'a3x.skills.auto_generated']
                    discover_skills(default_skill_packages)
                    logger.info(f"Skills loaded successfully from {default_skill_packages}.")
                    if not SKILL_REGISTRY.list_tools():
                         logger.warning("discover_skills completed but skill registry still seems empty.")
                except Exception as e:
                    logger.exception("Failed to load skills automatically.")
                    # Return an error, as execution likely cannot proceed without skills
                    return {"status": "error", "message": f"Symbolic execution requires skills, but loading failed: {e}"}
            else:
                logger.info("Skill registry already contains skills.")
            # We assume the component calling parse_and_execute (or parse_and_execute itself)
            # will handle creating the necessary execution context, including the ToolExecutor.
            # try:
            #     # Instantiate necessary components for a basic context
            #     tool_executor = ToolExecutor(tool_registry=SKILL_REGISTRY)
            #     llm_interface = LLMInterface() # Instantiates with default config/fallbacks
            #     
            #     # Create the basic Context object
            #     context_to_pass = Context(
            #         logger=logger, # Use module logger
            #         # tool_executor=tool_executor, # REMOVED: Context doesn't take this
            #         llm_interface=llm_interface,
            #         workspace_root=PROJECT_ROOT, 
            #         # Provide None or defaults for other potentially required fields
            #         mem={},
            #         tools=SKILL_REGISTRY.list_tools(), # Pass descriptions perhaps?
            #     )
            #     logger.info("Basic components (ToolExecutor, LLMInterface) instantiated for potential use by interpreter.")
            # except Exception as e:
            #     logger.exception("Failed to create basic context components.")
            #     return {"status": "error", "message": f"Failed to initialize basic context components: {e}"}

        try:
            # Import the interpreter function
            from a3x.a3lang.interpreter import parse_and_execute
            
            # Call the interpreter, passing the original context (which might be None)
            # The interpreter is now responsible for creating its execution context if context_to_pass is None.
            logger.info(f"Calling a3lang.parse_and_execute with command and context (type: {type(context_to_pass).__name__})...")
            result = await parse_and_execute(command, execution_context=context_to_pass)
            logger.info(f"A³Lang execution finished. Result status: {result.get('status')}")
            return result
            
        except ImportError as e:
            logger.exception(f"Failed to import the symbolic interpreter (a3lang): {e}")
            return {"status": "error", "message": f"Symbolic interpreter import failed: {e}"}
        except Exception as e:
            logger.exception(f"Error during symbolic command execution '{command}': {e}")
            return {"status": "error", "message": f"Symbolic execution failed: {e}"}
            
    elif mode == "neural":
        logger.warning(f"Neural execution mode requested for command '{command}', but it is not yet implemented.")
        # Placeholder for neural execution
        # from a3x.a3net.bridge import execute_neural_command # Example
        # return await execute_neural_command(command, context)
        raise NotImplementedError("Modo 'neural' ainda não implementado.")
        
    else:
        logger.error(f"Unknown execution mode requested: '{mode}'")
        # Return an error or raise ValueError? Returning error for now.
        return {"status": "error", "message": f"Modo desconhecido: {mode}"}
        # raise ValueError(f"Modo desconhecido: {mode}")

# --- Helper function (example for future) ---
# def determine_interpreter(command: str, context: Any) -> str:
#     """Placeholder logic to determine which interpreter to use."""
#     # Simple logic for now: default to a3lang
#     return "a3lang"

# --- Potential synchronous wrapper (if needed) ---
# import asyncio
# def execute_sync(command: str, context: Any = None, mode: str = DEFAULT_MODE) -> Dict[str, Any]:
#    """Synchronous wrapper for the execute function."""
#    # Be careful with running async event loops within existing ones if wrapping sync calls
#    # Might need asyncio.get_event_loop().run_until_complete(...) if already in a loop
#    try:
#        return asyncio.run(execute(command, context, mode))
#    except RuntimeError as e:
#        if "cannot run current event loop" in str(e):
#             logger.warning("Attempted to run execute_sync from within an existing event loop. Consider calling execute directly.")
#             # Handle nested loop scenario if necessary
#             loop = asyncio.get_running_loop()
#             return loop.run_until_complete(execute(command, context, mode))
#        else:
#             raise 

class MemoryManager:
    """Manages different types of memory: episodic, semantic, and heuristic."""

    def __init__(self, db_path: Optional[str] = None, semantic_index_path: Optional[str] = None, heuristic_log_path: Optional[str] = None, episodic_limit: int = 10, tool_registry: Optional['ToolRegistry'] = None):
        """Initializes the MemoryManager.
        Args:
            db_path: Path to the episodic/heuristic database.
            semantic_index_path: Path to the FAISS semantic index.
            heuristic_log_path: Path to the heuristic log.
            episodic_limit: Max number of episodic memories to retrieve by default.
            tool_registry: Optional ToolRegistry instance. If not provided, a default one will be created.
        """
        from a3x.core.config import DATABASE_PATH, SEMANTIC_INDEX_PATH
        self.db_path = db_path or DATABASE_PATH
        self.semantic_index_path = semantic_index_path or SEMANTIC_INDEX_PATH if 'SEMANTIC_INDEX_PATH' in globals() else None
        self.heuristic_log_path = heuristic_log_path or None
        self.episodic_limit = episodic_limit
        self.faiss_enabled = FAISS_ENABLED
        self.tool_registry = tool_registry or ToolRegistry()
        # Initialize DB
        initialize_database(self.db_path)
        logger.info(f"MemoryManager initialized with DB: {self.db_path}")

        # Initialize FAISS if enabled
        if self.faiss_enabled:
            initialize_faiss_index(index_path_base=self.semantic_index_path)
            logger.info(f"FAISS index initialized at: {self.semantic_index_path}")

        # Tool Registry (primarily for semantic memory context)
        if tool_registry:
             self.tool_registry = tool_registry
             logger.debug("MemoryManager using provided ToolRegistry instance.")
        else:
             logger.warning("No ToolRegistry provided to MemoryManager. Creating a default one and loading skills.")
             self.tool_registry = ToolRegistry()
             try:
                 discover_skills() # Populate the global SKILL_REGISTRY
                 # Now populate the local tool_registry instance
                 if SKILL_REGISTRY:
                      for skill_name, skill_info in SKILL_REGISTRY.items():
                           # Simplified registration - assumes skill_info structure is correct
                           # Add more robust checks if necessary
                           if isinstance(skill_info, dict) and "function" in skill_info and "schema" in skill_info:
                                self.tool_registry.register_tool(
                                     name=skill_name,
                                     instance=None, # Assuming function based
                                     tool=skill_info["function"],
                                     schema=skill_info["schema"]
                                )
                           else:
                                logger.warning(f"Skipping registration for '{skill_name}' due to invalid format in SKILL_REGISTRY.")
                      logger.info(f"Default ToolRegistry populated with {len(self.tool_registry.list_tools())} skills for MemoryManager.")
                 else:
                      logger.warning("Global SKILL_REGISTRY is empty after discovery. Default ToolRegistry for MemoryManager is empty.")

             except Exception as e:
                  logger.exception("Failed to discover/load skills for MemoryManager's default ToolRegistry.")

    # --- Episodic Memory --- #

    def add_episodic(self, description: str, event_type: str, outcome: Union[str, int, bool], metadata: Optional[Dict[str, Any]] = None) -> int:
        """Adds a record to the episodic memory (SQLite)."""
        return add_episodic_record(description, event_type, outcome, metadata, db_path=self.db_path)

    def get_recent_episodic(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves recent episodic records."""
        effective_limit = limit if limit is not None else self.episodic_limit
        return get_recent_episodes(limit=effective_limit, db_path=self.db_path)

    # --- Semantic Memory (Conceptual - relies on FAISS + DB for metadata) --- #

    def add_semantic(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Adds content to semantic memory (generates embedding, stores in FAISS + metadata in DB)."""
        if not self.faiss_enabled:
            logger.warning("Cannot add semantic memory: FAISS is disabled.")
            return None
        
        metadata = metadata or {}
        metadata['content'] = content # Ensure content is in metadata for retrieval
        
        try:
            embedding = get_embedding(content)
            if embedding is None:
                 logger.error("Failed to generate embedding for semantic content. Cannot add.")
                 return None

            # Here, we need a mechanism to store the metadata and associate it with the FAISS index entry.
            # Option 1: Store metadata in DB, get an ID, add ID to FAISS vector metadata.
            # Option 2: Rely on FAISS metadata storage (limited size). Let's assume FAISS handles it for now.
            # For FAISS, we usually need a unique ID for each vector.
            vector_id = hash(content) # Simple hash as ID, might need better approach
            
            # Add to FAISS index
            add_to_faiss_index(
                index_path_base=self.semantic_index_path, 
                embeddings=[embedding], 
                ids=[vector_id], 
                metadatas=[metadata] # Pass metadata directly
            )
            logger.info(f"Added semantic content (ID: {vector_id}) to FAISS index.")
            return str(vector_id) # Return the ID used in FAISS

        except Exception as e:
            logger.exception(f"Failed to add semantic memory entry: {e}")
            return None

    def find_semantic(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Finds similar semantic records using embeddings and FAISS."""
        if not self.faiss_enabled:
            logger.warning("Cannot search semantic memory: FAISS is disabled.")
            return []
        
        try:
            query_embedding = get_embedding(query)
            if query_embedding is None:
                 logger.error("Failed to generate embedding for semantic query. Cannot search.")
                 return []

            search_results = search_index(
                index_path_base=self.semantic_index_path, 
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Results from search_index should already contain metadata if stored with add_to_faiss_index
            return search_results # Directly return results [{id: ..., distance: ..., metadata: {...}}, ...]

        except Exception as e:
            logger.exception(f"Failed to search semantic memory: {e}")
            return []

    # --- Heuristic Memory --- #

    def add_heuristic(self, heuristic_type: str, trigger: str, action: str, confidence: float, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Adds a heuristic record to the heuristic memory (SQLite)."""
        return add_heuristic_record(heuristic_type, trigger, action, confidence, metadata, db_path=self.db_path)

    def get_heuristic(self, heuristic_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a specific heuristic by its ID."""
        return get_heuristic_record_by_id(heuristic_id, db_path=self.db_path)

    def get_all_heuristics(self) -> List[Dict[str, Any]]:
        """Retrieves all stored heuristics."""
        return get_all_heuristics(db_path=self.db_path)

    def get_contextual_summary(self, query: Optional[str] = None, episodic_limit: int = 5, semantic_top_k: int = 3) -> str:
        """Provides a summary of relevant memories based on a query or recent activity."""
        summary = "" 
        
        # Episodic Summary
        try:
            recent_episodes = self.get_recent_episodic(limit=episodic_limit)
            if recent_episodes:
                summary += "\n--- Recent Activity (Episodic Memory) ---\n"
                for ep in recent_episodes:
                     ts = ep.get('timestamp', '').split('.')[0]
                     summary += f"[{ts}] {ep.get('event_type', '?')}: {ep.get('description', '')[:80]}... (Outcome: {ep.get('outcome')})\n"
            else:
                summary += "\nNo recent episodic memory found.\n"
        except Exception as e:
             logger.error(f"Error retrieving recent episodic memory for summary: {e}")
             summary += "\nError accessing episodic memory.\n"
             
        # Semantic Summary (if query provided and FAISS enabled)
        if query and self.faiss_enabled:
            try:
                semantic_results = self.find_semantic(query, top_k=semantic_top_k)
                if semantic_results:
                     summary += f"\n--- Relevant Concepts (Semantic Memory for '{query[:30]}...') ---\n"
                     for res in semantic_results:
                          meta = res.get('metadata', {})
                          content = meta.get('content', '<content missing>')
                          dist = res.get('distance', -1.0)
                          summary += f"[Dist: {dist:.3f}] {content[:100]}...\n"
                else:
                     summary += f"\nNo relevant semantic memory found for query.\n"
            except Exception as e:
                 logger.error(f"Error retrieving semantic memory for summary: {e}")
                 summary += "\nError accessing semantic memory.\n"
        elif query:
             summary += "\nSemantic search disabled (FAISS not available).\n"
             
        # Heuristic Summary (Optional - might be too verbose)
        # try:
        #     heuristics = self.get_all_heuristics()
        #     if heuristics:
        #         summary += "\n--- Available Heuristics ---\n"
        #         for h in heuristics[:5]: # Limit summary
        #             summary += f"[{h.get('type')}] Trigger: {h.get('trigger')[:50]}... -> Action: {h.get('action')[:50]}... (Conf: {h.get('confidence')})\n"
        # except Exception as e:
        #      logger.error(f"Error retrieving heuristics for summary: {e}")

        return summary.strip() 