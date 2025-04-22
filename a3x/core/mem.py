# a3x/core/mem.py
import logging
from typing import Any, Dict, Optional

# Imports for context creation and execution
from a3x.core.context import Context
from a3x.core.tool_executor import ToolExecutor
from a3x.core.registry_instance import SKILL_REGISTRY
from a3x.core.skills import load_all_skills
from a3x.core.llm_interface import LLMInterface # To instantiate if context is None
from a3x.core.config import PROJECT_ROOT # Default workspace root

logger = logging.getLogger(__name__)

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
                    load_all_skills(default_skill_packages)
                    logger.info(f"Skills loaded successfully from {default_skill_packages}.")
                    if not SKILL_REGISTRY.list_tools():
                         logger.warning("load_all_skills completed but skill registry still seems empty.")
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