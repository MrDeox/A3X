# a3x/skills/core/reload_fragments.py
import logging
from typing import Dict, Any, Optional

from a3x.core.skills import skill
from a3x.core.tool_executor import _ToolExecutionContext
# Import FragmentRegistry to check type, avoid circular import if possible
# from a3x.fragments.registry import FragmentRegistry

logger = logging.getLogger(__name__)

@skill(
    name="reload_fragments",
    description="Reloads and re-discovers all available agent fragments dynamically.",
    parameters={ 
        # No parameters needed, uses context
        "ctx": {"type": "Context", "description": "The execution context providing access to the fragment registry."}
    }
)
async def reload_fragments(ctx: _ToolExecutionContext) -> Dict[str, Any]:
    """Dynamically reloads agent fragments by re-scanning the fragments directory.

    This skill attempts to find the `FragmentRegistry` instance within the provided
    execution context (`ctx`). If found, it calls the registry's
    `discover_and_register_fragments` method with `force_reload=True`.

    This process:
    1. Clears the registry's current definitions, classes, and instances.
    2. Re-imports (or reloads if already imported) all modules within the
       `a3x.fragments` package (except specific registry/base files).
    3. Scans the reloaded modules for classes decorated with `@fragment`.
    4. Registers the discovered fragments based on their decorator metadata.

    This allows adding new fragment files or modifying existing ones and having
    the changes reflected in the running agent without a restart.

    Args:
        ctx: The execution context, which must contain an attribute
             (e.g., `fragment_registry`) pointing to the `FragmentRegistry` instance.

    Returns:
        A dictionary indicating the status ('success' or 'error') and providing
        a message, including the count of registered definitions after the reload.
        Example success: `{"status": "success", "action": "fragments_reloaded", "data": {"message": "...", "registered_count": 5}}`
        Example error: `{"status": "error", "action": "reload_failed", "data": {"message": "..."}}`
    """
    log_prefix = "[ReloadFragments Skill]"
    logger.info(f"{log_prefix} Attempting to reload agent fragments...")

    # Access the FragmentRegistry instance - How is it passed? Assume it's in ctx for now.
    # This is a potential point of failure if the registry isn't accessible this way.
    fragment_registry = None
    if hasattr(ctx, 'fragment_registry'): # Check if context has the registry
        fragment_registry = ctx.fragment_registry
        # Optional: Check type if FragmentRegistry was imported
        # if not isinstance(fragment_registry, FragmentRegistry):
        #     logger.error(f"{log_prefix} Context attribute 'fragment_registry' is not the correct type.")
        #     return {"status": "error", "data": {"message": "Internal error: FragmentRegistry not found or invalid in context."}}
    
    if not fragment_registry or not hasattr(fragment_registry, 'discover_and_register_fragments'):
        logger.error(f"{log_prefix} FragmentRegistry instance not found or accessible in the provided context.")
        return {"status": "error", "data": {"message": "Internal error: Could not access FragmentRegistry."}}

    try:
        # Call the discovery method with force_reload=True
        await asyncio.to_thread(fragment_registry.discover_and_register_fragments, force_reload=True)
        
        # Get updated count
        num_registered = len(fragment_registry.get_all_definitions())
        message = f"Fragment registry reloaded successfully. {num_registered} definitions are now registered."
        logger.info(f"{log_prefix} {message}")
        return {
            "status": "success",
            "action": "fragments_reloaded",
            "data": {"message": message, "registered_count": num_registered}
        }
    except Exception as e:
        logger.exception(f"{log_prefix} Error occurred during fragment reload:")
        return {
            "status": "error",
            "action": "reload_failed",
            "data": {"message": f"Failed to reload fragments: {e}"}
        } 