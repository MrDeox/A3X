import logging
from typing import Dict, Optional, Any
from pathlib import Path

# Assuming these are available in the environment
from a3x.core.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

async def handle_export_fragment(
    directive: Dict[str, Any],
    memory_manager: MemoryManager
) -> Optional[Dict[str, Any]]:
    """Handles the 'export_fragment' directive logic."""

    fragment_id = directive.get("fragment_id")
    export_path_str = directive.get("path") # Optional path from directive

    if not fragment_id or not isinstance(fragment_id, str):
        logger.error("[A3X Bridge Handler - Export] 'fragment_id' (string) missing or invalid.")
        # Original code used print here, kept logger consistent
        return { "status": "error", "error": f"Export failed: 'fragment_id' missing or invalid" }

    # Determine final path
    export_path: Optional[Path] = None
    if export_path_str:
        export_path = Path(export_path_str)
        logger.info(f"[A3X Bridge Handler - Export] Using provided path: {export_path}")
        # Original code used print here
    else:
        logger.info(f"[A3X Bridge Handler - Export] No path provided, using default export location for {fragment_id}.")
        # Original code used print here
    
    # Call MemoryManager export
    try:
        success = memory_bank.export(fragment_id, export_path) # Pass None if path wasn't provided
        if success:
            final_path = export_path if export_path else (memory_bank.export_dir / f"{fragment_id}.a3xfrag")
            logger.info(f"[A3X Bridge Handler - Export] Successfully exported '{fragment_id}' to {final_path.resolve()}")
            # Original code used print here
            return {"status": "success", "fragment_id": fragment_id, "path": str(final_path.resolve())}
        else:
            logger.error(f"[A3X Bridge Handler - Export] MemoryBank.export failed for '{fragment_id}'. Check logs.")
            # Original code used print here
            return { "status": "error", "error": f"Export failed for fragment_id {fragment_id}" }
    except Exception as e:
        logger.exception(f"[A3X Bridge Handler - Export] Unexpected error exporting fragment {fragment_id}")
        # Original code used print here
        return { "status": "error", "error": f"Export failed for fragment_id {fragment_id}: {e}" } 