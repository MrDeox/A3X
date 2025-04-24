import logging
from typing import Dict, Optional, Any
from pathlib import Path

# Assuming these are available in the environment
from a3x.core.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

async def handle_import_fragment(
    directive: Dict[str, Any],
    memory_manager: MemoryManager
) -> Optional[Dict[str, Any]]:
    """Handles the 'import_fragment' directive logic."""

    import_path_str = directive.get("path")

    if not import_path_str or not isinstance(import_path_str, str):
        logger.error("[A3X Bridge Handler - Import] 'path' (string) missing or invalid.")
        # Original code used print here
        return { "status": "error", "error": "Import failed: 'path' missing or invalid" }
    
    import_path = Path(import_path_str)
    logger.info(f"[A3X Bridge Handler - Import] Attempting import from: {import_path}")
    # Original code used print here

    # Call MemoryManager import
    try:
        success = memory_bank.import_a3xfrag(import_path)
        if success:
            logger.info(f"[A3X Bridge Handler - Import] Successfully imported fragment from {import_path}")
            # Original code used print here
            # We don't necessarily know the fragment_id here unless we parse metadata again
            # Return path for confirmation
            return {"status": "success", "path": str(import_path)}
        else:
            logger.error(f"[A3X Bridge Handler - Import] MemoryBank.import_a3xfrag failed for {import_path}. Check logs.")
            # Original code used print here
            return { "status": "error", "error": f"Import failed from path {import_path}" }
    except Exception as e:
        logger.exception(f"[A3X Bridge Handler - Import] Unexpected error importing fragment from {import_path}")
        # Original code used print here
        return { "status": "error", "error": f"Import failed from path {import_path}: {e}" } 