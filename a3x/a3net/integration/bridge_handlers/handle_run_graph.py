import logging
import torch
from typing import Dict, Optional, Any

# Assuming these are available in the environment
from a3x.a3net.core.memory_bank import MemoryBank
from a3x.a3net.core.cognitive_graph import CognitiveGraph

logger = logging.getLogger(__name__)

async def handle_run_graph(
    directive: Dict[str, Any],
    memory_bank: MemoryBank
) -> Optional[Dict[str, Any]]:
    """Handles the 'run_graph' directive logic."""

    graph_id = directive.get("graph_id")
    input_data = directive.get("input_data") # Could be text, list, dict, etc.

    if not graph_id:
        logger.error("[A3X Bridge Handler - RunGraph] 'graph_id' missing.")
        return { "status": "error", "message": "'graph_id' missing" }

    if input_data is None: # Allow empty input data if the graph supports it
        logger.warning("[A3X Bridge Handler - RunGraph] 'input_data' missing. Proceeding...")
        # return { "status": "error", "message": "'input_data' missing" }

    # --- Load Cognitive Graph ---
    try:
        logger.info(f"[A3X Bridge Handler - RunGraph] Loading graph '{graph_id}'...")
        cognitive_graph = memory_bank.load(graph_id)
        if not cognitive_graph:
            logger.error(f"[A3X Bridge Handler - RunGraph] Graph '{graph_id}' not found in memory bank.")
            return {"status": "error", "message": f"Graph '{graph_id}' not found."}
        if not isinstance(cognitive_graph, CognitiveGraph):
             logger.error(f"[A3X Bridge Handler - RunGraph] Object '{graph_id}' is not a CognitiveGraph, but a {type(cognitive_graph)}.")
             return {"status": "error", "message": f"Object '{graph_id}' is not a CognitiveGraph."}
        logger.info(f"[A3X Bridge Handler - RunGraph] Graph '{graph_id}' loaded successfully.")
    except Exception as e:
        logger.error(f"[A3X Bridge Handler - RunGraph] Error loading graph '{graph_id}': {e}", exc_info=True)
        return {"status": "error", "message": f"Error loading graph '{graph_id}': {e}"}

    # --- Execute Graph ---
    try:
        logger.info(f"[A3X Bridge Handler - RunGraph] Executing graph '{graph_id}' with input data: {str(input_data)[:100]}...") # Log truncated input
        # CognitiveGraph's __call__ method should handle the input data appropriately
        # It might need tensors or other formats depending on its implementation.
        # For now, we pass the raw input_data. The graph itself should validate/convert.
        output = await cognitive_graph(input_data) # Assuming CognitiveGraph.__call__ is async
        logger.info(f"[A3X Bridge Handler - RunGraph] Graph '{graph_id}' execution completed.")
        logger.debug(f"[A3X Bridge Handler - RunGraph] Output: {str(output)[:100]}") # Log truncated output
        return { "status": "success", "output": output }
    except Exception as e:
        logger.exception(f"[A3X Bridge Handler - RunGraph] Error during graph '{graph_id}' execution")
        return { "status": "error", "message": f"Graph execution failed: {e}" } 