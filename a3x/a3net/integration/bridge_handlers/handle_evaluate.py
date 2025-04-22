import logging
import time
from typing import Dict, Optional, Any

# Assuming these are available in the environment
from a3x.a3net.core.memory_bank import MemoryBank
from a3x.a3net.core.context_store import ContextStore

logger = logging.getLogger(__name__)

async def handle_evaluate_fragment(
    directive: Dict[str, Any],
    memory_bank: MemoryBank,
    context_store: Optional[ContextStore]
) -> Optional[Dict[str, Any]]:
    """Handles the 'avaliar_fragmento' directive logic."""

    fragment_id = directive.get("fragment_id")
    task_name = directive.get("task_name") # Task name specifies the dataset
    test_split_ratio = directive.get("test_split", 0.2) # Optional split ratio

    if not fragment_id or not task_name:
        logger.error("[A3X Bridge Handler - Evaluate] 'fragment_id' or 'task_name' missing.")
        return { "status": "error", "message": "'fragment_id' or 'task_name' missing for avaliar_fragmento" }
    
    if not context_store:
         logger.error("[A3X Bridge Handler - Evaluate] ContextStore instance not provided.")
         return { "status": "error", "message": "ContextStore not available for saving evaluation results." }

    # --- Load Fragment ---
    logger.info(f"[A3X Bridge Handler - Evaluate] Loading fragment '{fragment_id}' for evaluation on task '{task_name}'.")
    fragment = memory_bank.load(fragment_id)

    if fragment is None:
        logger.error(f"[A3X Bridge Handler - Evaluate] Fragment '{fragment_id}' not found.")
        return { "status": "error", "message": f"Fragment '{fragment_id}' not found for evaluation." }

    # --- Check if Fragment Supports Evaluation ---
    if not hasattr(fragment, 'evaluate') or not callable(getattr(fragment, 'evaluate')):
         logger.error(f"[A3X Bridge Handler - Evaluate] Fragment '{fragment_id}' ({type(fragment).__name__}) does not support the 'evaluate' method.")
         return { "status": "error", "message": f"Fragment type {type(fragment).__name__} does not support evaluation." }

    # --- Run Evaluation ---
    logger.info(f"[A3X Bridge Handler - Evaluate] Calling evaluate() on fragment '{fragment_id}' for task '{task_name}'.")
    try:
        evaluation_results = await fragment.evaluate(task_name=task_name, test_split_ratio=test_split_ratio)
        logger.info(f"[A3X Bridge Handler - Evaluate] Evaluation results for '{fragment_id}': {evaluation_results}")
        
        eval_status = evaluation_results.get("status")
        
        # --- Store Results in ContextStore (if successful/warning) ---
        if eval_status in ["success", "warning_small_dataset"]:
            try:
                timestamp_ms = int(time.time() * 1000)
                score_key = f"evaluation_score:{fragment_id}:{task_name}:{timestamp_ms}"
                
                save_data = evaluation_results.copy()
                save_data["timestamp"] = timestamp_ms
                
                await context_store.set(score_key, save_data)
                logger.info(f"[A3X Bridge Handler - Evaluate] Saved evaluation results for '{fragment_id}' to ContextStore with key '{score_key}'.")
            except Exception as cs_err:
                 logger.error(f"[A3X Bridge Handler - Evaluate] Failed to save evaluation results for '{fragment_id}' to ContextStore: {cs_err}", exc_info=True)
                 evaluation_results["message"] = evaluation_results.get("message", "") + f" [Error saving score: {cs_err}]"
                 # Optionally change status to warning if saving failed
                 if eval_status == "success": 
                     evaluation_results["status"] = "warning_save_failed"

        return evaluation_results # Return the original (potentially updated) results

    except Exception as e:
        logger.error(f"[A3X Bridge Handler - Evaluate] Unexpected error during evaluation call for '{fragment_id}': {e}", exc_info=True)
        return { 
            "status": "error", 
            "fragment_id": fragment_id,
            "task_name": task_name,
            "message": f"Unexpected error during fragment evaluation: {e}" 
        } 