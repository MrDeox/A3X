import logging
from typing import Dict, Optional, Any
from pathlib import Path
import json

# Assuming these are available in the environment where this handler is called
# Or they could be passed as arguments if that's cleaner
from a3x.a3net.core.memory_bank import MemoryBank 
from a3x.a3net.core.neural_language_fragment import NeuralLanguageFragment
from a3x.a3net.core.reflective_language_fragment import ReflectiveLanguageFragment
from a3x.a3net.trainer.dataset_builder import get_embedding_model
from a3x.a3net.core.context_store import ContextStore
from a3x.a3net.trainer.train_loop import train_fragment_cell

logger = logging.getLogger(__name__)

async def handle_train_fragment(
    directive: Dict[str, Any], 
    memory_bank: MemoryBank, 
    context_store: Optional[ContextStore] = None
) -> Optional[Dict[str, Any]]:
    """Handles the 'train_fragment' directive logic."""
    
    # --- Extract parameters ---
    fragment_id = directive.get("fragment_id")
    max_epochs = directive.get("max_epochs", 50) 
    context_id = directive.get("context_id") # Optional context for data
    target_accuracy = directive.get("target_accuracy")

    if not fragment_id or not isinstance(fragment_id, str):
        logger.error("[A3X Bridge Handler - Train] Error: 'fragment_id' missing or invalid.")
        return { "status": "error", "message": "'fragment_id' missing or invalid" }
        
    if not isinstance(max_epochs, int) or max_epochs <= 0:
        logger.warning(f"[A3X Bridge Handler - Train] Warning: Invalid 'max_epochs' ({max_epochs}). Using default 50.")
        max_epochs = 50

    # --- Load Fragment --- 
    try:
        logger.info(f"[A3X Bridge Handler - Train] Loading fragment '{fragment_id}'...")
        fragment_to_train = memory_bank.load(fragment_id)
        if not fragment_to_train:
            raise ValueError(f"Fragment '{fragment_id}' not found in Memory Bank.")
        if not isinstance(fragment_to_train, (NeuralLanguageFragment, ReflectiveLanguageFragment)):
            raise TypeError(f"Fragment '{fragment_id}' is not a trainable type (NeuralLanguageFragment or ReflectiveLanguageFragment).")
    except (ValueError, TypeError) as load_err:
        logger.error(f"[A3X Bridge Handler - Train] Error loading fragment '{fragment_id}': {load_err}")
        return {"status": "error", "message": f"Error loading fragment: {load_err}"}
    except Exception as e:
        logger.error(f"[A3X Bridge Handler - Train] Unexpected error loading fragment '{fragment_id}': {e}", exc_info=True)
        return {"status": "error", "message": f"Unexpected error loading fragment: {e}"}

    # --- Determine Task Name (PRIORITIZE DIRECTIVE) --- 
    task_name = directive.get("task_name")
    if task_name:
        logger.info(f"[A3X Bridge Handler - Train] Using task name '{task_name}' specified in the directive for fragment '{fragment_id}'.")
    else:
        task_name = getattr(fragment_to_train, 'associated_task_name', None)
        if task_name:
            logger.info(f"[A3X Bridge Handler - Train] Using associated task name '{task_name}' from fragment '{fragment_id}'.")
        else:
            task_name = f"task_for_{fragment_id}"
            logger.warning(f"[A3X Bridge Handler - Train] No task name in directive or association for '{fragment_id}'. Using default: '{task_name}'")

    # --- Call Fragment's Training Method --- 
    log_target_acc = f" target_accuracy={target_accuracy:.2f}" if target_accuracy is not None else ""
    logger.info(f"[A3X Bridge Handler - Train] Initiating training for '{fragment_id}' on task '{task_name}' for up to {max_epochs} epochs{log_target_acc}...")
    try:
        train_results = await fragment_to_train.train_on_task(
            task_name=task_name, 
            max_epochs=max_epochs, 
            target_accuracy=target_accuracy,
            context_store=context_store
        )
        
        if train_results.get("status") == "error":
             error_message = train_results.get("message", f"Training reported failure for fragment '{fragment_id}'. Check fragment logs.")
             logger.error(f"[A3X Bridge Handler - Train] {error_message}")
             return { "status": "error", "message": error_message, "fragment_id": fragment_id, **train_results }
        
        success_message = train_results.get("message", "Training completed (no specific message).")
        logger.info(f"[A3X Bridge Handler - Train] {success_message} for '{fragment_id}'. Results: {train_results}")
        
        # --- Save Updated Fragment State after successful/completed training --- 
        try:
            memory_bank.save(fragment_id, fragment_to_train) 
            logger.info(f"[A3X Bridge Handler - Train] Fragment {fragment_id} state saved/updated after training.")
        except Exception as e:
            logger.error(f"[A3X Bridge Handler - Train] Error: Failed to save fragment '{fragment_id}' after training: {e}", exc_info=True)
            # Add warning to results if saving failed after training succeeded
            train_results["status"] = "warning" # Downgrade status
            train_results["message"] = train_results.get("message", "Training completed") + f" BUT failed to save fragment state: {e}"

        return train_results # Return results whether save succeeded or failed (if training was okay)
            
    except Exception as e:
        logger.error(f"[A3X Bridge Handler - Train] Error: Exception during fragment.train_on_task call: {e}", exc_info=True)
        return { "status": "error", "message": f"Training loop failed: {e}", "fragment_id": fragment_id } 