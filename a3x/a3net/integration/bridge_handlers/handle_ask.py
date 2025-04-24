import logging
from typing import Dict, Optional, Any
import torch
import uuid
import datetime as dt

# Assuming these are available in the environment
from a3x.core.memory.memory_manager import MemoryManager
from a3x.a3net.core.context_store import ContextStore
from a3x.a3net.trainer.dataset_builder import get_embedding_model
from a3x.a3net.core.neural_language_fragment import NeuralLanguageFragment
from a3x.a3net.core.reflective_language_fragment import ReflectiveLanguageFragment

logger = logging.getLogger(__name__)

async def handle_ask(
    directive: Dict[str, Any],
    memory_manager: MemoryManager,
    context_store: Optional[ContextStore]
) -> Optional[Dict[str, Any]]:
    """Handles the 'ask' directive logic."""

    fragment_id = directive.get("fragment_id")
    input_list = directive.get("input_list") # Prefer list if provided
    text_input = directive.get("text_input") # Text input from "sobre"
    
    if not fragment_id or not isinstance(fragment_id, str):
        logger.error("[A3X Bridge Handler - Ask] 'fragment_id' missing or invalid.")
        return { "status": "error", "message": "'fragment_id' missing or invalid" }

    # Validate that *at least one* input type is provided
    if input_list is None and text_input is None:
        logger.error("[A3X Bridge Handler - Ask] 'input_list' or 'text_input' missing.")
        return { "status": "error", "message": "Input (list or text) missing for ask directive." }

    # Log parsed parameters
    log_input_type = "list" if input_list is not None else "text"
    log_input_value = input_list if input_list is not None else f'"{text_input[:50]}..."'
    logger.info(f"[A3X Bridge Handler - Ask] Parsed: fragment_id={fragment_id}, input_type={log_input_type}, input_value={log_input_value}")

    # --- Get Input Tensor --- 
    input_tensor: Optional[torch.Tensor] = None
    try:
        if input_list is not None:
            if not isinstance(input_list, list):
                raise TypeError("'input_list' must be a list.")
            logger.debug("[A3X Bridge Handler - Ask] Converting input_list to tensor.")
            input_tensor = torch.tensor(input_list, dtype=torch.float32)
        elif text_input is not None:
            if not isinstance(text_input, str):
                raise TypeError("'text_input' must be a string.")
            logger.debug(f"[A3X Bridge Handler - Ask] Getting embedding for text_input: '{text_input[:100]}...'" )
            embedding_model = get_embedding_model()
            if not embedding_model:
                raise RuntimeError("Embedding model could not be loaded.")
            # Ensure embedding model returns tensor directly
            input_tensor = embedding_model.encode(text_input, convert_to_tensor=True)
            if input_tensor is None or not isinstance(input_tensor, torch.Tensor):
                raise RuntimeError("Embedding model failed to return a valid tensor.")
            logger.debug(f"[A3X Bridge Handler - Ask] Obtained embedding tensor shape: {input_tensor.shape}")
        else:
            raise ValueError("No valid input (list or text) found after validation.")

        # Ensure tensor has batch dimension [1, dim]
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0) 
        elif len(input_tensor.shape) != 2 or input_tensor.shape[0] != 1:
             raise ValueError(f"Input tensor has unexpected shape {input_tensor.shape}. Expected [dim] or [1, dim].")
        logger.info(f"[A3X Bridge Handler - Ask] Prepared input tensor with final shape: {input_tensor.shape}")

    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(f"[A3X Bridge Handler - Ask] Error preparing input tensor: {e}", exc_info=True)
        return { "status": "error", "message": f"Input processing failed: {e}" }
    except Exception as e: # Catch other unexpected errors
        logger.error(f"[A3X Bridge Handler - Ask] Unexpected error preparing input tensor: {e}", exc_info=True)
        return { "status": "error", "message": f"Unexpected input processing error: {e}" }
        
    # --- Load Fragment --- 
    logger.info(f"[A3X Bridge Handler - Ask] Loading fragment '{fragment_id}' from MemoryBank...")
    fragment = memory_bank.load(fragment_id)

    if fragment is None:
        logger.error(f"[A3X Bridge Handler - Ask] Fragment '{fragment_id}' not found in MemoryBank.")
        return { "status": "error", "message": f"Fragment '{fragment_id}' not found" }
    
    # --- Check Fragment Type and Get Prediction --- 
    try:
        logger.info(f"[A3X Bridge Handler - Ask] Running prediction with input shape {input_tensor.shape}...")
        # Handle batch input if necessary (predicting only first sample for now)
        if input_tensor.shape[0] != 1:
             logger.warning(f"[A3X Bridge Handler - Ask] Received batch input (size {input_tensor.shape[0]}). Predicting only for the first sample.")
             input_tensor = input_tensor[0].unsqueeze(0)

        # Initialize result storage
        prediction_result_dict: Optional[Dict[str, Any]] = None
        return_payload: Optional[Dict[str, Any]] = None
        
        # Call the appropriate predict method
        if isinstance(fragment, (NeuralLanguageFragment, ReflectiveLanguageFragment)):
             prediction_result_dict = fragment.predict(input_tensor)
        else:
             logger.error(f"[A3X Bridge Handler - Ask] Fragment '{fragment_id}' type ({type(fragment).__name__}) does not support 'ask'.")
             return { "status": "error", "message": f"Fragment type {type(fragment).__name__} incompatible with 'ask'" }

        # Validate the prediction result dictionary structure
        if not prediction_result_dict or 'output' not in prediction_result_dict or 'confidence' not in prediction_result_dict:
             logger.error(f"[A3X Bridge Handler - Ask] Fragment '{fragment_id}' predict() returned unexpected format: {prediction_result_dict}")
             return { "status": "error", "message": "Invalid prediction format from fragment" }

        # Format the return payload based on fragment type
        logger.info(f"[A3X Bridge Handler - Ask] Prediction complete. Output: {prediction_result_dict['output']}, Confidence: {prediction_result_dict['confidence']:.4f}")
        return_payload = {
            "status": "success", 
            "fragment_id": fragment_id, 
            "output": prediction_result_dict["output"],
            "confidence": prediction_result_dict["confidence"]
        }
        # Add explanation if it's a reflective fragment
        if isinstance(fragment, ReflectiveLanguageFragment) and 'explanation' in prediction_result_dict:
            return_payload["explanation"] = prediction_result_dict["explanation"]
            logger.info(f"[A3X Bridge Handler - Ask] Explanation: {prediction_result_dict['explanation']}")
        
        # --- Store result in ContextStore --- 
        if context_store:
            try:
                task_id = uuid.uuid4().hex
                timestamp_iso = dt.datetime.now(dt.timezone.utc).isoformat()
                
                # Determine input type and value from original directive
                input_type = 'text' if 'text_input' in directive else 'list'
                input_value = directive.get('text_input') if input_type == 'text' else directive.get('input_list')
                
                ask_context_data = {
                    "task_id": task_id,
                    "fragment_id": fragment_id,
                    "input_type": input_type,
                    "input_value": input_value,
                    "output": return_payload['output'],
                    "confidence": return_payload['confidence'],
                    "timestamp": timestamp_iso
                }
                if "explanation" in return_payload:
                     ask_context_data["explanation"] = return_payload["explanation"]

                # Store full record
                context_key_unique = f"resposta:{task_id}:{fragment_id}"
                await context_store.set(context_key_unique, ask_context_data)
                logger.debug(f"[A3X Bridge Handler - Ask] Stored full response context to key: {context_key_unique}")

                # Store latest result
                context_key_latest = f"last_ask_result:{fragment_id}"
                await context_store.set(context_key_latest, ask_context_data)
                logger.debug(f"[A3X Bridge Handler - Ask] Stored latest response context to key: {context_key_latest}")
                
                # Mark feedback as pending
                context_key_feedback = f"feedback_pendente:{fragment_id}"
                await context_store.set(context_key_feedback, task_id)
                logger.debug(f"[A3X Bridge Handler - Ask] Marked feedback pending for task {task_id} under key: {context_key_feedback}")

            except Exception as cs_e:
                logger.error(f"[A3X Bridge Handler - Ask] Failed to store ask result in ContextStore: {cs_e}", exc_info=True)
                # Update message but keep status success as prediction worked
                return_payload["message"] = return_payload.get("message", "Prediction successful") + f" [Warning: Failed to save context: {cs_e}]"
                return_payload["status"] = "warning" # Or keep success? Let's use warning.
        
        return return_payload # Return the formatted payload

    except Exception as e:
        logger.error(f"[A3X Bridge Handler - Ask] Error during prediction: {e}", exc_info=True)
        return { "status": "error", "message": f"Prediction failed: {e}" } 