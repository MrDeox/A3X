from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import re # Import regex for cleaning goal string
import logging # Added logging
from pathlib import Path # <<< Add Path >>>
import os
import shlex # For parsing command-line like strings
import ast   # For safely evaluating literals
import time # <<< Add time import for timestamp >>>
import asyncio # Added for async handling
import json # Import json for loading dataset

# Corrected A³Net imports (relative within a3x)
from ..trainer.dataset_builder import build_dataset_from_context, get_embedding_model
from ..core.fragment_cell import FragmentCell
from ..trainer.train_loop import train_fragment_cell # <<< Correct function name
from ..core.memory_bank import MemoryBank # <<< Import MemoryBank >>>
from ..core.cognitive_graph import CognitiveGraph # <<< Import CognitiveGraph >>>
from ..core.neural_language_fragment import NeuralLanguageFragment # <<< Import NeuralLanguageFragment >>>
from ..core.reflective_language_fragment import ReflectiveLanguageFragment # <<< Import ReflectiveLanguageFragment >>>
from ..core.professor_llm_fragment import ProfessorLLMFragment # <<< Import ProfessorLLMFragment >>>
from ..core.context_store import ContextStore # <<< Ensure ContextStore is imported >>>
# <<< REMOVE Import from run.py >>>
# from ..run import post_message_handler # Assuming it's accessible here 

# Configure logging
logger = logging.getLogger(__name__)

# --- Module-Level Memory Bank Instance ---
# Ensure correct path relative to execution context if needed, or use absolute/configurable path
# Using a relative path like "a3net_memory" assumes execution from project root.
MEMORY_BANK = MemoryBank(save_dir="a3x/a3net/a3net_memory", export_dir="a3x/a3net/a3x_repo") 
# Note: This creates one instance when the module is first imported.

# --- Context Store Instance (Needs to be passed or accessible) ---
# We need access to the context store instance used in run.py
# Option 1: Pass it as an argument to handle_directive (preferred)
# Option 2: Make it a globally accessible object (less ideal)
# For now, we assume it will be passed via fragment_instances or a dedicated arg

# --- Module-Level State for Last Ask --- 
LAST_ASK_RESULT: Optional[Dict[str, Any]] = None

# --- Example Registration Function --- 
async def registrar_exemplo_de_aprendizado(context_store: ContextStore, task_name: str, input_data: str, label_text: str):
    # ... (existing function, seems okay or replaced by data_logger) ...
    pass
# ---------------------------------

# <<< Make handle_directive async >>>
# <<< Add context_store and post_message_handler parameters >>>
async def handle_directive(
    directive: Dict, 
    fragment_instances: Optional[Dict[str, Any]] = None,
    context_store: Optional[ContextStore] = None, # <<< Added context_store param
    post_message_handler: Optional[Callable[..., Awaitable[None]]] = None # <<< Added post_message_handler param
) -> Optional[Dict[str, Any]]:
    """Handles directives received from the symbolic A³X system.
    
    Supports 'train_fragment', 'run_graph', 'ask', 'export_fragment', 'import_fragment', 
    'reflect_fragment', 'conditional_directive', 'avaliar_fragmento', 'comparar_desempenho' types.
    
    Args:
        directive: The dictionary representing the A3L command.
        fragment_instances: Optional dictionary of currently active fragment instances.
        context_store: Optional instance of the ContextStore for saving evaluation results.
        post_message_handler: Optional async function to post messages back to the queue. <<< Added

    Returns:
        An optional dictionary containing the status and results of the directive execution.
    """
    # Access and potentially modify the global state
    global LAST_ASK_RESULT
    
    directive_type = directive.get("type")
    goal = directive.get('goal', None) 
    logger.info(f"[A3X Bridge] <<< Handling directive type: '{directive_type}', goal: {goal or 'Not specified'}") # Use logger

    # =======================================
    # === Handle 'train_fragment' type ===
    # =======================================
    if directive_type == "train_fragment":
        # --- Extract parameters ---
        fragment_id = directive.get("fragment_id")
        epochs = directive.get("epochs")
        context_id = directive.get("context_id") # Optional context for data

        if not fragment_id or not isinstance(fragment_id, str):
            logger.error("[A3X Bridge] Error (train): 'fragment_id' missing or invalid.")
            return { "status": "error", "message": "'fragment_id' missing or invalid" }
        if not epochs or not isinstance(epochs, int) or epochs <= 0:
            logger.error("[A3X Bridge] Error (train): 'epochs' missing or invalid (must be positive integer).")
            return { "status": "error", "message": "'epochs' missing or invalid" }

        # --- Load or Create Fragment ---
        fragment = MEMORY_BANK.load(fragment_id)
        if fragment:
            logger.info(f"[A3X Bridge] (train) Loaded existing fragment '{fragment_id}' for retraining.")
            # Ensure it's a trainable type (or adapt logic if needed)
            if not isinstance(fragment, (NeuralLanguageFragment, ReflectiveLanguageFragment)):
                 logger.error(f"[A3X Bridge] Error (train): Fragment '{fragment_id}' type ({type(fragment).__name__}) is not trainable.")
                 return { "status": "error", "message": f"Fragment type {type(fragment).__name__} not trainable" }
        else:
            logger.warning(f"[A3X Bridge] (train) Fragment '{fragment_id}' not found. Creating default NeuralLanguageFragment.")
            try:
                # --- Determine Input Dimension (Consistent) ---
                embedding_model = get_embedding_model()
                actual_input_dim = 128 # Fallback
                if embedding_model:
                    actual_input_dim = embedding_model.get_sentence_embedding_dimension()
                    logger.info(f"[A3X Bridge] (train-create) Using input dimension {actual_input_dim} from embedding model.")
                else:
                    logger.warning(f"[A3X Bridge] (train-create) Could not get embedding model dimension, falling back to default input_dim={actual_input_dim}")

                # --- Try to Infer Output Layer Params from Dataset --- 
                default_hidden_dim = 64 
                inferred_num_classes = None
                inferred_id_to_label = None
                associated_task_name = f"task_for_{fragment_id}" # Default task name
                dataset_path = Path(f"a3x/a3net/datasets/{fragment_id}.jsonl") # Assume task name maps to fragment_id for now
                
                logger.info(f"[A3X Bridge] (train-create) Trying to infer output params directly from dataset: {dataset_path}")
                try:
                    if dataset_path.is_file():
                        unique_labels = set()
                        with open(dataset_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    data = json.loads(line.strip())
                                    if isinstance(data, dict) and 'label' in data and data['label'] is not None:
                                        label_val = data['label']
                                        label_str = json.dumps(label_val, sort_keys=True) if not isinstance(label_val, str) else label_val
                                        unique_labels.add(label_str)
                                except json.JSONDecodeError:
                                    pass # Ignore malformed lines for inference
                        
                        if unique_labels:
                            sorted_labels = sorted(list(unique_labels))
                            id_to_label_map = {i: (json.loads(lbl) if lbl.startswith( ('{', '[') ) else lbl) for i, lbl in enumerate(sorted_labels)}
                            inferred_num_classes = len(sorted_labels)
                            inferred_id_to_label = id_to_label_map
                            logger.info(f"[A3X Bridge] (train-create) Inferred {inferred_num_classes} classes and label map directly from '{dataset_path}'.")
                            associated_task_name = fragment_id 
                            logger.info(f"[A3X Bridge] (train-create) Setting associated_task_name to inferred task: '{associated_task_name}'")
                        else:
                            logger.warning(f"[A3X Bridge] (train-create) Dataset '{dataset_path}' found but contains no valid labels. Using defaults.")
                    else:
                         logger.warning(f"[A3X Bridge] (train-create) Dataset '{dataset_path}' not found. Using defaults for output layer.")
                except Exception as ds_err:
                    logger.error(f"[A3X Bridge] (train-create) Error inferring params directly from '{dataset_path}': {ds_err}. Using defaults.", exc_info=True)
                
                # --- Set final parameters (use inferred or defaults) ---
                final_num_classes = inferred_num_classes if inferred_num_classes is not None else 3
                final_id_to_label = inferred_id_to_label if inferred_id_to_label is not None else {i: f"AUTO_CLASS_{i}" for i in range(final_num_classes)}
                if inferred_num_classes is None:
                     logger.warning(f"[A3X Bridge] (train-create) Using default num_classes={final_num_classes} and default label map.")

                # --- Create the Fragment --- 
                fragment = NeuralLanguageFragment(
                    fragment_id=fragment_id,
                    description=f"Implicitly created neural fragment '{fragment_id}' for training", # Updated description
                    input_dim=actual_input_dim, 
                    hidden_dim=default_hidden_dim,
                    num_classes=final_num_classes, # Use inferred or default
                    id_to_label=final_id_to_label, # Use inferred or default
                    associated_task_name=associated_task_name # Use inferred or default
                )
                MEMORY_BANK.save(fragment_id, fragment) 
                logger.info(f"[A3X Bridge] Created and saved initial state for new fragment '{fragment_id}' with input_dim={actual_input_dim}, num_classes={final_num_classes}.")
            except Exception as creation_err:
                 logger.error(f"[A3X Bridge] Failed to automatically create fragment '{fragment_id}': {creation_err}", exc_info=True)
                 return { "status": "error", "message": f"Failed to create fragment '{fragment_id}': {creation_err}" }

        # Check if fragment is trainable and has the new method
        if not isinstance(fragment, (NeuralLanguageFragment, ReflectiveLanguageFragment)) or not hasattr(fragment, 'train_on_task'):
             logger.error(f"[A3X Bridge] Error (train): Fragment '{fragment_id}' type ({type(fragment).__name__}) is not trainable via train_on_task.")
             return { "status": "error", "message": f"Fragment type {type(fragment).__name__} not trainable via train_on_task" }

        # --- Determine Task Name --- 
        task_name = getattr(fragment, 'associated_task_name', None)
        if not task_name:
            task_name = f"task_for_{fragment_id}"
            logger.warning(f"[A3X Bridge] (train) Fragment '{fragment_id}' has no 'associated_task_name'. Using default: '{task_name}'")
        else:
            logger.info(f"[A3X Bridge] (train) Using associated task name '{task_name}' for fragment '{fragment_id}'.")
        
        # --- Call Fragment's Training Method --- 
        logger.info(f"[A3X Bridge] (train) Initiating training for '{fragment_id}' on task '{task_name}' for {epochs} epochs...")
        try:
            success = await fragment.train_on_task(task_name=task_name, epochs=epochs)
            if not success:
                 logger.error(f"[A3X Bridge] (train) Training method for fragment '{fragment_id}' reported failure.")
                 return { "status": "error", "message": f"Training failed for fragment '{fragment_id}'. Check fragment logs.", "fragment_id": fragment_id }
            logger.info(f"[A3X Bridge] (train) Training reported successful for '{fragment_id}'.")
        except Exception as e:
            logger.error(f"[A3X Bridge] Error (train): Exception during fragment.train_on_task call: {e}", exc_info=True)
            return { "status": "error", "message": f"Training loop failed: {e}", "fragment_id": fragment_id }

        # --- Save Updated Fragment State --- 
        try:
            MEMORY_BANK.save(fragment_id, fragment) 
            logger.info(f"[A3X Bridge] (train) Fragment {fragment_id} state saved/updated after training.")
            return { "status": "success", "message": "Training complete, fragment saved/updated", "fragment_id": fragment_id }
        except Exception as e:
            logger.error(f"[A3X Bridge] Error (train): Failed to save fragment '{fragment_id}' after training: {e}", exc_info=True)
            return { "status": "error", "message": f"Failed to save fragment after training: {e}", "fragment_id": fragment_id }

    # =======================================
    # === Handle 'create_fragment' type (NEW) ===
    # =======================================
    elif directive_type == "create_fragment":
        fragment_id = directive.get("fragment_id")
        fragment_type = directive.get("fragment_type", "neural").lower() # Default to neural
        params = directive.get("params", {}) # Optional parameters like dim, classes etc.

        if not fragment_id:
            logger.error("[A3X Bridge] Error (create): 'fragment_id' missing.")
            return { "status": "error", "message": "'fragment_id' missing for create_fragment" }

        # Check if fragment already exists
        if MEMORY_BANK.exists(fragment_id):
             logger.warning(f"[A3X Bridge] (create) Fragment '{fragment_id}' already exists. Skipping creation.")
             return { "status": "skipped", "message": f"Fragment '{fragment_id}' already exists.", "fragment_id": fragment_id }

        logger.info(f"[A3X Bridge] (create) Creating fragment '{fragment_id}' of type '{fragment_type}' with params: {params}")

        try:
            fragment_instance = None
            if fragment_type == "neural":
                # Extract relevant parameters for NeuralLanguageFragment
                provided_input_dim = params.get("input_dim") # Get provided dim
                hidden_dim = params.get("hidden_dim", 64) # Default
                num_classes = params.get("num_classes")
                description = params.get("description", f"Neural fragment '{fragment_id}'")
                task_name = params.get("task_name") # Optional association
                id_to_label = params.get("id_to_label") # Optional label map

                # --- Determine CORRECT input_dim from embedding model --- 
                actual_input_dim = None
                try:
                    embedding_model = get_embedding_model()
                    if embedding_model:
                        actual_input_dim = embedding_model.get_sentence_embedding_dimension()
                        logger.info(f"[A3X Bridge] (create) Determined actual input_dim={actual_input_dim} from embedding model.")
                        # Warn if provided dim differs
                        if provided_input_dim is not None and provided_input_dim != actual_input_dim:
                            logger.warning(f"[A3X Bridge] (create) Provided input_dim ({provided_input_dim}) for fragment '{fragment_id}' differs from embedding model dimension ({actual_input_dim}). USING {actual_input_dim}.")
                    else:
                        raise ValueError("Failed to load embedding model.")
                except Exception as emb_err:
                    logger.error(f"[A3X Bridge] Error (create): Could not determine input_dim from embedding model: {emb_err}")
                    return { "status": "error", "message": f"Cannot create neural fragment: failed to get embedding dimension ({emb_err})." }
                
                # Determine num_classes (important for output layer)
                if num_classes is None:
                    if id_to_label and isinstance(id_to_label, dict):
                        num_classes = len(id_to_label)
                        logger.info(f"[A3X Bridge] (create) Inferred num_classes={num_classes} from provided id_to_label map.")
                    else:
                         # Default to 3 if not specified and cannot be inferred
                         num_classes = 3 
                         logger.warning(f"[A3X Bridge] (create) 'num_classes' not specified and cannot infer from id_to_label. Defaulting to {num_classes}.")
                         # Use default label map only if num_classes ends up being 3
                         if num_classes == 3 and id_to_label is None:
                              id_to_label = NeuralLanguageFragment.DEFAULT_ID_TO_LABEL
                              logger.info("(create) Using default id_to_label map.")

                fragment_instance = NeuralLanguageFragment(
                    fragment_id=fragment_id,
                    description=description,
                    input_dim=actual_input_dim, # <<< ALWAYS use actual_input_dim
                    hidden_dim=hidden_dim,
                    num_classes=num_classes,
                    id_to_label=id_to_label,
                    associated_task_name=task_name
                )

            elif fragment_type == "reflective":
                # Similar parameter extraction for ReflectiveLanguageFragment
                provided_input_dim = params.get("input_dim") # Get provided dim
                hidden_dim = params.get("hidden_dim", 64)
                num_classes = params.get("num_classes")
                description = params.get("description", f"Reflective fragment '{fragment_id}'")
                task_name = params.get("task_name")
                id_to_label = params.get("id_to_label")
                # Additional reflective params?
                reflection_threshold = params.get("reflection_threshold", 0.7)

                # --- Determine CORRECT input_dim from embedding model --- 
                actual_input_dim = None
                try:
                    embedding_model = get_embedding_model()
                    if embedding_model:
                        actual_input_dim = embedding_model.get_sentence_embedding_dimension()
                        logger.info(f"[A3X Bridge] (create) Determined actual input_dim={actual_input_dim} from embedding model for reflective fragment.")
                        # Warn if provided dim differs
                        if provided_input_dim is not None and provided_input_dim != actual_input_dim:
                            logger.warning(f"[A3X Bridge] (create) Provided input_dim ({provided_input_dim}) for reflective fragment '{fragment_id}' differs from embedding model dimension ({actual_input_dim}). USING {actual_input_dim}.")
                    else:
                        raise ValueError("Failed to load embedding model.")
                except Exception as emb_err:
                    logger.error(f"[A3X Bridge] Error (create): Could not determine input_dim for reflective fragment: {emb_err}")
                    return { "status": "error", "message": f"Cannot create reflective fragment: failed to get embedding dimension ({emb_err})." }

                if num_classes is None:
                    if id_to_label and isinstance(id_to_label, dict): num_classes = len(id_to_label)
                    else: num_classes = 3; logger.warning("(create) Defaulting num_classes to 3 for reflective fragment.")
                
                fragment_instance = ReflectiveLanguageFragment(
                     fragment_id=fragment_id,
                     description=description,
                     input_dim=actual_input_dim, # <<< ALWAYS use actual_input_dim
                     hidden_dim=hidden_dim,
                     num_classes=num_classes,
                     id_to_label=id_to_label,
                     associated_task_name=task_name,
                     reflection_threshold=reflection_threshold
                )
            
            # Add other fragment types here (e.g., 'professor', 'supervisor') if needed

            else:
                 logger.error(f"[A3X Bridge] Error (create): Unsupported fragment type '{fragment_type}'.")
                 return { "status": "error", "message": f"Unsupported fragment type: {fragment_type}" }
            
            # --- Save the newly created fragment --- 
            logger.info(f"[A3X Bridge] (create) Attempting to save fragment '{fragment_id}' instance: {fragment_instance}") # <<< ADD LOGGING
            MEMORY_BANK.save(fragment_id, fragment_instance)
            logger.info(f"[A3X Bridge] (create) Successfully created and saved fragment '{fragment_id}' type '{fragment_type}'.") # <<< LOGGING CONFIRMS SAVE
            return { "status": "success", "message": f"Fragment '{fragment_id}' created.", "fragment_id": fragment_id }

        except Exception as e:
             logger.error(f"[A3X Bridge] (create) Error creating fragment '{fragment_id}': {e}", exc_info=True)
             # Clean up potentially inconsistent state? (e.g., remove from cache if added)
             MEMORY_BANK.delete(fragment_id, ignore_errors=True) # Attempt to remove if creation failed mid-way
             return { "status": "error", "message": f"Failed to create fragment '{fragment_id}': {e}" }

    # =======================================
    # === Handle 'run_graph' type ========
    # =======================================
    elif directive_type == "run_graph":
        # --- Extract parameters ---
        fragment_ids = directive.get("fragment_ids")
        input_data = directive.get("input") # Expecting List[float] or List[List[float]]

        if not fragment_ids or not isinstance(fragment_ids, list):
            print("[A3X Bridge] Error (run): 'fragment_ids' (list) missing or invalid.")
            logger.error("'fragment_ids' missing or invalid for run_graph directive.")
            return { "status": "error", "message": "'fragment_ids' missing or invalid" }
        
        if not input_data or not isinstance(input_data, list):
            print("[A3X Bridge] Error (run): 'input' (list) missing or invalid.")
            logger.error("'input' missing or invalid for run_graph directive.")
            return { "status": "error", "message": "'input' missing or invalid" }
            
        print(f"[A3X Bridge] (run) Parsed parameters: fragment_ids={fragment_ids}")

        # --- Convert Input to Tensor ---
        try:
            # Assuming input_data is a flat list for a single sample, or list of lists for batch
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            # Reshape if it's a flat list representing a single sample
            if len(input_tensor.shape) == 1:
                input_tensor = input_tensor.unsqueeze(0) # Add batch dimension -> [1, input_dim]
            print(f"[A3X Bridge] (run) Converted input data to tensor with shape: {input_tensor.shape}")
        except Exception as e:
            print(f"[A3X Bridge] Error (run) converting input data to tensor: {e}")
            logger.exception("Error converting input data to tensor")
            return { "status": "error", "message": f"Input data conversion failed: {e}" }

        # --- Create Cognitive Graph ---
        try:
            print(f"[A3X Bridge] (run) Instantiating CognitiveGraph with {len(fragment_ids)} IDs...")
            cognitive_graph = CognitiveGraph(memory_bank=MEMORY_BANK, fragment_ids=fragment_ids)
            if len(cognitive_graph.fragments) == 0:
                 print("[A3X Bridge] Error (run): CognitiveGraph loaded 0 fragments.")
                 logger.error("CognitiveGraph loaded 0 fragments.")
                 # Return error if no fragments could be loaded for the graph
                 return { "status": "error", "message": "Graph creation failed: No fragments loaded" }
            print(f"[A3X Bridge] (run) CognitiveGraph created successfully.")
        except Exception as e:
            print(f"[A3X Bridge] Error (run) creating CognitiveGraph: {e}")
            logger.exception("Error creating CognitiveGraph")
            return { "status": "error", "message": f"Graph creation failed: {e}" }

        # --- Run Inference ---
        try:
            print(f"[A3X Bridge] (run) Running inference with input shape {input_tensor.shape}...")
            output_tensor = cognitive_graph(input_tensor)
            output_list = output_tensor.tolist() # Convert to list for return
            print(f"[A3X Bridge] (run) Inference completed. Output shape: {output_tensor.shape}")
            print(f"[A3X Bridge] (run) Output: {output_list}")
            return { "status": "success", "output": output_list }
        except Exception as e:
            print(f"[A3X Bridge] Error (run) during graph inference: {e}")
            logger.exception("Error during graph inference")
            return { "status": "error", "message": f"Inference failed: {e}" }
            
    # =======================================
    # === Handle 'ask' type ==============
    # =======================================
    elif directive_type == "ask":
        # --- Extract parameters ---
        fragment_id = directive.get("fragment_id")
        input_data = directive.get("input") # Expecting List[float]
        
        if not fragment_id or not isinstance(fragment_id, str):
            print("[A3X Bridge] Error (ask): 'fragment_id' (string) missing or invalid.")
            logger.error("'fragment_id' missing or invalid for ask directive.")
            return { "status": "error", "message": "'fragment_id' missing or invalid" }

        if not input_data or not isinstance(input_data, list):
            print("[A3X Bridge] Error (ask): 'input' (list) missing or invalid.")
            logger.error("'input' missing or invalid for ask directive.")
            return { "status": "error", "message": "'input' missing or invalid" }

        print(f"[A3X Bridge] (ask) Parsed parameters: fragment_id={fragment_id}")

        # --- Convert Input to Tensor ---
        try:
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            if len(input_tensor.shape) == 1:
                input_tensor = input_tensor.unsqueeze(0) 
            elif len(input_tensor.shape) > 2:
                 raise ValueError("Input data must be a single vector or a batch of vectors.")
            print(f"[A3X Bridge] (ask) Converted input data to tensor with shape: {input_tensor.shape}")
        except Exception as e:
            print(f"[A3X Bridge] Error (ask) converting input data to tensor: {e}")
            logger.exception("Error converting input data to tensor for ask directive")
            return { "status": "error", "message": f"Input data conversion failed: {e}" }
            
        # --- Load Fragment ---
        print(f"[A3X Bridge] (ask) Loading fragment '{fragment_id}' from MemoryBank...")
        fragment = MEMORY_BANK.load(fragment_id)

        if fragment is None:
            print(f"[A3X Bridge] Error (ask): Fragment '{fragment_id}' not found in MemoryBank.")
            logger.error(f"Fragment '{fragment_id}' not found for ask directive.")
            return { "status": "error", "message": f"Fragment '{fragment_id}' not found" }
        
        # --- Check Fragment Type and Get Prediction --- 
        try:
            print(f"[A3X Bridge] (ask) Running prediction with input shape {input_tensor.shape}...")
            # Handle batch input if necessary (predicting only first sample for now)
            if input_tensor.shape[0] != 1:
                 logger.warning(f"Ask directive received batch input (size {input_tensor.shape[0]}). Predicting only for the first sample.")
                 input_tensor = input_tensor[0].unsqueeze(0)

            # Initialize result storage
            prediction_result_dict: Optional[Dict[str, Any]] = None
            return_payload: Optional[Dict[str, Any]] = None
            
            # Call the appropriate predict method
            if isinstance(fragment, (NeuralLanguageFragment, ReflectiveLanguageFragment)):
                 prediction_result_dict = fragment.predict(input_tensor)
            else:
                 # Fragment type doesn't support .predict() in a meaningful way for 'ask'
                 print(f"[A3X Bridge] Error (ask): Fragment '{fragment_id}' type ({type(fragment).__name__}) does not support 'ask' directive prediction.")
                 logger.error(f"Fragment '{fragment_id}' type ({type(fragment).__name__}) does not support 'ask' directive.")
                 return { "status": "error", "message": f"Fragment type {type(fragment).__name__} incompatible with 'ask'" }

            # Validate the prediction result dictionary structure
            if not prediction_result_dict or 'output' not in prediction_result_dict or 'confidence' not in prediction_result_dict:
                 logger.error(f"Fragment '{fragment_id}' predict() returned unexpected format: {prediction_result_dict}")
                 return { "status": "error", "message": "Invalid prediction format from fragment" }

            # Format the return payload based on fragment type
            print(f"[A3X Bridge] (ask) Prediction complete. Output: {prediction_result_dict['output']}, Confidence: {prediction_result_dict['confidence']:.4f}")
            return_payload = {
                "status": "success", 
                "fragment_id": fragment_id, 
                "output": prediction_result_dict["output"],
                "confidence": prediction_result_dict["confidence"] # Always include confidence
            }
            # Add explanation if it's a reflective fragment
            if isinstance(fragment, ReflectiveLanguageFragment) and 'explanation' in prediction_result_dict:
                return_payload["explanation"] = prediction_result_dict["explanation"]
                print(f"[A3X Bridge] (ask) Explanation: {prediction_result_dict['explanation']}")
            
            # --- Store result for confidence check --- 
            LAST_ASK_RESULT = return_payload.copy() # Store the successful result
            print(f"[A3X Bridge] (ask) Stored last ask result: {LAST_ASK_RESULT}")
            
            return return_payload

        except Exception as e:
            print(f"[A3X Bridge] Error (ask) during prediction: {e}")
            # Clear last ask result on error
            LAST_ASK_RESULT = None 
            logger.exception(f"Error during prediction for fragment '{fragment_id}'")
            return { "status": "error", "message": f"Prediction failed: {e}" }

    # =======================================
    # === Handle 'export_fragment' type ===
    # =======================================
    elif directive_type == "export_fragment":
        fragment_id = directive.get("fragment_id")
        export_path_str = directive.get("path") # Optional path from directive

        if not fragment_id or not isinstance(fragment_id, str):
            print("[A3X Bridge] Error (export): 'fragment_id' (string) missing or invalid.")
            logger.error("'fragment_id' missing or invalid for export_fragment directive.")
            return { "status": "error", "error": f"Export failed: 'fragment_id' missing or invalid" }

        # Determine final path
        export_path: Optional[Path] = None
        if export_path_str:
            export_path = Path(export_path_str)
            print(f"[A3X Bridge] (export) Using provided path: {export_path}")
        else:
            # If no path provided, export_path remains None, MemoryBank.export will use default
            print(f"[A3X Bridge] (export) No path provided, using default export location for {fragment_id}.")
        
        # Call MemoryBank export
        try:
            success = MEMORY_BANK.export(fragment_id, export_path) # Pass None if path wasn't provided
            if success:
                # Resolve the actual path used by export (handles default case)
                final_path = export_path if export_path else (MEMORY_BANK.export_dir / f"{fragment_id}.a3xfrag")
                print(f"[A3X Bridge] (export) Successfully exported '{fragment_id}' to {final_path.resolve()}")
                return {"status": "success", "fragment_id": fragment_id, "path": str(final_path.resolve())}
            else:
                print(f"[A3X Bridge] Error (export): MemoryBank.export failed for '{fragment_id}'. Check logs.")
                # MemoryBank.export logs details, so keep error message brief here
                return { "status": "error", "error": f"Export failed for fragment_id {fragment_id}" }
        except Exception as e:
            print(f"[A3X Bridge] Error (export): Unexpected error during export of '{fragment_id}': {e}")
            logger.exception(f"Unexpected error exporting fragment {fragment_id}")
            return { "status": "error", "error": f"Export failed for fragment_id {fragment_id}: {e}" }
            
    # =======================================
    # === Handle 'import_fragment' type ===
    # =======================================
    elif directive_type == "import_fragment":
        import_path_str = directive.get("path")

        if not import_path_str or not isinstance(import_path_str, str):
            print("[A3X Bridge] Error (import): 'path' (string) missing or invalid.")
            logger.error("'path' missing or invalid for import_fragment directive.")
            return { "status": "error", "error": "Import failed: 'path' missing or invalid" }
        
        import_path = Path(import_path_str)
        print(f"[A3X Bridge] (import) Attempting import from: {import_path}")

        # Call MemoryBank import
        try:
            success = MEMORY_BANK.import_a3xfrag(import_path)
            if success:
                print(f"[A3X Bridge] (import) Successfully imported fragment from {import_path}")
                # We don't necessarily know the fragment_id here unless we parse metadata again
                # Return path for confirmation
                return {"status": "success", "path": str(import_path)}
            else:
                print(f"[A3X Bridge] Error (import): MemoryBank.import_a3xfrag failed for {import_path}. Check logs.")
                return { "status": "error", "error": f"Import failed from path {import_path}" }
        except Exception as e:
            print(f"[A3X Bridge] Error (import): Unexpected error during import from '{import_path}': {e}")
            logger.exception(f"Unexpected error importing fragment from {import_path}")
            return { "status": "error", "error": f"Import failed from path {import_path}: {e}" }

    # =======================================
    # === Handle 'reflect_fragment' type ==
    # =======================================
    elif directive_type == "reflect_fragment":
        fragment_id = directive.get("fragment_id")
        output_format = directive.get("format", "a3l") # Default to a3l

        # --- MOCK FOR TESTING KNOWLEDGE INTERPRETER ---
        if fragment_id == 'dummy_reflector':
            print(f"--- MOCK REFLECTION for {fragment_id} ---")
            # Simulate LLM reflection suggesting training and creation
            mock_reflection_text = "O fragmento 'dummy_reflector' parece impreciso, talvez devêssemos treiná-lo novamente por 3 épocas. Também, podemos criar um fragmento 'dummy_reflector_v2' com base em 'dummy_reflector'."
            print(f"--- Returning mock text: '{mock_reflection_text}' ---")
            # Return raw text for the interpreter to process
            return {"status": "success", "result": mock_reflection_text, "format": "text"} 
        # --- END MOCK ---

        if not fragment_id or not isinstance(fragment_id, str):
            print("[A3X Bridge] Error (reflect): 'fragment_id' (string) missing or invalid.")
            logger.error("'fragment_id' missing or invalid for reflect_fragment directive.")
            return { "status": "error", "error": "Reflection failed: 'fragment_id' missing or invalid" }
        
        print(f"[A3X Bridge] (reflect) Attempting to reflect on fragment '{fragment_id}' (format: {output_format})...")
        fragment = MEMORY_BANK.load(fragment_id)

        if fragment is None:
            print(f"[A3X Bridge] Error (reflect): Fragment '{fragment_id}' not found.")
            logger.error(f"Fragment '{fragment_id}' not found for reflection.")
            return { "status": "error", "error": f"Fragment '{fragment_id}' not found." }

        # --- Handle A3L Format --- 
        if output_format == "a3l":
             if hasattr(fragment, 'generate_reflection_a3l') and callable(getattr(fragment, 'generate_reflection_a3l')):
                 try:
                     a3l_reflection = fragment.generate_reflection_a3l()
                     print(f"[A3X Bridge] (reflect) Generated A3L reflection for '{fragment_id}'.")
                     return {"status": "success", "reflection_a3l": a3l_reflection}
                 except Exception as e:
                     logger.error(f"Error generating A3L reflection for {fragment_id}: {e}", exc_info=True)
                     return {"status": "error", "error": f"Failed to generate A3L reflection: {e}"}
             else:
                  logger.error(f"Fragment '{fragment_id}' ({type(fragment).__name__}) does not support generate_reflection_a3l method.")
                  return { "status": "error", "error": f"Fragment type {type(fragment).__name__} does not support A3L reflection format." }
        
        # --- Handle Dictionary Format (Default) --- 
        elif output_format == "dict":
            # Gather reflection data
            reflection_data = {
                "fragment_id": fragment_id,
                "class_name": fragment.__class__.__name__,
                "module": fragment.__class__.__module__
            }

            # Add dimensional attributes if they exist
            for attr in ['input_dim', 'hidden_dim', 'output_dim', 'num_classes']:
                if hasattr(fragment, attr):
                    reflection_data[attr] = getattr(fragment, attr)
            
            # Add label map if it exists
            if hasattr(fragment, 'id_to_label'):
                reflection_data['id_to_label'] = getattr(fragment, 'id_to_label')
            
            # Add description if it exists
            if hasattr(fragment, 'description'):
                 reflection_data['description'] = getattr(fragment, 'description', None)

            # Get last modified time of the state file
            try:
                state_file_path = MEMORY_BANK._get_save_path(fragment_id, "pt")
                if state_file_path.exists():
                    mtime_timestamp = os.path.getmtime(state_file_path)
                    reflection_data['last_modified_timestamp'] = mtime_timestamp
                    # Convert to readable string format as well (optional)
                    from datetime import datetime
                    reflection_data['last_modified_utc'] = datetime.utcfromtimestamp(mtime_timestamp).isoformat() + 'Z'
                else:
                     reflection_data['last_modified_timestamp'] = None
                     reflection_data['last_modified_utc'] = "State file not found"
            except Exception as e:
                logger.warning(f"Could not get modification time for {fragment_id}: {e}")
                reflection_data['last_modified_timestamp'] = None
                reflection_data['last_modified_utc'] = f"Error retrieving: {e}"

            print(f"[A3X Bridge] (reflect) Reflection data gathered for '{fragment_id}'.")
            return {"status": "success", "reflection": reflection_data}
        
        # --- Handle Invalid Format --- 
        else:
            logger.error(f"Invalid reflection format specified: {output_format}")
            return { "status": "error", "error": f"Invalid reflection format: {output_format}. Supported formats: dict, a3l." }

    # =======================================
    # === Handle 'conditional_directive' ==
    # =======================================
    elif directive_type == "conditional_directive":
        condition = directive.get("condition")
        action_directive = directive.get("action")
        logger.info(f"[A3X Bridge] Evaluating conditional directive. Condition: {condition}")

        condition_met = False
        condition_type = condition.get("condition_type", "attribute_check") # Default to original check
        print(f"[A3X Bridge] (conditional) Evaluating condition type: {condition_type}")

        # --- Evaluate Condition Type: Attribute Check --- 
        if condition_type == "attribute_check":
            cond_fragment_id = condition.get("fragment_id")
            cond_attribute = condition.get("attribute")
            cond_expected_value = condition.get("expected_value")

            if not all([cond_fragment_id, cond_attribute]) or cond_expected_value is None:
                print("[A3X Bridge] Error (conditional): Incomplete condition details.")
                logger.error(f"Incomplete condition details in conditional directive: {condition}")
                return { "status": "error", "error": "Incomplete condition in attribute check." }
                
            print(f"[A3X Bridge] (conditional) Evaluating: Check if fragment '{cond_fragment_id}' attribute '{cond_attribute}' == {cond_expected_value}")
            try:
                fragment = MEMORY_BANK.load(cond_fragment_id)
                if fragment is None:
                     print(f"[A3X Bridge] (conditional) Condition fragment '{cond_fragment_id}' not found. Condition is FALSE.")
                elif not hasattr(fragment, cond_attribute):
                     print(f"[A3X Bridge] (conditional) Condition fragment '{cond_fragment_id}' does not have attribute '{cond_attribute}'. Condition is FALSE.")
                     logger.warning(f"Attribute '{cond_attribute}' not found on fragment '{cond_fragment_id}' for conditional check.")
                else:
                    actual_value = getattr(fragment, cond_attribute)
                    if isinstance(actual_value, (int, float)) and isinstance(cond_expected_value, (int, float)):
                        if actual_value == cond_expected_value:
                             condition_met = True
                             print(f"[A3X Bridge] (conditional) Condition MET: {cond_attribute}={actual_value}")
                        else:
                             print(f"[A3X Bridge] (conditional) Condition FALSE: {cond_attribute}={actual_value} (expected {cond_expected_value})")
                    else:
                         print(f"[A3X Bridge] (conditional) Attribute type mismatch or unsupported type for comparison ({type(actual_value)} vs {type(cond_expected_value)}). Condition is FALSE.")
                         logger.warning(f"Unsupported comparison type for attribute '{cond_attribute}' on fragment '{cond_fragment_id}'")
            except Exception as e:
                print(f"[A3X Bridge] (conditional) Error evaluating attribute condition for fragment '{cond_fragment_id}': {e}")
                logger.exception(f"Error evaluating attribute condition for {cond_fragment_id}")
                # condition_met remains False
        
        # --- Evaluate Condition Type: Confidence Check --- 
        elif condition_type == "confidence_check":
            threshold = condition.get("threshold")
            if threshold is None:
                 print("[A3X Bridge] Error (conditional): Missing 'threshold' for confidence check.")
                 logger.error("Missing threshold for confidence_check conditional.")
                 return { "status": "error", "error": "Missing threshold in confidence condition." }
            
            print(f"[A3X Bridge] (conditional) Evaluating: Check if last ask confidence < {threshold:.3f}")
            
            if LAST_ASK_RESULT is None:
                 print("[A3X Bridge] (conditional) No previous 'ask' result found. Condition is FALSE.")
                 # condition_met remains False
            elif not isinstance(LAST_ASK_RESULT.get('confidence'), (int, float)):
                 print(f"[A3X Bridge] (conditional) Last ask result missing or invalid confidence ({LAST_ASK_RESULT.get('confidence')}). Condition is FALSE.")
                 logger.warning(f"Invalid or missing confidence in LAST_ASK_RESULT: {LAST_ASK_RESULT}")
                 # condition_met remains False
            else:
                last_confidence = LAST_ASK_RESULT['confidence']
                if last_confidence < threshold:
                     condition_met = True
                     print(f"[A3X Bridge] (conditional) Condition MET: Last confidence {last_confidence:.4f} < {threshold:.3f}")
                else:
                     print(f"[A3X Bridge] (conditional) Condition FALSE: Last confidence {last_confidence:.4f} >= {threshold:.3f}")
                     # condition_met remains False
                     
        # --- Unknown Condition Type --- 
        else:
             print(f"[A3X Bridge] Error (conditional): Unknown condition type '{condition_type}'.")
             logger.error(f"Unknown condition type in conditional directive: {condition_type}")
             return { "status": "error", "error": f"Unknown condition type: {condition_type}." }

        # --- Execute Action if Condition Met --- 
        if condition_met:
            logger.info("[A3X Bridge] Condition MET. Executing conditional action...")
            # <<< Recursive call needs await and context_store and post_message_handler >>>
            return await handle_directive(action_directive, fragment_instances, context_store, post_message_handler)
        else:
            logger.info("[A3X Bridge] Condition NOT MET. Skipping conditional action.")
            return {"status": "skipped", "message": "Condition not met"}

    # =======================================
    # === Handle 'solicitar_exemplos' type (NEW) ===
    # =======================================
    elif directive_type == "solicitar_exemplos":
        task_name = directive.get("task_name")
        professor_id = directive.get("professor_id", "prof_geral") # Default professor
        num_examples = directive.get("num_examples", 10) # Default number
        example_format = directive.get("example_format") # <<< NOVO: Get optional format

        if not task_name:
            logger.error("[A3X Bridge] Error (solicitar_exemplos): 'task_name' missing.")
            return { "status": "error", "message": "'task_name' missing for solicitar_exemplos" }
        
        if not fragment_instances:
             logger.error("[A3X Bridge] Error (solicitar_exemplos): fragment_instances dictionary not provided.")
             return { "status": "error", "message": "Fragment instances not available to find Professor." }

        # --- Find Professor Fragment ---
        professor_fragment = fragment_instances.get(professor_id)
        if not isinstance(professor_fragment, ProfessorLLMFragment):
             logger.error(f"[A3X Bridge] Error (solicitar_exemplos): Professor fragment '{professor_id}' not found or is not a ProfessorLLMFragment.")
             return { "status": "error", "message": f"Professor fragment '{professor_id}' not available." }
        
        logger.info(f"[A3X Bridge] (solicitar_exemplos) Requesting {num_examples} examples for task '{task_name}' from Professor '{professor_id}'.")
        
        # --- Generate Prompt (Adapts based on example_format) ---
        if example_format and isinstance(example_format, dict):
            # <<< NOVO: Prompt uses the provided format >>>
            try:
                format_str = json.dumps(example_format, ensure_ascii=False, indent=2) # Pretty print for prompt
                logger.info(f"[A3X Bridge] (solicitar_exemplos) Using planned example format: {format_str}")
                prompt = f'''
Por favor, gere {num_examples} exemplos de dados para a tarefa '{task_name}'.
Use EXATAMENTE a seguinte estrutura JSON para cada exemplo:
{format_str}

Retorne a resposta como uma lista JSON contendo os {num_examples} exemplos.
Exemplo de Resposta:
[
  {format_str}, 
  {{... outro exemplo ...}}
]
Retorne APENAS a lista JSON, sem nenhum texto adicional antes ou depois.
'''
            except Exception as json_err:
                logger.error(f"[A3X Bridge] (solicitar_exemplos) Error serializing example_format: {json_err}. Falling back to default prompt.", exc_info=True)
                example_format = None # Force fallback to default prompt
        
        if not example_format: # If format wasn't provided or serialization failed
            # <<< Original Prompt >>>
            logger.info("[A3X Bridge] (solicitar_exemplos) Using default example format prompt (input/label).")
            prompt = f'''
Por favor, gere {num_examples} exemplos de dados de entrada e rótulos correspondentes para a tarefa '{task_name}'.
Os exemplos devem ser representativos da tarefa.
Retorne a resposta como uma lista JSON de objetos, onde cada objeto tem as chaves "input" e "label".
Exemplo de formato:
[
  {{"input": "exemplo de entrada 1", "label": "rótulo A"}},
  {{"input": "exemplo de entrada 2", "label": "rótulo B"}}
]
Retorne APENAS a lista JSON, sem nenhum texto adicional antes ou depois.
'''
        
        try:
            # --- Call Professor ---
            response = await professor_fragment.ask_llm(prompt)
            if not response:
                 logger.error(f"[A3X Bridge] (solicitar_exemplos) Professor '{professor_id}' returned an empty response.")
                 return { "status": "error", "message": "Professor returned empty response." }
                 
            logger.info(f"[A3X Bridge] (solicitar_exemplos) Received response from Professor: {response[:200]}...")

            # --- Parse Response and Register Examples ---
            from ..utils.data_logger import registrar_exemplo_de_aprendizado as log_example # Use the specific logger

            parsed_examples = []
            try:
                # Attempt to find JSON list within the response
                json_match = re.search(r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\})*\s*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_examples = json.loads(json_str)
                    if not isinstance(parsed_examples, list):
                        raise ValueError("Parsed JSON is not a list.")
                    logger.info(f"[A3X Bridge] (solicitar_exemplos) Successfully parsed {len(parsed_examples)} examples from JSON.")
                else:
                    logger.error("[A3X Bridge] (solicitar_exemplos) No valid JSON list found in Professor's response.")
                    return { "status": "error", "message": "Could not find valid JSON list in Professor response." }

            except (json.JSONDecodeError, ValueError) as json_err:
                logger.error(f"[A3X Bridge] (solicitar_exemplos) Failed to parse JSON response from Professor: {json_err}. Response: {response}")
                return { "status": "error", "message": f"Failed to parse Professor response as JSON list: {json_err}" }

            # --- Register Examples Asynchronously ---
            registration_tasks = []
            registered_count = 0
            failed_count = 0
            for example in parsed_examples:
                if isinstance(example, dict) and "input" in example and "label" in example:
                    input_data = example["input"]
                    label_data = example["label"]
                    # Validate types (simple check)
                    if isinstance(input_data, str) and isinstance(label_data, str):
                        # Create task to register the example
                        registration_tasks.append(
                            log_example(task_name=task_name, input_data=input_data, label=label_data)
                        )
                    else:
                        logger.warning(f"[A3X Bridge] (solicitar_exemplos) Skipping invalid example format (non-string input/label): {example}")
                        failed_count += 1
                else:
                    logger.warning(f"[A3X Bridge] (solicitar_exemplos) Skipping invalid example format: {example}")
                    failed_count += 1
            
            # Execute registration tasks concurrently
            if registration_tasks:
                results = await asyncio.gather(*registration_tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"[A3X Bridge] (solicitar_exemplos) Error registering example: {result}", exc_info=result)
                        failed_count += 1
                    elif result is True: # Assuming log_example returns True on success
                        registered_count += 1
                    else: # Assuming log_example returned False or None on failure
                        logger.warning(f"[A3X Bridge] (solicitar_exemplos) data_logger reported failure for an example.")
                        failed_count +=1

            logger.info(f"[A3X Bridge] (solicitar_exemplos) Registration complete for task '{task_name}'. Registered: {registered_count}, Failed/Skipped: {failed_count}")
            return { 
                "status": "success", 
                "message": f"Requested examples for '{task_name}'. Registered: {registered_count}, Failed/Skipped: {failed_count}",
                "registered_count": registered_count,
                "failed_count": failed_count
            }

        except Exception as e:
            logger.error(f"[A3X Bridge] (solicitar_exemplos) Unexpected error: {e}", exc_info=True)
            return { "status": "error", "message": f"Unexpected error during solicitar_exemplos: {e}" }

    # =======================================
    # === Handle 'avaliar_fragmento' type (NEW) ===
    # =======================================
    elif directive_type == "avaliar_fragmento":
        fragment_id = directive.get("fragment_id")
        task_name = directive.get("task_name") # Task name specifies the dataset
        test_split_ratio = directive.get("test_split", 0.2) # Optional split ratio

        if not fragment_id or not task_name:
            logger.error("[A3X Bridge] Error (avaliar): 'fragment_id' or 'task_name' missing.")
            return { "status": "error", "message": "'fragment_id' or 'task_name' missing for avaliar_fragmento" }
        
        if not context_store:
             logger.error("[A3X Bridge] Error (avaliar): ContextStore instance not provided.")
             return { "status": "error", "message": "ContextStore not available for saving evaluation results." }

        # --- Load Fragment ---
        logger.info(f"[A3X Bridge] (avaliar) Loading fragment '{fragment_id}' for evaluation on task '{task_name}'.")
        fragment = MEMORY_BANK.load(fragment_id)

        if fragment is None:
            logger.error(f"[A3X Bridge] Error (avaliar): Fragment '{fragment_id}' not found.")
            return { "status": "error", "message": f"Fragment '{fragment_id}' not found for evaluation." }

        # --- Check if Fragment Supports Evaluation ---
        if not hasattr(fragment, 'evaluate') or not callable(getattr(fragment, 'evaluate')):
             logger.error(f"[A3X Bridge] Error (avaliar): Fragment '{fragment_id}' ({type(fragment).__name__}) does not support the 'evaluate' method.")
             return { "status": "error", "message": f"Fragment type {type(fragment).__name__} does not support evaluation." }

        # --- Run Evaluation ---
        logger.info(f"[A3X Bridge] (avaliar) Calling evaluate() on fragment '{fragment_id}' for task '{task_name}'.")
        try:
            evaluation_results = await fragment.evaluate(task_name=task_name, test_split_ratio=test_split_ratio)
            logger.info(f"[A3X Bridge] (avaliar) Evaluation results for '{fragment_id}': {evaluation_results}")
            
            eval_status = evaluation_results.get("status")
            
            # --- Store Results in ContextStore (if successful/warning) ---
            if eval_status in ["success", "warning_small_dataset"]:
                try:
                    # Create a unique key including a timestamp
                    timestamp_ms = int(time.time() * 1000)
                    score_key = f"evaluation_score:{fragment_id}:{task_name}:{timestamp_ms}"
                    
                    # Prepare data to save (add timestamp to results dict)
                    save_data = evaluation_results.copy()
                    save_data["timestamp"] = timestamp_ms
                    
                    await context_store.set(score_key, save_data)
                    logger.info(f"[A3X Bridge] (avaliar) Saved evaluation results for '{fragment_id}' to ContextStore with key '{score_key}'.")
                except Exception as cs_err:
                     logger.error(f"[A3X Bridge] (avaliar) Failed to save evaluation results for '{fragment_id}' to ContextStore: {cs_err}", exc_info=True)
                     # Update status/message to reflect storage failure?
                     evaluation_results["message"] = evaluation_results.get("message", "") + f" [Error saving score: {cs_err}]"
                     # Don't necessarily change status from success if eval itself worked

            # Return the original evaluation results dictionary
            return evaluation_results

        except Exception as e:
            logger.error(f"[A3X Bridge] (avaliar) Unexpected error during evaluation call for '{fragment_id}': {e}", exc_info=True)
            return { 
                "status": "error", 
                "fragment_id": fragment_id,
                "task_name": task_name,
                "message": f"Unexpected error during fragment evaluation: {e}" 
            }

    # ==================================================
    # === Handle 'comparar_desempenho' type (NEW) ===
    # ==================================================
    elif directive_type == "comparar_desempenho":
        fragment_id = directive.get("fragment_id")
        task_name = directive.get("task_name")

        if not fragment_id or not task_name:
            logger.error(f"[A3X Bridge] Error (comparar): Missing 'fragment_id' or 'task_name'. Directive: {directive}")
            return { "status": "error", "message": "Missing fragment_id or task_name for comparison." }
        
        if not context_store:
            logger.error(f"[A3X Bridge] Error (comparar): ContextStore instance not provided.")
            return { "status": "error", "message": "ContextStore not available for comparing performance." }
        
        logger.info(f"[A3X Bridge] (comparar) Comparing performance for fragment '{fragment_id}' on task '{task_name}'.")

        try:
            # --- Scan for relevant scores --- 
            scan_prefix = f"evaluation_score:{fragment_id}:{task_name}:"
            logger.debug(f"Scanning ContextStore with prefix: {scan_prefix}")
            all_scores_dict = await context_store.scan(prefix=scan_prefix)
            
            if not all_scores_dict:
                 logger.warning(f"[A3X Bridge] (comparar) No evaluation scores found for prefix '{scan_prefix}'. Cannot compare.")
                 return { "status": "no_history", "message": f"No evaluation history found for fragment '{fragment_id}' on task '{task_name}'." }

            # --- Sort scores by timestamp --- 
            scores_list = sorted(all_scores_dict.values(), key=lambda x: x.get("timestamp", 0), reverse=True)
            
            if len(scores_list) < 2:
                logger.warning(f"[A3X Bridge] (comparar) Less than 2 scores found ({len(scores_list)}). Cannot compare.")
                latest_score_info = f"Latest score: {scores_list[0]['accuracy']:.2%} at {scores_list[0]['timestamp']}" if scores_list else "No scores available."
                return { "status": "insufficient_history", "message": f"Insufficient evaluation history ({len(scores_list)} scores) to compare. {latest_score_info}" }

            # --- Compare latest two scores --- 
            latest_score = scores_list[0]
            previous_score = scores_list[1]
            latest_acc = latest_score.get("accuracy", -1)
            previous_acc = previous_score.get("accuracy", -1)

            logger.info(f"[A3X Bridge] (comparar) Latest score: {latest_acc:.4f} (at {latest_score['timestamp']}), Previous score: {previous_acc:.4f} (at {previous_score['timestamp']})")

            comparison_result = ""
            corrective_action_taken = False

            if latest_acc > previous_acc:
                comparison_result = f"Fragmento evoluiu com sucesso (Acurácia: {previous_acc:.2%} -> {latest_acc:.2%})"
                logger.info(f"[A3X Bridge] (comparar) Performance improved for '{fragment_id}' on '{task_name}'.")
            elif latest_acc < previous_acc:
                comparison_result = f"O aprendizado não foi eficaz. Verificar qualidade do dataset. (Acurácia: {previous_acc:.2%} -> {latest_acc:.2%})"
                logger.warning(f"[A3X Bridge] (comparar) Performance DEGRADED for '{fragment_id}' on '{task_name}'.")
                
                # --- Trigger Self-Correction: Request More Examples --- 
                # <<< Check if post_message_handler was provided >>>
                if post_message_handler:
                    try:
                        logger.info(f"[A3X Bridge] (comparar) Performance degraded. Triggering 'solicitar exemplos' for task '{task_name}'.")
                        correction_directive = {
                            "type": "solicitar_exemplos",
                            "task_name": task_name,
                            "origin_suggestion": {"source": f"SelfCorrection triggered by {fragment_id} performance degradation on {task_name}"}
                        }
                        # <<< Use the passed handler >>>
                        await post_message_handler(
                            message_type="a3l_command",
                            content=correction_directive,
                            target_fragment="Executor" # Send to message processor
                        )
                        corrective_action_taken = True
                        logger.info(f"[A3X Bridge] (comparar) 'solicitar exemplos' directive posted to queue.")
                    except Exception as post_err:
                        logger.error(f"[A3X Bridge] (comparar) Failed to post corrective 'solicitar exemplos' directive: {post_err}", exc_info=True)
                        comparison_result += " [Falha ao solicitar mais exemplos (Handler Error)]"
                else:
                     logger.error("[A3X Bridge] (comparar) Cannot trigger self-correction: post_message_handler not provided.")
                     comparison_result += " [Falha ao solicitar mais exemplos (No Handler)]"

            else: # latest_acc == previous_acc
                comparison_result = f"Desempenho estável (Acurácia: {latest_acc:.2%})"
                logger.info(f"[A3X Bridge] (comparar) Performance stable for '{fragment_id}' on '{task_name}'.")

            return {
                "status": "success", 
                "message": comparison_result,
                "latest_accuracy": latest_acc,
                "previous_accuracy": previous_acc,
                "latest_timestamp": latest_score['timestamp'],
                "previous_timestamp": previous_score['timestamp'],
                "corrective_action_triggered": corrective_action_taken
            }

        except Exception as comp_err:
            logger.error(f"[A3X Bridge] (comparar) Error comparing performance for fragment '{fragment_id}': {comp_err}", exc_info=True)
            return { "status": "error", "message": f"Error during performance comparison: {comp_err}" }

    # =======================================
    # === Handle 'planejar_dados' type (NEW) ===
    # =======================================
    elif directive_type == "planejar_dados":
        task_name = directive.get("task_name")
        professor_id = directive.get("professor_id", "prof_geral") # Default professor

        if not task_name:
            logger.error("[A3X Bridge] Error (planejar_dados): 'task_name' missing.")
            return { "status": "error", "message": "'task_name' missing for planejar_dados" }
        
        if not post_message_handler:
            logger.error("[A3X Bridge] Error (planejar_dados): post_message_handler not provided.")
            return { "status": "error", "message": "post_message_handler not available for planning data." }
        
        if not fragment_instances:
             logger.error("[A3X Bridge] Error (planejar_dados): fragment_instances dictionary not provided.")
             return { "status": "error", "message": "Fragment instances not available to find Professor." }

        # --- Find Professor Fragment ---
        professor_fragment = fragment_instances.get(professor_id)
        if not isinstance(professor_fragment, ProfessorLLMFragment):
             logger.error(f"[A3X Bridge] Error (planejar_dados): Professor fragment '{professor_id}' not found or is not a ProfessorLLMFragment.")
             return { "status": "error", "message": f"Professor fragment '{professor_id}' not available." }
        
        logger.info(f"[A3X Bridge] (planejar_dados) Requesting dataset format plan for task '{task_name}' from Professor '{professor_id}'.")
        
        # --- Formulate Prompt for Professor ---
        prompt = f"""Qual seria a melhor estrutura JSON para um exemplo de dataset (input/label) para treinar um modelo na tarefa: '{task_name}'?
Retorne APENAS a estrutura JSON de um único exemplo, como {"input": "exemplo", "label": "categoria"} ou {"texto": "...", "sentimento": "positivo"}. NÃO inclua em uma lista, retorne apenas o objeto JSON.
Exemplo de Resposta:
{{"input": "texto de entrada", "label": "classe_predita"}}
"""
        
        try:
            # --- Call Professor --- 
            response = await professor_fragment.ask_llm(prompt)
            if not response:
                 logger.error(f"[A3X Bridge] (planejar_dados) Professor '{professor_id}' returned an empty response for format planning.")
                 return { "status": "error", "message": "Professor returned empty response for format planning." }
                 
            logger.info(f"[A3X Bridge] (planejar_dados) Received format plan response from Professor: {response[:200]}...")

            # --- Extract JSON format from response --- 
            extracted_format = None
            try:
                # Attempt to find JSON object within the response
                # More robust regex might be needed depending on LLM output variations
                json_match = re.search(r'{\s*"[^\"]+"\s*:\s*.*?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_json = json.loads(json_str)
                    if isinstance(parsed_json, dict):
                        extracted_format = parsed_json
                        logger.info(f"[A3X Bridge] (planejar_dados) Successfully extracted example format: {extracted_format}")
                    else:
                        logger.warning(f"[A3X Bridge] (planejar_dados) Parsed JSON is not a dictionary: {json_str}")
                else:
                    logger.error("[A3X Bridge] (planejar_dados) No valid JSON object found in Professor's response for format plan.")
                    # Consider fallback or error

            except json.JSONDecodeError as json_err:
                logger.error(f"[A3X Bridge] (planejar_dados) Failed to parse JSON format response from Professor: {json_err}. Response: {response}")
                # Consider fallback or error

            if not extracted_format:
                # Fallback or error if format extraction failed
                logger.error("[A3X Bridge] (planejar_dados) Failed to obtain valid example format from Professor. Cannot proceed with solicitar_exemplos.")
                return { "status": "error", "message": "Failed to obtain valid example format from Professor." }
            
            # --- Post solicitar_exemplos directive with the format ---
            solicitar_directive = {
                "type": "solicitar_exemplos",
                "task_name": task_name,
                "example_format": extracted_format, # Pass the extracted format
                "professor_id": professor_id, # Pass professor_id along if needed
                "_origin": { # Add origin for traceability
                    "source": "planejar_dados",
                    "original_task": task_name
                }
            }
            await post_message_handler(
                message_type="a3l_command", 
                content=solicitar_directive,
                target_fragment="Executor" # Or appropriate target
            )
            logger.info(f"[A3X Bridge] (planejar_dados) Posted 'solicitar_exemplos' directive with planned format for task '{task_name}'.")
            
            return { "status": "success", "message": f"Data planning initiated for '{task_name}'. Posted solicitar_exemplos.", "planned_format": extracted_format }

        except Exception as e:
            logger.error(f"[A3X Bridge] (planejar_dados) Unexpected error: {e}", exc_info=True)
            return { "status": "error", "message": f"Unexpected error during planejar_dados: {e}" }

    # =======================================
    # === Unknown Directive Type ==========
    # =======================================
    else:
        logger.error(f"[A3X Bridge] Received unhandled directive type: '{directive_type}'")
        return { "status": "error", "message": f"Unhandled directive type: {directive_type}" }

    # Should not be reached if types are handled correctly
    return None 