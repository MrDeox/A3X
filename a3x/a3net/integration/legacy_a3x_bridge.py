from typing import Dict, List, Any, Optional
import torch
import torch.optim as optim
import re # Import regex for cleaning goal string
import logging # Added logging

# A³Net imports
from a3x.a3net.trainer.dataset_builder import build_dataset_from_context
from a3x.a3net.core.fragment_cell import FragmentCell
from a3x.a3net.trainer.train_loop import train_fragment_cell
from a3x.a3net.core.memory_bank import MemoryBank # <<< Import MemoryBank >>>
from a3x.a3net.core.cognitive_graph import CognitiveGraph # <<< Import CognitiveGraph >>>
from a3x.a3net.core.neural_language_fragment import NeuralLanguageFragment # <<< Import NeuralLanguageFragment >>>
from a3x.a3net.core.reflective_language_fragment import ReflectiveLanguageFragment # <<< Import ReflectiveLanguageFragment >>>

# Configure logging
logger = logging.getLogger(__name__)

# --- Module-Level Memory Bank Instance ---
# Instantiate with a specific directory for persistence
MEMORY_BANK = MemoryBank(save_dir="a3net_memory") 
# Note: This creates one instance when the module is first imported.

def handle_directive(directive: Dict) -> Optional[Dict[str, Any]]: # <<< Add return type >>>
    """Handles directives received from the symbolic A³X system.
    
    Supports 'train_fragment', 'run_graph', and 'ask' types.
    """
    directive_type = directive.get("type")
    goal = directive.get('goal', None) 
    print(f"[A3X Bridge] Received directive type: '{directive_type}', goal: {goal or 'Not specified'}")

    # =======================================
    # === Handle 'train_fragment' type ===
    # =======================================
    if directive_type == "train_fragment":
        # --- Extract parameters ---
        context_id = directive.get('context_id')
        input_dim = directive.get('input_dim')
        output_dim = directive.get('output_dim')
        hidden_dim = directive.get('hidden_dim', 64)
        epochs = directive.get('epochs', 10) 
        learning_rate = directive.get('learning_rate', 0.001)

        if not all([context_id, input_dim, output_dim]):
            print("[A3X Bridge] Error (train): Directive missing required fields (context_id, input_dim, output_dim).")
            logger.error("Missing required fields for train_fragment directive.")
            return { "status": "error", "message": "Missing required fields" }

        print(f"[A3X Bridge] (train) Parsed parameters: context_id={context_id}, input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}, epochs={epochs}")

        # --- Build Dataset ---
        dataset = build_dataset_from_context(context_id)
        if not dataset:
            print(f"[A3X Bridge] Error (train): Failed to build dataset for context {context_id}.")
            logger.error(f"Failed to build dataset for context {context_id}")
            return { "status": "error", "message": "Dataset build failed" }

        # --- Create Model ---
        try:
            cell = FragmentCell(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
            print(f"[A3X Bridge] (train) Created FragmentCell: {input_dim} -> {hidden_dim} -> {output_dim}")
        except Exception as e:
            print(f"[A3X Bridge] Error (train) creating FragmentCell: {e}")
            logger.exception("Error creating FragmentCell")
            return { "status": "error", "message": f"Model creation failed: {e}" }

        # --- Create Optimizer ---
        optimizer = optim.Adam(cell.parameters(), lr=learning_rate)
        print(f"[A3X Bridge] (train) Created Adam optimizer with lr={learning_rate}")

        # --- Train Model ---
        print(f"[A3X Bridge] (train) Starting training process...")
        training_successful = False
        try:
            train_fragment_cell(cell=cell, dataset=dataset, epochs=epochs, optimizer=optimizer)
            print(f"[A3X Bridge] (train) Training completed for context {context_id}.")
            training_successful = True
        except Exception as e:
            print(f"[A3X Bridge] Error (train) during training: {e}")
            logger.exception("Error during training")
            # Continue to save even if training fails? For now, no.

        # --- Save Trained Fragment ---
        if training_successful:
            if goal:
                clean_goal = re.sub(r'[^a-z0-9_]+', '', goal.lower().replace(" ", "_"))
                fragment_id = f"frag_{clean_goal[:30]}"
            else:
                fragment_id = f"frag_{context_id[:20].replace(' ', '_')}" # Use context ID as fallback
            
            try:
                MEMORY_BANK.save(fragment_id, cell)
                print(f"[A3X Bridge] (train) Fragment {fragment_id} saved to memory.") 
                return { "status": "success", "message": "Training complete, fragment saved", "fragment_id": fragment_id }
            except Exception as e:
                print(f"[A3X Bridge] Error (train) saving fragment '{fragment_id}' to MemoryBank: {e}")
                logger.exception("Error saving fragment to MemoryBank")
                return { "status": "error", "message": f"Save failed: {e}" } # Return error even if training worked
        else:
            print("[A3X Bridge] (train) Skipping fragment save due to training error or incompletion.")
            return { "status": "error", "message": "Training failed" }

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

            # <<< Check specific fragment type for prediction >>>
            if isinstance(fragment, ReflectiveLanguageFragment):
                prediction_result = fragment.predict(input_tensor)
                # Validate expected dictionary structure
                if isinstance(prediction_result, dict) and 'output' in prediction_result and 'explanation' in prediction_result:
                    print(f"[A3X Bridge] (ask) Reflective prediction complete. Output: {prediction_result['output']}, Explanation: {prediction_result['explanation']}")
                    return {
                        "status": "success", 
                        "fragment_id": fragment_id, 
                        "output": prediction_result["output"],
                        "explanation": prediction_result["explanation"]
                    }
                else:
                    logger.error(f"Reflective fragment '{fragment_id}' predict() returned unexpected format: {prediction_result}")
                    return { "status": "error", "message": "Invalid prediction format from reflective fragment" }
            
            elif isinstance(fragment, NeuralLanguageFragment):
                 # Standard language fragment, expect string output
                 predicted_label = fragment.predict(input_tensor)
                 if isinstance(predicted_label, str):
                     print(f"[A3X Bridge] (ask) Standard prediction complete. Output: {predicted_label}")
                     return {"status": "success", "fragment_id": fragment_id, "output": predicted_label}
                 else:
                     logger.error(f"Standard language fragment '{fragment_id}' predict() returned unexpected format: {predicted_label}")
                     return { "status": "error", "message": "Invalid prediction format from standard fragment" }
            
            else:
                 # Fragment type doesn't support .predict() in a meaningful way for 'ask'
                 print(f"[A3X Bridge] Error (ask): Fragment '{fragment_id}' type ({type(fragment).__name__}) does not support 'ask' directive prediction.")
                 logger.error(f"Fragment '{fragment_id}' type ({type(fragment).__name__}) does not support 'ask' directive.")
                 return { "status": "error", "message": f"Fragment type {type(fragment).__name__} incompatible with 'ask'" }

        except Exception as e:
            print(f"[A3X Bridge] Error (ask) during prediction: {e}")
            logger.exception(f"Error during prediction for fragment '{fragment_id}'")
            return { "status": "error", "message": f"Prediction failed: {e}" }

    # =======================================
    # === Handle Unknown/Missing type ===
    # =======================================
    else:
        print(f"[A3X Bridge] Error: Unknown or missing directive type: '{directive_type}'. Supported types: 'train_fragment', 'run_graph', 'ask'.")
        logger.error(f"Unknown or missing directive type: {directive_type}")
        return { "status": "error", "message": f"Unknown directive type: {directive_type}" }

    # Should not be reached if types are handled correctly
    return None 