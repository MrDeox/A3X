import logging
from typing import Dict, Any, Optional

# Assuming these are available in the environment
from a3x.a3net.core.memory_bank import MemoryBank
from a3x.a3net.core.neural_language_fragment import NeuralLanguageFragment
from a3x.a3net.core.reflective_language_fragment import ReflectiveLanguageFragment
from a3x.a3net.trainer.dataset_builder import get_embedding_model

logger = logging.getLogger(__name__)

async def handle_create_fragment(
    directive: Dict[str, Any], 
    memory_bank: MemoryBank
) -> Optional[Dict[str, Any]]:
    """Handles the 'create_fragment' directive logic."""
    
    fragment_id = directive.get("fragment_id")
    fragment_type = directive.get("fragment_type", "neural").lower() # Default to neural
    params = directive.get("params", {}) # Optional parameters like dim, classes etc.

    if not fragment_id:
        logger.error("[A3X Bridge Handler - Create] Error: 'fragment_id' missing.")
        return { "status": "error", "message": "'fragment_id' missing for create_fragment" }

    # Check if fragment already exists
    existing_fragment = memory_bank.load(fragment_id)
    if existing_fragment:
        logger.warning(f"[A3X Bridge Handler - Create] Fragment '{fragment_id}' already exists. Skipping creation.")
        return { "status": "skipped", "message": f"Fragment '{fragment_id}' already exists.", "fragment_id": fragment_id }

    logger.info(f"[A3X Bridge Handler - Create] Creating fragment '{fragment_id}' of type '{fragment_type}' with params: {params}")

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
                    logger.info(f"[A3X Bridge Handler - Create] Determined actual input_dim={actual_input_dim} from embedding model.")
                    # Warn if provided dim differs
                    if provided_input_dim is not None and provided_input_dim != actual_input_dim:
                        logger.warning(f"[A3X Bridge Handler - Create] Provided input_dim ({provided_input_dim}) for fragment '{fragment_id}' differs from embedding model dimension ({actual_input_dim}). USING {actual_input_dim}.")
                else:
                    raise ValueError("Failed to load embedding model.")
            except Exception as emb_err:
                logger.error(f"[A3X Bridge Handler - Create] Error: Could not determine input_dim from embedding model: {emb_err}")
                return { "status": "error", "message": f"Cannot create neural fragment: failed to get embedding dimension ({emb_err})." }
            
            # Determine num_classes (important for output layer)
            if num_classes is None:
                if id_to_label and isinstance(id_to_label, dict):
                    num_classes = len(id_to_label)
                    logger.info(f"[A3X Bridge Handler - Create] Inferred num_classes={num_classes} from provided id_to_label map.")
                else:
                     # Default to 3 if not specified and cannot be inferred
                     num_classes = 3 
                     logger.warning(f"[A3X Bridge Handler - Create] 'num_classes' not specified and cannot infer from id_to_label. Defaulting to {num_classes}.")
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
                    logger.info(f"[A3X Bridge Handler - Create] Determined actual input_dim={actual_input_dim} from embedding model for reflective fragment.")
                    # Warn if provided dim differs
                    if provided_input_dim is not None and provided_input_dim != actual_input_dim:
                        logger.warning(f"[A3X Bridge Handler - Create] Provided input_dim ({provided_input_dim}) for reflective fragment '{fragment_id}' differs from embedding model dimension ({actual_input_dim}). USING {actual_input_dim}.")
                else:
                    raise ValueError("Failed to load embedding model.")
            except Exception as emb_err:
                logger.error(f"[A3X Bridge Handler - Create] Error: Could not determine input_dim for reflective fragment: {emb_err}")
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
             logger.error(f"[A3X Bridge Handler - Create] Error: Unsupported fragment type '{fragment_type}'.")
             return { "status": "error", "message": f"Unsupported fragment type: {fragment_type}" }
        
        # --- Save the newly created fragment --- 
        logger.info(f"[A3X Bridge Handler - Create] Attempting to save fragment '{fragment_id}' instance: {fragment_instance}")
        memory_bank.save(fragment_id, fragment_instance)
        logger.info(f"[A3X Bridge Handler - Create] Successfully created and saved fragment '{fragment_id}' type '{fragment_type}'.")
        return { "status": "success", "message": f"Fragment '{fragment_id}' created.", "fragment_id": fragment_id }

    except Exception as e:
         logger.error(f"[A3X Bridge Handler - Create] Error creating fragment '{fragment_id}': {e}", exc_info=True)
         # Clean up potentially inconsistent state?
         memory_bank.delete(fragment_id, ignore_errors=True) # Attempt to remove if creation failed mid-way
         return { "status": "error", "message": f"Failed to create fragment '{fragment_id}': {e}" } 