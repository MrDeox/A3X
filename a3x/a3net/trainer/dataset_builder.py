import torch
from torch import Tensor
from typing import List, Tuple, Dict, Any, Optional
import logging
import json
from pathlib import Path
import os # Keep os import for potential path operations if needed

# --- Add necessary imports ---
# Assume context store is accessible or passed somehow
# For now, let's import it directly (might need adjustment based on project structure)
try:
    #    from ..core.context_store import ContextStore # No longer needed for data loading here
    #    from ..integration.a3x_bridge import MEMORY_BANK # Used implicitly by context store potentially
    # Import sentence-transformers (NEEDS INSTALLATION: pip install sentence-transformers)
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"[Dataset Builder] Import Error: {e}. Ensure sentence-transformers are installed/accessible.")
    SentenceTransformer = None

logger = logging.getLogger(__name__)

# --- Global Embedding Model Cache ---
# Load model only once
embedding_model_cache: Optional[SentenceTransformer] = None
embedding_model_name = 'all-MiniLM-L6-v2' # Or choose another model
def get_embedding_model() -> Optional[SentenceTransformer]:
    global embedding_model_cache
    if embedding_model_cache is None and SentenceTransformer:
        try:
            logger.info(f"Loading sentence transformer model: {embedding_model_name}")
            embedding_model_cache = SentenceTransformer(embedding_model_name)
            logger.info("Sentence transformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model '{embedding_model_name}': {e}", exc_info=True)
            embedding_model_cache = None # Ensure it stays None on error
    return embedding_model_cache
# ------------------------------------

def build_dataset_from_context(context_id: str, num_classes: int) -> List[Tuple[Tensor, Tensor]]:
    """Builds a dataset based on a context ID (task name).
    
    Attempts to load examples from 'datasets/{context_id}.jsonl'.
    If not found, falls back to generating synthetic data.
    Requires 'sentence-transformers' library for embedding.
    """
    logger.info(f"[Dataset Builder] Attempting to build dataset for task: {context_id}")
    
    examples: Optional[List[Dict[str, Any]]] = None # Use Any for label flexibility initially
    dataset_dir = Path("a3x/a3net/datasets")
    # Sanitize context_id for filename safety
    safe_context_id = "".join(c for c in context_id if c.isalnum() or c in ('_', '-')).rstrip()
    if not safe_context_id:
        logger.error(f"Context ID '{context_id}' resulted in an invalid filename after sanitization. Cannot load dataset.")
        # Fallback to synthetic data directly
        examples = None
    else:
        dataset_file = dataset_dir / f"{safe_context_id}.jsonl"

        if dataset_file.is_file():
            logger.info(f"Found dataset file: {dataset_file}")
            examples = [] # <<< Initialize list here
            try:
                # --- REMOVED ContextStore loading block ---
                # No need to load from ContextStore here, we read the file directly.
                # -----------------------------------------

                with open(dataset_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        try:
                            line_content = line.strip()
                            if not line_content: continue # Skip empty lines

                            example_dict = json.loads(line_content)
                            # Check for required keys, label can be None
                            if isinstance(example_dict, dict) and "input" in example_dict and "label" in example_dict:
                                examples.append(example_dict)
                            else:
                                logger.warning(f"Skipping malformed example (missing keys 'input' or 'label') on line {i+1} in {dataset_file}")
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid JSON on line {i+1} in {dataset_file}: {line_content[:100]}...") # Log part of the line
                
                if examples:
                     logger.info(f"Loaded {len(examples)} valid examples from {dataset_file}.")
                else:
                     logger.warning(f"Dataset file {dataset_file} was empty or contained no valid examples.")
                     examples = None # Ensure examples is None if file was empty/invalid

            except Exception as e:
                logger.error(f"Failed to read or parse dataset file {dataset_file}: {e}", exc_info=True)
                examples = None # Ensure fallback on error
        else:
            logger.warning(f"Dataset file not found: {dataset_file}. Will generate synthetic data.")
            examples = None # Explicitly set to None if file not found

    # --- Process Loaded Examples --- 
    if examples and isinstance(examples, list) and len(examples) > 0:
        embedding_model = get_embedding_model()
        if not embedding_model:
             logger.error("Sentence embedding model not available. Cannot process real data. Falling back to synthetic.")
        else:
            processed_dataset: List[Tuple[Tensor, Tensor]] = []
            # --- Label Processing: Map unique labels (even complex ones via JSON string) to integers --- 
            unique_labels: Dict[str, int] = {}
            next_label_index = 0
            
            logger.info("Processing loaded examples into tensors...")
            # Extract valid inputs and labels
            valid_examples = [ex for ex in examples if isinstance(ex, dict) and "input" in ex and "label" in ex]
            input_texts = [str(ex["input"]) for ex in valid_examples]
            # Convert labels to string representation for consistent mapping (handles dicts, lists, etc.)
            labels_as_strings = []
            for ex in valid_examples:
                label_data = ex["label"]
                if isinstance(label_data, str):
                    labels_as_strings.append(label_data)
                else:
                    try:
                        # Use sorted keys for consistent JSON string representation of dicts
                        labels_as_strings.append(json.dumps(label_data, sort_keys=True))
                    except TypeError:
                         logger.warning(f"Label in example could not be JSON serialized, using repr(): {label_data!r}")
                         labels_as_strings.append(repr(label_data))


            if not input_texts:
                 logger.error("No valid input texts found after processing examples. Falling back to synthetic.")
                 examples = None # Trigger fallback
            else:
                try:
                     # Get embeddings in batches
                     embeddings = embedding_model.encode(input_texts, convert_to_tensor=True)
                     logger.info(f"Generated embeddings with shape: {embeddings.shape}")
                     
                     # Map labels to indices
                     label_indices = []
                     for label_str in labels_as_strings:
                         if label_str not in unique_labels:
                             unique_labels[label_str] = next_label_index
                             next_label_index += 1
                         label_indices.append(unique_labels[label_str])
                     
                     # Create final dataset list
                     for i in range(len(embeddings)):
                         label_tensor = torch.tensor(label_indices[i], dtype=torch.long)
                         processed_dataset.append((embeddings[i], label_tensor))
                     
                     # Log the found label mapping
                     logger.info(f"Created label mapping for task '{safe_context_id}': {unique_labels}")
                     num_found_classes = len(unique_labels)
                     logger.info(f"Number of unique labels found: {num_found_classes}")
                     
                     if num_found_classes == 0:
                          logger.error("No unique labels found in the dataset, even though examples exist. This indicates an issue. Falling back to synthetic.")
                          examples = None # Trigger fallback
                     else:
                        # Check against num_classes expected by the Neural Fragment
                        if num_found_classes > num_classes:
                            logger.error(f"CRITICAL: Found {num_found_classes} unique labels, but fragment expects only {num_classes}. Training will likely fail or produce incorrect results!")
                            # Decide on behavior: fallback or proceed with warning? For safety, fallback.
                            logger.warning("Falling back to synthetic data due to label mismatch.")
                            examples = None # Trigger fallback
                        elif num_found_classes < num_classes:
                             logger.warning(f"Found {num_found_classes} unique labels, but fragment expects {num_classes}. The model might not learn effectively for all potential classes.")
                             # Proceeding, but with a warning.

                        if examples: # Check if fallback wasn't triggered
                            logger.info(f"Successfully processed {len(processed_dataset)} real examples into tensors.")
                            return processed_dataset
                     
                except Exception as e:
                     logger.error(f"Error processing real examples: {e}", exc_info=True)
                     logger.warning("Falling back to synthetic data due to processing error.")
                     examples = None # Ensure fallback on error

    # --- Fallback to Synthetic Data --- 
    # This block is reached if examples is None or processing failed
    logger.warning(f"[Dataset Builder] No suitable real examples processed for task '{safe_context_id}'. Generating synthetic dataset.")
    num_samples = 500 # Or make this configurable
    embedding_model = get_embedding_model() # Try to get model again for dim
    input_dim = embedding_model.get_sentence_embedding_dimension() if embedding_model else 128 
    if not embedding_model: logger.warning(f"Embedding model not loaded, using default input_dim={input_dim}")
    
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(num_samples):
        x = torch.randn(input_dim)
        # Ensure y is within the expected num_classes for the fragment
        y_val = torch.randint(0, num_classes, ()).item()
        y = torch.tensor(y_val, dtype=torch.long)
        dataset.append((x, y))
        
    logger.info(f"[Dataset Builder] Generated {len(dataset)} synthetic samples with {num_classes} classes.")
    return dataset

# Example usage (optional, for testing):
if __name__ == '__main__':
    dummy_context_id = "task_123_feedback"
    # Note: Running this standalone requires the 'datasets' dir and potentially a dummy file
    # Also requires sentence-transformers installed
    logging.basicConfig(level=logging.INFO)
    
    # Ensure dummy dataset dir exists for testing
    dummy_dir = Path("a3x/a3net/datasets")
    dummy_dir.mkdir(parents=True, exist_ok=True)
    dummy_file = dummy_dir / f"{dummy_context_id}.jsonl"
    # Create a dummy file if it doesn't exist for the test
    if not dummy_file.exists():
        with open(dummy_file, 'w') as f:
            f.write('{"input": "test input 1", "label": "A"}\n')
            f.write('{"input": "test input 2", "label": "B"}\n')
            f.write('{"input": "test input 3", "label": "A"}\n')
        print(f"Created dummy dataset file: {dummy_file}")
    
    # Test with 2 expected classes (A, B)
    real_or_synthetic_data = build_dataset_from_context(dummy_context_id, 2)
    
    # Check the first sample
    if real_or_synthetic_data:
        x_sample, y_sample = real_or_synthetic_data[0]
        print(f"\nFirst sample:")
        print(f"  Input shape: {x_sample.shape}")
        print(f"  Output shape: {y_sample.shape}")
        # print(f"  Input data (first 5 elements): {x_sample[:5]}...")
        # print(f"  Output data: {y_sample}") 

    # Clean up dummy file after test (optional)
    # os.remove(dummy_file)
    # print(f"Removed dummy dataset file: {dummy_file}") 