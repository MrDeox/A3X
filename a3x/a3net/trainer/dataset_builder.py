import torch
from torch import Tensor
from typing import List, Tuple, Dict, Any, Optional
import logging
import json
from pathlib import Path
import os # Keep os import for potential path operations if needed
import asyncio

from ..core.context_store import ContextStore

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
def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:
    global embedding_model_cache
    if embedding_model_cache is None and SentenceTransformer:
        try:
            logger.info(f"Loading sentence transformer model: {model_name}")
            embedding_model_cache = SentenceTransformer(model_name)
            logger.info("Sentence transformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model '{model_name}': {e}", exc_info=True)
            embedding_model_cache = None # Ensure it stays None on error
    return embedding_model_cache
# ------------------------------------

async def build_dataset_from_context(
    fragment_id: str,
    context_store: ContextStore,
    model: Any,
    max_samples: int = 100
) -> List[Tuple[Tensor, Tensor]]:
    """Builds a training dataset from relevant context entries."""
    try:
        # Get relevant context entries
        keys = await context_store.find_keys_by_tag(fragment_id)
        if not keys:
            logger.warning(f"No context entries found for fragment {fragment_id}")
            return []
            
        # Limit number of samples
        keys = keys[:max_samples]
        
        dataset = []
        for key in keys:
            try:
                # Get entry data
                entry = await context_store.get(key)
                if not entry:
                    continue
                    
                # Get text and label from context
                data = entry if isinstance(entry, dict) else json.loads(entry)
                text = data.get('text', '')
                label = data.get('label', '')
                
                if not text or not label:
                    continue
                    
                # Convert label to tensor
                label_map = {"SIM": 0, "NÃO": 1, "REAVALIAR": 2}
                if label not in label_map:
                    logger.warning(f"Invalid label {label} in entry {key}")
                    continue
                    
                label_tensor = torch.tensor(label_map[label])
                
                # Get embedding
                if model:
                    embedding = model.encode(text, convert_to_tensor=True)
                else:
                    # For testing, use random embedding
                    embedding = torch.randn(384)
                    
                dataset.append((embedding, label_tensor))
                
            except Exception as e:
                logger.error(f"Error processing context entry {key}: {e}", exc_info=True)
                continue
                
        logger.info(f"Built dataset with {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        logger.error(f"Error building dataset: {e}", exc_info=True)
        return []

async def create_or_update_dataset_jsonl(task_name: str, examples: List[Dict[str, str]], append: bool = True):
    """Creates or appends examples to a JSONL dataset file for a specific task."""
    dataset_dir = Path("data/datasets/a3net") 
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = dataset_dir / f"{task_name}.jsonl"
    
    mode = 'a' if append and dataset_file.exists() else 'w'
    with open(dataset_file, mode) as f:
        for example in examples:
            json.dump(example, f)
            f.write('\n')
            
    logger.info(f"Saved {len(examples)} examples to {dataset_file}")

# The __main__ block causing the SyntaxError has been removed. 