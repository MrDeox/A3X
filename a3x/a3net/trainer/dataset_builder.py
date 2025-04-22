import torch
from torch import Tensor
from typing import List, Tuple, Dict, Any, Optional
import logging
import json
from pathlib import Path
import os # Keep os import for potential path operations if needed
import asyncio

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

def build_dataset_from_context(fragment_id: str, context_store: ContextStore, model: Any, max_samples: int = 100) -> List[Tuple[Tensor, Tensor]]:
    """Builds a training dataset from relevant context entries."""
    # ... (Implementation details)

async def create_or_update_dataset_jsonl(task_name: str, examples: List[Dict[str, str]], append: bool = True):
    """Creates or appends examples to a JSONL dataset file for a specific task."""
    # Use the new data structure path
    dataset_dir = Path("data/datasets/a3net") 
    dataset_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    dataset_file = dataset_dir / f"{task_name}.jsonl"
    # ... existing code ...

# The __main__ block causing the SyntaxError has been removed. 