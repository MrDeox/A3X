# a3x/core/semantic_indexer_unified.py

import os
import json
import logging
import sys
import time
import numpy as np

# --- Dependency Check and Imports ---
try:
    import faiss
except ImportError:
    print("FAISS library not found. Please install it (e.g., 'pip install faiss-cpu' or 'pip install faiss-gpu').", file=sys.stderr)
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers library not found. Please install it ('pip install sentence-transformers').", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2' # Model for generating embeddings
EMBEDDING_DIM = 384            # Dimension for all-MiniLM-L6-v2
BATCH_SIZE = 64                # Batch size for embedding generation

# --- Path Calculation ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up 2 levels from a3x/core/ to project root a3x/
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

INPUT_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "arthur_unified_dataset.jsonl")
OUTPUT_INDEX_PATH = os.path.join(PROJECT_ROOT, "memory.db.unified.vss_semantic_memory.faissindex")
OUTPUT_MAPPING_PATH = os.path.join(PROJECT_ROOT, "memory.db.unified.vss_semantic_memory.faissindex.mapping.json")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "semantic_indexer_unified.log")

# --- Logging Setup ---
logger = logging.getLogger("SemanticIndexerUnified")
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

# File Handler
try:
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    fh = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
except Exception as e:
    print(f"Error setting up file logger: {e}", file=sys.stderr)

# Console Handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(ch_formatter)
logger.addHandler(ch)

# --- Main Logic ---

def create_unified_index():
    """Loads unified data, generates embeddings, creates FAISS index, and saves results."""
    logger.info("--- Starting Unified Semantic Indexing ---")

    # 1. Load Data
    logger.info(f"Loading data from {INPUT_DATASET_PATH}...")
    records = []
    texts_to_embed = []
    metadata_map = {}
    try:
        with open(INPUT_DATASET_PATH, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    if record.get("text"):
                        # Store text for embedding
                        texts_to_embed.append(record["text"])
                        # Store metadata for mapping, keyed by the index (line number)
                        metadata_map[len(texts_to_embed) - 1] = {
                            "type": record.get("type", "unknown"),
                            "source": record.get("source", "unknown"),
                            "meta": record.get("meta", {})
                        }
                    else:
                        logger.warning(f"Record on line {line_num+1} missing 'text' field. Skipping.")
                except json.JSONDecodeError:
                    logger.error(f"Skipping invalid JSON on line {line_num+1} in {INPUT_DATASET_PATH}")
    except FileNotFoundError:
        logger.error(f"Input dataset not found: {INPUT_DATASET_PATH}")
        return
    except Exception as e:
        logger.error(f"Error reading input dataset {INPUT_DATASET_PATH}: {e}", exc_info=True)
        return

    if not texts_to_embed:
        logger.error("No valid records with 'text' found in the input dataset. Aborting.")
        return

    logger.info(f"Loaded {len(texts_to_embed)} records for indexing.")

    # 2. Load Model
    logger.info(f"Loading sentence transformer model: {MODEL_NAME}...")
    try:
        start_time = time.time()
        model = SentenceTransformer(MODEL_NAME)
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Failed to load sentence transformer model {MODEL_NAME}: {e}", exc_info=True)
        return

    # 3. Generate Embeddings
    logger.info(f"Generating embeddings for {len(texts_to_embed)} texts (Batch size: {BATCH_SIZE})...")
    try:
        start_time = time.time()
        embeddings = model.encode(texts_to_embed, batch_size=BATCH_SIZE, show_progress_bar=True)
        logger.info(f"Embeddings generated in {time.time() - start_time:.2f} seconds.")

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        expected_shape = (len(texts_to_embed), EMBEDDING_DIM)
        if embeddings.shape != expected_shape:
             logger.error(f"Unexpected embedding shape: {embeddings.shape}. Expected {expected_shape}")
             return

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
        return

    # 4. Create and Populate FAISS Index
    logger.info(f"Creating FAISS index (IndexFlatL2, Dim: {EMBEDDING_DIM})...")
    try:
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        index.add(embeddings)
        logger.info(f"FAISS index created and populated with {index.ntotal} vectors.")
    except Exception as e:
        logger.error(f"Failed to create or populate FAISS index: {e}", exc_info=True)
        return

    # 5. Save Index
    logger.info(f"Saving FAISS index to {OUTPUT_INDEX_PATH}...")
    try:
        os.makedirs(os.path.dirname(OUTPUT_INDEX_PATH), exist_ok=True)
        faiss.write_index(index, OUTPUT_INDEX_PATH)
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}", exc_info=True)
        return

    # 6. Save Mapping
    logger.info(f"Saving index mapping to {OUTPUT_MAPPING_PATH}...")
    try:
        os.makedirs(os.path.dirname(OUTPUT_MAPPING_PATH), exist_ok=True)
        with open(OUTPUT_MAPPING_PATH, 'w', encoding='utf-8') as f:
            # metadata_map already has the correct structure {index: {metadata}}
            json.dump(metadata_map, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save index mapping: {e}", exc_info=True)
        return

    logger.info("--- Unified Semantic Indexing Finished Successfully ---")
    logger.info(f"Index size: {index.ntotal} vectors")
    logger.info(f"Index saved to: {OUTPUT_INDEX_PATH}")
    logger.info(f"Mapping saved to: {OUTPUT_MAPPING_PATH}")
    logger.info(f"Log file: {LOG_FILE_PATH}")

if __name__ == "__main__":
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    create_unified_index() 