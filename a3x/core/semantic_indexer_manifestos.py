# a3x/core/semantic_indexer_manifestos.py

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
    # Assumes script is in a3x/core/SOMETHING/
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up 2 levels to get to project root (a3x/core/SOMETHING -> a3x/core -> a3x -> project_root)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) # Corrected level
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

INPUT_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "arthur_manifest_dataset.jsonl")
OUTPUT_INDEX_PATH = os.path.join(PROJECT_ROOT, "memory.db.manifest.vss_semantic_memory.faissindex")
OUTPUT_MAPPING_PATH = os.path.join(PROJECT_ROOT, "memory.db.manifest.vss_semantic_memory.faissindex.mapping.json")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "semantic_indexer_manifestos.log")

# --- Logging Setup ---
logger = logging.getLogger("SemanticIndexerManifestos")
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

def create_manifest_index():
    """Loads manifesto data, generates embeddings, creates FAISS index, and saves results."""
    logger.info("--- Starting Manifesto Semantic Indexing ---")

    # 1. Load Data
    logger.info(f"Loading data from {INPUT_DATASET_PATH}...")
    records = []
    texts_to_embed = []
    try:
        with open(INPUT_DATASET_PATH, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    if record.get("text"):
                        records.append(record)
                        texts_to_embed.append(record["text"])
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

    if not records:
        logger.error("No valid records found in the input dataset. Aborting.")
        return

    logger.info(f"Loaded {len(records)} records for indexing.")

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

        # Ensure embeddings are float32 numpy arrays
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if embeddings.shape[0] != len(records) or embeddings.shape[1] != EMBEDDING_DIM:
             logger.error(f"Unexpected embedding shape: {embeddings.shape}. Expected ({len(records)}, {EMBEDDING_DIM})")
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

    # 6. Create and Save Mapping
    logger.info(f"Creating and saving index mapping to {OUTPUT_MAPPING_PATH}...")
    mapping = {}
    for i, record in enumerate(records):
        # Store essential info to locate the original text chunk
        mapping[i] = {
            "source": record.get("source", "unknown"),
            "chunk_index": record.get("chunk_index", -1),
            # Optionally store the text itself if needed, but increases mapping size
            # "text": record.get("text", "")
        }

    try:
        os.makedirs(os.path.dirname(OUTPUT_MAPPING_PATH), exist_ok=True)
        with open(OUTPUT_MAPPING_PATH, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save index mapping: {e}", exc_info=True)
        return

    logger.info("--- Manifesto Semantic Indexing Finished Successfully ---")
    logger.info(f"Index size: {index.ntotal} vectors")
    logger.info(f"Index saved to: {OUTPUT_INDEX_PATH}")
    logger.info(f"Mapping saved to: {OUTPUT_MAPPING_PATH}")
    logger.info(f"Log file: {LOG_FILE_PATH}")

if __name__ == "__main__":
    # Add project root to sys.path if needed (for potential imports later)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    create_manifest_index() 