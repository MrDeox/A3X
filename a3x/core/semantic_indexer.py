# a3x/core/semantic_indexer.py

"""
Cria um índice semântico usando sentence-transformers e FAISS.
Lê os registros do dataset e gera embeddings para indexação.
"""

import json
import os
from sentence_transformers import SentenceTransformer
# Ensure faiss is installed: pip install faiss-cpu # or faiss-gpu
try:
    import faiss
except ImportError:
    print("FAISS library not found. Please install it: pip install faiss-cpu")
    exit(1)
import numpy as np

# Define paths relative to the project root for clarity
# Assumes the script might be run from different locations (e.g., root or via python -m)
try:
    # Assumes script is in a3x/core/
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive interpreter)
    PROJECT_ROOT = os.path.abspath('.')

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "arthur_decision_dataset.jsonl")
# Using the user-specified filename, placing it relative to the project root
# This path might be unconventional; consider placing it inside `.a3x/` or `data/`
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "memory.db.main.vss_semantic_memory.faissindex")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def load_dataset(path=DATASET_PATH):
    """Loads the JSONL dataset and returns a list of records."""
    records = []
    # Ensure the directory exists before trying to open the file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Warning: Dataset file not found at {path}. Creating an empty one.")
        with open(path, "w", encoding="utf-8") as f:
            pass # Create an empty file if it doesn't exist
        return records # Return empty list as the file is new and empty

    print(f"Loading dataset from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: # Skip empty lines
                continue
            try:
                record = json.loads(line)
                if record: # Ensure the loaded record is not empty
                    # Optionally add an ID if needed later, though FAISS might map by position
                    # record['original_index'] = i
                    records.append(record)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {i+1}: {line}. Error: {e}")
                continue # Skip malformed lines
    print(f"Loaded {len(records)} records.")
    return records

def prepare_text(record):
    """Prepares a single string representation from a record for embedding."""
    parts = []
    # Flexible key checking based on potential dataset structures
    possible_keys = ["input", "context", "arthur_response", "reasoning", "user_input", "decision", "text", "query", "response", "thought"]
    for key in possible_keys:
        if key in record and record[key]:
            value = record[key]
            # Handle lists by joining elements
            if isinstance(value, list):
                value = " ".join(map(str, value))
            # Convert non-strings to string, ensuring it's not None
            elif not isinstance(value, str):
                 value = str(value)

            if value and value.strip(): # Append only if there's non-whitespace content
                 parts.append(value.strip())

    return " ".join(parts).strip() # Use strip() to avoid leading/trailing whitespace

def build_embeddings(records, model):
    """Generates embeddings for the prepared texts from records."""
    texts = [prepare_text(r) for r in records]
    # Filter out any empty strings that might result from prepare_text
    valid_texts_with_indices = [(i, text) for i, text in enumerate(texts) if text]

    if not valid_texts_with_indices:
        print("No valid text content found in records to generate embeddings.")
        return None, [] # Return None for embeddings, empty list for indices

    original_indices = [i for i, text in valid_texts_with_indices]
    valid_texts = [text for i, text in valid_texts_with_indices]

    print(f"Generating embeddings for {len(valid_texts)} non-empty text entries...")

    embeddings = model.encode(valid_texts, show_progress_bar=True, convert_to_numpy=True)
    # Ensure float32 for FAISS
    embeddings_float32 = embeddings.astype(np.float32)
    print(f"Embeddings generated with shape: {embeddings_float32.shape}")
    return embeddings_float32, original_indices

def create_faiss_index(embeddings):
    """Creates a FAISS index from the provided embeddings."""
    if embeddings is None or embeddings.shape[0] == 0:
        print("No embeddings provided to create the FAISS index.")
        return None
    if embeddings.ndim != 2:
        print(f"Embeddings have incorrect dimensions ({embeddings.ndim}D). Expected 2D.")
        return None

    dim = embeddings.shape[1]
    print(f"Creating FAISS index (IndexFlatL2) with {embeddings.shape[0]} vectors of dimension {dim}.")
    # Using IndexFlatL2 as specified. Simple and good for moderate datasets.
    # Consider IndexIVFFlat for larger datasets.
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index created successfully. Total vectors in index: {index.ntotal}")
    return index

def save_faiss_index(index, path=FAISS_INDEX_PATH):
    """Saves the FAISS index to the specified path."""
    if index is None:
        print("No FAISS index object provided to save.")
        return
    try:
        # Ensure the directory for the index exists
        index_dir = os.path.dirname(path)
        os.makedirs(index_dir, exist_ok=True)
        print(f"Saving FAISS index to: {path}")
        faiss.write_index(index, path)
        print("FAISS index saved successfully.")
    except Exception as e:
        print(f"Error saving FAISS index to {path}: {e}")

def main():
    print("--- Starting Semantic Indexer ---")
    # Paths are defined globally relative to the project root

    # 1. Load Dataset
    records = load_dataset(DATASET_PATH)
    if not records:
        print("Dataset is empty or could not be loaded. No index will be generated.")
        print("--- Semantic Indexer Finished ---")
        return

    # 2. Load Embedding Model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        # Specify cache directory for models if desired, e.g., cache_folder='./hf_cache'
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded successfully.")
    except Exception as e:
        print(f"Error loading SentenceTransformer model '{EMBEDDING_MODEL_NAME}': {e}")
        print("--- Semantic Indexer Finished ---")
        return

    # 3. Generate Embeddings
    print("Generating embeddings for dataset records...")
    embeddings, original_indices = build_embeddings(records, model)

    if embeddings is None:
        print("Failed to generate embeddings. Aborting index creation.")
        print("--- Semantic Indexer Finished ---")
        return
    if len(original_indices) != embeddings.shape[0]:
         print(f"Warning: Mismatch between number of embeddings ({embeddings.shape[0]}) and original indices mapping ({len(original_indices)}).")
         # This case should ideally not happen with the current logic, but good to check.

    # 4. Create FAISS Index
    print("Creating FAISS index...")
    index = create_faiss_index(embeddings)

    if index is None:
        print("Failed to create FAISS index. Aborting save.")
        print("--- Semantic Indexer Finished ---")
        return

    # 5. Save FAISS Index
    # The index contains embeddings only for records that had valid text.
    # The `original_indices` list maps the position in the FAISS index back to the
    # index in the originally loaded `records` list.
    # The `simulate_arthur_response` skill will need this mapping.
    # We might need to save `original_indices` alongside the FAISS index.
    # For now, just saving the FAISS index as requested.
    save_faiss_index(index, FAISS_INDEX_PATH)

    # Optional: Save the mapping from FAISS index position to original record index
    mapping_path = FAISS_INDEX_PATH + ".mapping.json"
    try:
        mapping_data = { "faiss_to_original_record_index": original_indices }
        with open(mapping_path, 'w', encoding='utf-8') as f_map:
            json.dump(mapping_data, f_map)
        print(f"Saved FAISS index to original record mapping to: {mapping_path}")
    except Exception as e:
        print(f"Warning: Could not save index mapping to {mapping_path}: {e}")


    print("--- Semantic Indexer Finished Successfully ---")

if __name__ == "__main__":
    main() 