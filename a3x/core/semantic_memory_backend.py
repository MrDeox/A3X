import faiss
import numpy as np
import json
import os
import logging
from typing import List, Dict, Any, Optional

# Configure logger for this module
logger = logging.getLogger(__name__)

# Default directory for storing indexes relative to the execution path (adjust if needed)
# It's generally better if the caller specifies the full path including the desired directory.
# INDEX_DIR = "data/memory/indexes" 

def _get_full_paths(index_path_base: str) -> tuple[str, str]:
    """Constructs full paths for index and metadata files."""
    index_file = index_path_base + ".index"
    metadata_file = index_path_base + ".index.jsonl"
    return index_file, metadata_file

def init_index(index_path_base: str, embedding_dim: int = 768) -> Optional[faiss.Index]:
    """
    Initializes or loads a FAISS index from the specified path.
    Creates the directory if it doesn't exist.

    Args:
        index_path_base: The base path for the index (e.g., "data/memory/indexes/semantic_memory").
                         '.index' will be appended for the FAISS file.
        embedding_dim: The dimension of the embeddings (required if creating a new index).

    Returns:
        A loaded or newly created faiss.Index object, or None if an error occurs.
    """
    index_file, _ = _get_full_paths(index_path_base)
    index_dir = os.path.dirname(index_file)

    try:
        # Ensure the directory exists
        os.makedirs(index_dir, exist_ok=True)
        logger.info(f"Ensured index directory exists: {index_dir}")

        if os.path.exists(index_file):
            logger.info(f"Loading existing FAISS index from: {index_file}")
            index = faiss.read_index(index_file)
            # Sanity check for dimension if index loaded
            if index.d != embedding_dim:
                 logger.warning(f"Loaded index dimension ({index.d}) does not match expected dimension ({embedding_dim}).")
                 # Decide how to handle: raise error, return None, re-initialize? Returning None for now.
                 # raise ValueError(f"Index dimension mismatch: loaded {index.d}, expected {embedding_dim}")
                 return None 
            logger.info(f"Successfully loaded index with {index.ntotal} vectors.")
            return index
        else:
            logger.info(f"Creating new FAISS index (IndexFlatL2) with dimension {embedding_dim} at: {index_file}")
            index = faiss.IndexFlatL2(embedding_dim)
            # Note: Index is not saved here, only created in memory. 
            # Saving happens after adding data.
            return index
    except FileNotFoundError:
         logger.error(f"Index file not found during load attempt (should not happen if os.path.exists passed): {index_file}", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"Failed to initialize or load FAISS index at {index_file}: {e}", exc_info=True)
        return None

def add_to_index(index_path_base: str, embedding: List[float], metadata: Dict[str, Any], embedding_dim: int = 768):
    """
    Adds a single embedding and its corresponding metadata to the index and metadata file.
    Initializes the index if it doesn't exist. Saves the index after adding.

    Args:
        index_path_base: The base path for the index files.
        embedding: The embedding vector (as a list of floats).
        metadata: A dictionary containing metadata associated with the embedding.
        embedding_dim: The dimension required if the index needs to be created.
    """
    index = init_index(index_path_base, embedding_dim)
    if index is None:
        logger.error("Failed to add to index because index initialization failed.")
        return # Or raise an exception

    index_file, metadata_file = _get_full_paths(index_path_base)

    try:
        # Prepare embedding for FAISS (needs numpy array, float32)
        np_embedding = np.array([embedding]).astype('float32')
        if np_embedding.shape[1] != index.d:
             logger.error(f"Embedding dimension mismatch: Index dimension is {index.d}, provided embedding has dimension {np_embedding.shape[1]}.")
             return # Or raise

        # Add embedding to FAISS index
        index.add(np_embedding)
        vector_id = index.ntotal - 1 # ID is the new total count minus 1
        logger.debug(f"Added embedding to FAISS index. New total: {index.ntotal}")

        # Append metadata to the parallel JSONL file
        try:
            with open(metadata_file, 'a', encoding='utf-8') as f:
                metadata['_vector_id'] = vector_id
                json_line = json.dumps(metadata, ensure_ascii=False)
                f.write(json_line + '\n')
            logger.debug(f"Appended metadata to {metadata_file}")
        except IOError as e:
             logger.error(f"Failed to write metadata to {metadata_file}: {e}", exc_info=True)
             # Critical decision: Should we rollback the FAISS add if metadata fails?
             # For simplicity now, we don't rollback, but this could lead to inconsistency.
             # Consider implementing rollback or a transactional approach for production.
             # index.remove_ids(np.array([vector_id])) # Example rollback (if supported and desired)
             return # Or raise

        # Save the updated index to disk
        faiss.write_index(index, index_file)
        logger.info(f"Successfully added embedding and metadata. Index saved to {index_file}")

    except Exception as e:
        logger.error(f"Failed to add embedding/metadata to index {index_path_base}: {e}", exc_info=True)
        # Consider rollback here as well if applicable

def search_index(index_path_base: str, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Searches the FAISS index for the top_k nearest neighbors to the query embedding.

    Args:
        index_path_base: The base path for the index files.
        query_embedding: The query embedding vector (as a list of floats).
        top_k: The number of nearest neighbors to retrieve.

    Returns:
        A list of dictionaries, where each dictionary contains:
        - 'distance': The L2 distance to the query embedding.
        - 'metadata': The metadata associated with the retrieved embedding.
        Returns an empty list if the index doesn't exist or an error occurs.
    """
    index_file, metadata_file = _get_full_paths(index_path_base)
    results = []

    try:
        # 1. Load the FAISS index
        if not os.path.exists(index_file):
            logger.warning(f"Search attempted, but index file does not exist: {index_file}")
            return []
        index = faiss.read_index(index_file)
        logger.info(f"Loaded index from {index_file} for search ({index.ntotal} vectors).")

        # 2. Load the metadata (Line by Line)
        metadata_list = []
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line: # Skip empty lines
                            continue
                        try:
                            metadata_list.append(json.loads(line))
                        except json.JSONDecodeError as json_err:
                            logger.warning(f"Skipping invalid JSON line {line_num + 1} in {metadata_file}: {json_err} - Line: '{line}'")
                logger.info(f"Loaded {len(metadata_list)} metadata entries from {metadata_file}")
            except IOError as e:
                logger.error(f"Failed to read metadata file {metadata_file}: {e}", exc_info=True)
                return [] # Cannot proceed without metadata
        else:
            logger.warning(f"Search attempted, but metadata file does not exist: {metadata_file}")
            return [] 

        # Check consistency (optional but recommended)
        if index.ntotal != len(metadata_list):
            logger.error(f"Inconsistency detected: FAISS index has {index.ntotal} vectors, but metadata file has {len(metadata_list)} entries.")
            # Decide how to handle: return empty, try to recover, etc. Returning empty for safety.
            return []

        # 3. Prepare query embedding
        np_query_embedding = np.array([query_embedding]).astype('float32')
        if np_query_embedding.shape[1] != index.d:
             logger.error(f"Query embedding dimension mismatch: Index dimension is {index.d}, query has dimension {np_query_embedding.shape[1]}.")
             return []

        # Adjust k if the index has fewer vectors than requested top_k
        actual_k = min(top_k, index.ntotal)
        if actual_k == 0:
            logger.warning("Search attempted on an empty index.")
            return []
        if actual_k < top_k:
            logger.warning(f"Requested top_k={top_k}, but index only contains {index.ntotal} vectors. Searching for k={actual_k}.")


        # 4. Perform the search
        distances, indices = index.search(np_query_embedding, actual_k)
        logger.info(f"FAISS search completed. Found {len(indices[0])} results.")

        # 5. Combine results with metadata
        for i, idx in enumerate(indices[0]):
            if idx == -1: # FAISS uses -1 for invalid indices (e.g., if k > ntotal)
                 logger.warning(f"FAISS search returned invalid index -1 at position {i}. Skipping.")
                 continue
            if 0 <= idx < len(metadata_list):
                results.append({
                    "distance": float(distances[0][i]), # Ensure standard float type
                    "metadata": metadata_list[idx]
                })
            else:
                # This indicates a serious inconsistency if the earlier check passed
                logger.error(f"Search returned index {idx} which is out of bounds for metadata list (size {len(metadata_list)}).")

    except FileNotFoundError:
         # This specific case should be caught by the os.path.exists check above
         logger.error(f"Index file not found during search: {index_file}", exc_info=True)
         return []
    except Exception as e:
        logger.error(f"Failed to search FAISS index {index_path_base}: {e}", exc_info=True)
        return [] # Return empty list on error

    logger.info(f"Returning {len(results)} search results.")
    return results

# Example Usage (Optional - can be commented out or removed)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     
#     # Define path and dimension
#     idx_path = "data/memory/indexes/test_semantic_memory"
#     dim = 4 # Example dimension
#     
#     # Initialize (creates if not exists)
#     index_obj = init_index(idx_path, embedding_dim=dim)
#     
#     if index_obj:
#         # Add some data
#         add_to_index(idx_path, [0.1, 0.2, 0.3, 0.4], {"id": "doc1", "content": "This is the first document."}, embedding_dim=dim)
#         add_to_index(idx_path, [0.5, 0.6, 0.7, 0.8], {"id": "doc2", "content": "This is the second document, quite different."}, embedding_dim=dim)
#         add_to_index(idx_path, [0.11, 0.22, 0.33, 0.44], {"id": "doc3", "content": "A third document, similar to the first."}, embedding_dim=dim)
#         
#         # Search for vectors similar to the first document
#         query_vec = [0.1, 0.21, 0.31, 0.41]
#         search_results = search_index(idx_path, query_vec, top_k=2)
#         
#         logger.info("\\n--- Search Results ---")
#         for res in search_results:
#             logger.info(f"Distance: {res['distance']:.4f}, Metadata: {res['metadata']}")
#         logger.info("--------------------\\n") 