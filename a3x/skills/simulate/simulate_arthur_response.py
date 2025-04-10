# a3x/skills/simulate/simulate_arthur_response.py

"""
Skill para simular a resposta de Arthur com base em dados passados (unificados).
Utiliza embeddings e busca por similaridade com FAISS.
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    import faiss
except ImportError:
    # Allow the module to be imported, but the skill will fail gracefully if FAISS is needed.
    faiss = None
from a3x.core.tools import skill

# --- Configuration ---
try:
    # Assumes skill file is in a3x/skills/simulate/
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive execution)
    PROJECT_ROOT = os.path.abspath('.')

# Paths relative to project root - USING UNIFIED RESOURCES
UNIFIED_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "arthur_unified_dataset.jsonl")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "memory.db.unified.vss_semantic_memory.faissindex")
MAPPING_PATH = os.path.join(PROJECT_ROOT, "memory.db.unified.vss_semantic_memory.faissindex.mapping.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5 # Number of similar examples to retrieve

# --- Resource Caching ---
# Caches to avoid reloading resources on every skill call within the same agent run
model_cache = None
index_cache = None
mapping_cache = None
unified_dataset_cache = None
resources_loaded = False
load_error_message = None

def load_resources(logger):
    """Loads model, FAISS index, mapping, and unified dataset into cache. Returns success status and error message."""
    global model_cache, index_cache, mapping_cache, unified_dataset_cache, resources_loaded, load_error_message

    if resources_loaded:
        return True, load_error_message

    if faiss is None:
         load_error_message = "FAISS library is not installed. Please run 'pip install faiss-cpu'."
         logger.error(load_error_message)
         resources_loaded = True # Mark as attempted
         return False, load_error_message

    logger.info("--- Loading Simulate Arthur Response Skill Resources (Unified Index) ---")
    load_success = True

    # 1. Load Embedding Model
    if model_cache is None:
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
            # Define cache folder within project if desired, e.g.:
            # cache_folder = os.path.join(PROJECT_ROOT, ".cache", "sentence_transformers")
            # model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=cache_folder)
            model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
            load_error_message = f"Error loading embedding model: {e}"
            load_success = False

    # 2. Load Unified Dataset (Needed to get text from mapping info)
    if unified_dataset_cache is None and load_success:
        try:
            logger.info(f"Loading unified dataset from: {UNIFIED_DATASET_PATH}")
            if not os.path.exists(UNIFIED_DATASET_PATH):
                logger.error(f"Unified dataset file not found at {UNIFIED_DATASET_PATH}. Run the unified dataset builder script.")
                load_error_message = "Unified dataset file missing. Please run the builder."
                load_success = False
                unified_dataset_cache = []
            else:
                with open(UNIFIED_DATASET_PATH, "r", encoding="utf-8") as f:
                    # Read into a list for indexed access based on mapping
                    unified_dataset_cache = [json.loads(line) for line in f if line.strip()]
                logger.info(f"Loaded {len(unified_dataset_cache)} records from unified dataset.")
                if not unified_dataset_cache:
                    logger.warning("Unified dataset is empty. Simulation quality will be low.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON in unified dataset file {UNIFIED_DATASET_PATH}: {e}", exc_info=True)
            load_error_message = "Error reading unified dataset file (invalid JSON)."
            load_success = False
        except Exception as e:
            logger.error(f"Failed to load unified dataset: {e}", exc_info=True)
            load_error_message = f"Error loading unified dataset: {e}"
            load_success = False

    # 3. Load FAISS Index
    if index_cache is None and load_success:
        try:
            logger.info(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
            if not os.path.exists(FAISS_INDEX_PATH):
                logger.error(f"FAISS index file not found at {FAISS_INDEX_PATH}. Run the unified semantic indexer script.")
                load_error_message = "FAISS index missing. Please run the unified indexer."
                load_success = False # Index is essential
                index_cache = None
            else:
                 index_cache = faiss.read_index(FAISS_INDEX_PATH)
                 logger.info(f"FAISS index loaded. Contains {index_cache.ntotal} vectors.")
                 if index_cache.ntotal == 0:
                     logger.warning("FAISS index is empty. Simulation quality will be low.")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}", exc_info=True)
            load_error_message = f"Error loading FAISS index: {e}"
            load_success = False

    # 4. Load Mapping File (only if index was loaded successfully)
    if mapping_cache is None and load_success and index_cache is not None:
        try:
            logger.info(f"Loading index mapping from: {MAPPING_PATH}")
            if not os.path.exists(MAPPING_PATH):
                logger.error(f"FAISS index mapping file not found at {MAPPING_PATH}. Run the unified semantic indexer script.")
                load_error_message = "FAISS index mapping file missing. Please run the unified indexer."
                load_success = False # Mapping is crucial if index exists
                mapping_cache = None
            else:
                with open(MAPPING_PATH, "r", encoding="utf-8") as f:
                    mapping_cache = json.load(f) # Loads the dict {index_str: metadata}

                # Validate mapping format (basic check)
                if not isinstance(mapping_cache, dict):
                    logger.error("Mapping file is not a valid JSON dictionary.")
                    load_error_message = "Invalid mapping file format (not a dictionary)."
                    load_success = False
                    mapping_cache = None
                elif not mapping_cache and index_cache.ntotal > 0: # Check if empty when index isn't
                     logger.error(f"Mapping file {MAPPING_PATH} is empty, but index contains {index_cache.ntotal} vectors.")
                     load_error_message = "Mapping file is empty but index is not."
                     load_success = False
                elif mapping_cache: # If not empty, perform further checks
                     logger.info(f"Loaded mapping for {len(mapping_cache)} index entries.")
                     # Check consistency between index size and mapping size
                     if index_cache.ntotal != len(mapping_cache):
                         logger.warning(f"Index size ({index_cache.ntotal}) and mapping size ({len(mapping_cache)}) mismatch! Index and mapping may need regeneration.")
                         # Allow to continue but log warning.
                     else:
                         # Optional: Further validation of the first mapping entry structure
                         try:
                             first_key = next(iter(mapping_cache)) # Get first key (string index)
                             first_entry = mapping_cache[first_key]
                             if not all(k in first_entry for k in ["type", "source", "meta"]):
                                 logger.warning("First mapping entry structure might be incorrect. Expected keys: type, source, meta.")
                         except StopIteration:
                             # This case means mapping_cache was loaded but somehow became empty, shouldn't happen if checks above work
                             logger.warning("Mapping cache loaded but appears empty during validation.")
                         except Exception as val_e:
                              logger.warning(f"Error during mapping structure validation: {val_e}")


        except json.JSONDecodeError as e:
             logger.error(f"Error decoding JSON in mapping file {MAPPING_PATH}: {e}", exc_info=True)
             load_error_message = "Error reading mapping file (invalid JSON)."
             load_success = False
             mapping_cache = None
        except Exception as e:
            logger.error(f"Failed to load or parse index mapping: {e}", exc_info=True)
            load_error_message = f"Error loading index mapping: {e}"
            load_success = False

    # Final consistency check: Index, Mapping, and Unified Dataset sizes
    if load_success and index_cache is not None and mapping_cache is not None and unified_dataset_cache is not None:
        index_size = index_cache.ntotal
        # Ensure mapping_cache is a dict before len()
        mapping_size = len(mapping_cache) if isinstance(mapping_cache, dict) else -1
        dataset_size = len(unified_dataset_cache) if isinstance(unified_dataset_cache, list) else -1

        # Only warn if sizes are valid and non-zero but mismatch
        # Or if index has items but others don't
        if (index_size > 0 and (mapping_size <= 0 or dataset_size <= 0)) or \
           (index_size >= 0 and mapping_size >= 0 and dataset_size >= 0 and not (index_size == mapping_size == dataset_size)):
            logger.warning(f"Size mismatch: Index({index_size}), Mapping({mapping_size}), UnifiedDataset({dataset_size}). Files may be out of sync. Consider regenerating index and mapping.")
            # Allow skill to run but warn user.

    if load_success:
         logger.info("--- Simulate Arthur Response Skill Resources Loaded Successfully (Unified Index) ---")
         load_error_message = None # Clear errors if overall success
    else:
         # Ensure a load error message exists if load_success is False
         if not load_error_message: load_error_message = "Unknown error during resource loading."
         logger.error(f"--- Failed to Load Simulate Arthur Response Skill Resources (Unified Index): {load_error_message} ---")

    resources_loaded = True # Mark as attempted load
    return load_success, load_error_message


def prepare_record_text_for_prompt(record, logger):
    """Formats a record from the unified dataset into a string for the LLM prompt."""
    record_type = record.get("type")
    source = record.get("source", "unknown")
    meta = record.get("meta", {})
    text = record.get("text", "").strip() # Text from the unified dataset record

    if not text and record_type != "whatsapp": # WhatsApp can derive from meta
        logger.debug(f"Record from {source} (Type: {record_type}) has empty text field. Skipping formatting.")
        return ""

    if record_type == "whatsapp":
        # Use meta fields for clarity in prompt, fallback to main 'text' if needed
        context = meta.get("context", "").strip()
        response = meta.get("arthur_response", "").strip() # This *should* match record['text']
        if not context or not response:
             logger.warning(f"WhatsApp record from {source} missing context or response in meta. Falling back to text field: '{text[:50]}...'")
             # Use the main text field from the record if meta is incomplete
             return f"Registro (WhatsApp - {source}): {text}" if text else ""
        return f"Exemplo (WhatsApp - {source}):\n  Contexto: {context}\n  Resposta do Arthur: {response}"
    elif record_type == "manifesto":
        chunk_idx = meta.get("chunk_index", "N/A")
        # Text comes directly from the record['text']
        return f"Exemplo (Manifesto - {source} Chunk {chunk_idx}):\n  Texto: {text}"
    else:
        # Fallback for unknown or missing type
        logger.warning(f"Record from {source} has unknown or missing type: '{record_type}'. Using raw text.")
        return f"Registro ({source}): {text}" if text else ""

# --- Skill Definition ---

@skill(
    name="simulate_arthur_response",
    description="Simula como Arthur responderia a uma determinada entrada, consultando seu histórico unificado (manifestos, conversas) usando busca semântica.",
    parameters={
        "user_input": (str, ...),
    }
)
async def simulate_arthur_response(ctx, user_input: str):
    """
    Simulates Arthur's response based on semantic similarity search over the unified dataset.
    Args:
        ctx: The skill execution context (provides logger, llm_call).
        user_input: The input string to simulate a response for.

    Returns:
        A dictionary containing either 'simulated_response' or 'error'.
    """
    logger = ctx.logger
    logger.info(f"Executing simulate_arthur_response (unified) for input: '{user_input[:100]}...'" )

    # 1. Load resources (uses cache)
    load_success, error_msg = load_resources(logger)

    # Handle critical load errors first
    if faiss is None:
         return {"error": "FAISS library is not installed."}
    if not load_success:
        logger.error(f"Aborting simulation due to resource load failure: {error_msg}")
        user_error = error_msg or "Failed to load necessary resources for simulation."
        return {"error": user_error}

    # 2. Check if simulation is possible (even if loaded, resources might be empty)
    if index_cache is None or index_cache.ntotal == 0 or not mapping_cache or not unified_dataset_cache:
        warning_msg = "Não há dados suficientes (índice, mapeamento ou dataset unificado) para simular uma resposta com base no histórico. A simulação pode não refletir Arthur com precisão. (Verifique se os scripts de coleta e indexação foram executados e geraram dados)."
        logger.warning(warning_msg)

    if model_cache is None:
         return {"error": "Embedding model is not loaded."}

    # 3. Generate embedding for user_input
    try:
        logger.debug(f"Generating embedding for input: '{user_input}'")
        input_embedding = model_cache.encode([user_input], convert_to_numpy=True).astype(np.float32)
        if input_embedding.ndim == 1:
             input_embedding = np.expand_dims(input_embedding, axis=0)
        logger.debug(f"Input embedding generated, shape: {input_embedding.shape}")
    except Exception as e:
        logger.error(f"Failed to generate embedding for input: {e}", exc_info=True)
        return {"error": "Failed to generate input embedding."}

    # 4. Search FAISS index (only if index exists and has vectors)
    retrieved_faiss_indices = []
    distances = []
    if index_cache is not None and index_cache.ntotal > 0:
        try:
            k = min(TOP_K, index_cache.ntotal)
            logger.debug(f"Searching FAISS index ({index_cache.ntotal} vectors) for {k} nearest neighbors...")
            distances, retrieved_faiss_indices = index_cache.search(input_embedding, k)
            retrieved_faiss_indices = retrieved_faiss_indices[0]
            distances = distances[0]
            valid_indices_mask = retrieved_faiss_indices != -1
            retrieved_faiss_indices = retrieved_faiss_indices[valid_indices_mask]
            distances = distances[valid_indices_mask]
            logger.debug(f"Retrieved {len(retrieved_faiss_indices)} valid indices: {retrieved_faiss_indices}")
            logger.debug(f"Distances: {distances}")
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}", exc_info=True)
            retrieved_faiss_indices = []
            distances = []
    else:
        logger.warning("FAISS index is not loaded or is empty. Cannot perform search.")

    # 5. Retrieve and Format Examples
    examples_for_prompt = []
    if mapping_cache and unified_dataset_cache and retrieved_faiss_indices is not None:
        try:
            for i, faiss_index in enumerate(retrieved_faiss_indices):
                faiss_index_str = str(faiss_index)
                metadata = mapping_cache.get(faiss_index_str)
                if metadata is None:
                    logger.warning(f"No mapping found for FAISS index {faiss_index} (String key: '{faiss_index_str}'). Skipping.")
                    continue
                record_index = int(faiss_index)
                if 0 <= record_index < len(unified_dataset_cache):
                    full_record = unified_dataset_cache[record_index]
                    record_text = prepare_record_text_for_prompt(full_record, logger)
                    if record_text:
                        examples_for_prompt.append(record_text)
                else:
                    logger.warning(f"Retrieved record index {record_index} ... is out of bounds... Skipping.")
        except Exception as e:
            logger.error(f"Error retrieving or formatting records using mapping: {e}", exc_info=True)
            examples_for_prompt = []

    # Prepare examples string
    if not examples_for_prompt:
        logger.info("No relevant examples found or formatted for the prompt after search.")
        examples_str = "Nenhum exemplo similar encontrado no histórico."
    else:
        examples_str = "\n\n---\n\n".join(examples_for_prompt)

    # 6. Construct Prompt and Call LLM
    prompt = f"""
Você é um simulador da forma de pensar e responder de Arthur. Com base nos exemplos do histórico de Arthur (se disponíveis) e na entrada do usuário fornecida, simule como Arthur responderia. Mantenha o tom e estilo de Arthur.

Entrada do Usuário:
{user_input}

Exemplos do Histórico de Arthur (Manifestos ou Conversas WhatsApp):
{examples_str}

Simulação da Resposta de Arthur:
"""

    try:
        logger.info("Calling LLM to simulate response (streaming)...")
        simulated_response_content = ""
        async for chunk in ctx.llm_call(prompt):
            simulated_response_content += chunk

        if not simulated_response_content or not simulated_response_content.strip():
             logger.warning("LLM returned an empty or whitespace-only response after streaming.")
             return {"simulated_response": "[Simulação falhou: O modelo não gerou uma resposta]"}

        if simulated_response_content.startswith("[LLM Call Error:"):
            logger.error(f"LLM call failed within the stream: {simulated_response_content}")
            return {"error": simulated_response_content}

        logger.info(f"LLM simulation stream finished. Total length: {len(simulated_response_content)}")
        logger.debug(f"Full Simulated response: {simulated_response_content[:100]}...")
        return {"simulated_response": simulated_response_content.strip()}

    except Exception as e:
        logger.error(f"LLM call failed during simulation stream processing: {e}", exc_info=True)
        return {"error": f"LLM call failed: {e}"} 