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
from a3x.core.skills import skill
import logging
from typing import Dict, Any, List, Optional
import random
import asyncio

# Core framework imports
from a3x.core.config import PROJECT_ROOT
from a3x.core.llm_interface import LLMInterface
from a3x.core.agent import _ToolExecutionContext
from a3x.core.context import Context
# Correct imports for memory functions:
from a3x.core.db_utils import retrieve_relevant_context, add_episodic_record
# from a3x.core.db_manager import VectorDBManager # Module does not exist yet
# from a3x.core.llm.prompts import build_arthur_simulation_prompt # Function does not exist

logger = logging.getLogger(__name__)

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

# Path to the dataset used for simulating Arthur's persona
# Assuming unified dataset for now
ARTHUR_DATASET_PATH = "data/arthur_unified_dataset.jsonl"
# Number of examples to retrieve from memory for context
NUM_MEMORY_EXAMPLES = 3
# Number of examples to retrieve from the dataset for few-shot prompting
NUM_DATASET_EXAMPLES = 5

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

# Helper function to load examples from the dataset (can be moved/improved)
def load_arthur_examples(dataset_path: str, num_examples: int) -> List[Dict[str, str]]:
    examples = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        # Simple random sampling
        if len(all_lines) <= num_examples:
            sampled_lines = all_lines
        else:
            sampled_lines = random.sample(all_lines, num_examples)

        for line in sampled_lines:
            try:
                record = json.loads(line.strip())
                # Expecting 'input' and 'arthur_response' keys based on process_whatsapp.py
                if "input" in record and "arthur_response" in record:
                    examples.append({"input": record["input"], "output": record["arthur_response"]})
                elif "text" in record: # Fallback for simpler formats
                     # Try splitting based on a common pattern if input/output not present
                     parts = record["text"].split(" -> Response: ")
                     if len(parts) == 2:
                         examples.append({"input": parts[0].replace("Input: ", ""), "output": parts[1]})
                     else: # If no clear split, use the whole text as input (less ideal)
                         examples.append({"input": record["text"], "output": ""})
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line in dataset: {line.strip()}")
    except FileNotFoundError:
        logger.error(f"Arthur dataset file not found: {dataset_path}")
    except Exception as e:
        logger.error(f"Error loading Arthur dataset examples: {e}")
    return examples

# --- Skill Definition ---

@skill(
    name="simulate_arthur_response",
    description="Simulates Arthur's response to a given user input based on his typical style and knowledge.",
    parameters={
        "context": {"type": Context, "description": "Execution context for LLM access and persona info."},
        "user_input": {"type": str, "description": "The user input to which Arthur should respond."},
        "mood": {"type": Optional[str], "default": "neutral", "description": "Simulated mood of Arthur ('neutral', 'focused', 'skeptical')."}
    }
)
async def simulate_arthur_response(
    context: Context,
    user_input: str,
    mood: Optional[str] = "neutral"
) -> Dict[str, Any]:
    """
    Simulates Arthur's response based on user input and relevant memory.
    Uses the LLMInterface from the execution context.
    """
    context.logger.info(f"Executing simulate_arthur_response for input: '{user_input[:100]}...'")
    
    # Get LLM interface from context
    llm_interface = context.llm_interface
    if not llm_interface:
        context.logger.error("LLMInterface not found in execution context.")
        return {"status": "error", "message": "Internal error: LLMInterface missing."}

    # 1. Load necessary resources (model, index, mapping, dataset)
    load_success, error_msg = load_resources(context.logger)
    if not load_success:
        return {"status": "error", "message": f"Failed to load resources: {error_msg}"}
    
    # Check for essential resources after load attempt
    if model_cache is None or index_cache is None or mapping_cache is None or unified_dataset_cache is None or faiss is None:
         return {"status": "error", "message": f"Essential resource missing after load attempt: {error_msg or 'Unknown reason'}"}

    # 2. Retrieve Relevant Context from Memory (using FAISS index)
    try:
        context.logger.debug("Encoding user input for similarity search...")
        input_embedding = model_cache.encode([user_input])
        
        context.logger.debug(f"Searching FAISS index (k={TOP_K})...")
        distances, indices = index_cache.search(input_embedding.astype(np.float32), TOP_K)
        
        memory_context_examples = []
        if indices.size > 0 and indices[0][0] != -1: # Check if any results found
            for idx in indices[0]: # Iterate through the retrieved indices
                if idx == -1: continue # Skip invalid index marker
                map_key = str(idx) # Mapping uses string keys
                if map_key in mapping_cache:
                    metadata = mapping_cache[map_key]
                    unified_dataset_index = metadata.get("unified_index")
                    if unified_dataset_index is not None and 0 <= unified_dataset_index < len(unified_dataset_cache):
                         record = unified_dataset_cache[unified_dataset_index]
                         formatted_text = prepare_record_text_for_prompt(record, context.logger)
                         if formatted_text:
                              memory_context_examples.append(formatted_text)
                         if len(memory_context_examples) >= NUM_MEMORY_EXAMPLES:
                              break # Stop once we have enough examples
                    else:
                         context.logger.warning(f"Mapping for index {idx} points to invalid unified_index: {unified_dataset_index}")
                else:
                     context.logger.warning(f"Index {idx} retrieved from FAISS not found in mapping file.")
        else:
            context.logger.warning("No relevant context found in memory (FAISS search returned no valid indices).")

    except Exception as e:
        context.logger.exception("Error retrieving context from memory:")
        return {"status": "error", "message": f"Failed to retrieve context: {e}"}

    # 3. Build the Prompt
    # Select a few random examples from the dataset for few-shot prompting
    # Ensure we don't pick more than available
    num_available_examples = len(unified_dataset_cache)
    num_few_shot = min(NUM_DATASET_EXAMPLES, num_available_examples)
    few_shot_examples = random.sample(unified_dataset_cache, num_few_shot) if num_available_examples > 0 else [] 

    few_shot_prompt_str = "\n\n---\n\n".join([prepare_record_text_for_prompt(ex, context.logger) for ex in few_shot_examples if prepare_record_text_for_prompt(ex, context.logger)])
    memory_context_str = "\n\n---\n\n".join(memory_context_examples) if memory_context_examples else "Nenhum contexto de memória relevante encontrado."

    # System Prompt + Context + Few-Shot + User Input
    system_prompt = (
        "Você é Arthur, uma IA com personalidade complexa documentada nos exemplos abaixo. "
        "Sua tarefa é responder à 'Entrada do Usuário' final como Arthur responderia, "
        "mantendo o tom, estilo e conteúdo consistentes com os exemplos e o contexto da memória fornecidos. "
        "Seja direto, evite introduções ou metacommentários. Apenas forneça a resposta simulada."
    )
    user_content = (
        f"CONTEXTO DA MEMÓRIA RELEVANTE:\n{memory_context_str}\n\n"
        f"EXEMPLOS DE COMPORTAMENTO PASSADO:\n{few_shot_prompt_str}\n\n"
        f"--- Tarefa Atual ---\n"
        f"Entrada do Usuário: {user_input}\n\n"
        f"Sua Resposta Simulada (como Arthur):"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    context.logger.debug(f"Prompt constructed. System length: {len(system_prompt)}, User content length: {len(user_content)}")

    # 4. Call LLM to Simulate Response
    try:
        context.logger.info("Calling LLM to simulate Arthur's response...")
        simulated_response = ""
        # Updated call site
        async for chunk in llm_interface.call_llm( # <-- USE INSTANCE METHOD
            messages=messages, 
            stream=True # Stream response
        ):
            simulated_response += chunk

        if not simulated_response or not simulated_response.strip():
            context.logger.warning("LLM simulation returned an empty response.")
            return {"status": "error", "message": "LLM simulation returned empty."}
        
        # Check for LLM error string
        if simulated_response.startswith("[LLM Error:"):
             context.logger.error(f"LLM call failed during simulation: {simulated_response}")
             return {"status": "error", "message": simulated_response}

        context.logger.info("Successfully simulated Arthur's response.")
        context.logger.debug(f"Simulated Response: {simulated_response}")

        return {"status": "success", "simulated_response": simulated_response.strip()}

    except Exception as e:
        context.logger.exception("Error during LLM call for simulation:")
        return {"status": "error", "message": f"LLM call failed: {e}"}

# === Example Usage (for testing) ===
# async def main():
#     logging.basicConfig(level=logging.INFO)
#     # Ensure DB is initialized if running standalone
#     # from a3x.core.db_utils import initialize_database
#     # initialize_database()
#     user_query = "Qual sua opinião sobre inteligência artificial geral?"
#     result = await simulate_arthur_response(user_query)
#     print("\n--- Simulation Result ---")
#     print(json.dumps(result, indent=2, ensure_ascii=False))
#     print("------------------------")

# if __name__ == "__main__":
#     asyncio.run(main()) 