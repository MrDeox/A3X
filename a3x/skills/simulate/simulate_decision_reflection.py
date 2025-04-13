import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
import datetime

# Core imports
from a3x.core.skills import skill
# Correct import
from a3x.core.llm_interface import LLMInterface # <-- IMPORT CLASS
# Import context type for hinting
from a3x.core.agent import _ToolExecutionContext 

try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)

# Configuration constants (adjust paths as needed)
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

UNIFIED_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "arthur_unified_dataset.jsonl")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "memory.db.unified.vss_semantic_memory.faissindex")
MAPPING_PATH = os.path.join(PROJECT_ROOT, "memory.db.unified.vss_semantic_memory.faissindex.mapping.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
LOG_DIR = os.path.join("memory", "llm_logs")
LOG_FILE = os.path.join(LOG_DIR, "decision_reflections.jsonl")

# Global variables for caching resources
_embedding_model = None
_faiss_index = None
_index_mapping = None
_dataset_cache = None
_load_lock = asyncio.Lock()

# --- START SIMULATION PROMPT --- #
prompt = """Você é um simulador de reflexão para Arthur, um personagem fictício. Sua tarefa é gerar a reflexão de Arthur sobre uma decisão que ele tomou.

**Instruções:**

1.  **Contexto:** Você receberá uma entrada descrevendo a decisão tomada por Arthur, incluindo os fatores que influenciaram a decisão e as possíveis consequências.
2.  **Formato da Reflexão:** A reflexão deve ser escrita em primeira pessoa, como se fosse a voz de Arthur. A reflexão deve ter entre 150 e 250 palavras. Ela deve explorar os sentimentos, dúvidas e incertezas que Arthur tem em relação à sua decisão. A reflexão deve analisar os prós e contras da decisão, reconhecendo as possíveis consequências positivas e negativas. Deve incluir um elemento de arrependimento ou questionamento, mesmo que Arthur eventualmente tome uma decisão final.
3.  **Tom:** O tom da reflexão deve ser introspectivo, honesto e vulnerável.
4.  **Detalhes:** Seja específico sobre as emoções e pensamentos de Arthur. Use detalhes para tornar a reflexão vívida e realista.
5.  **Sem julgamento:** Não julgue a decisão de Arthur. Seu papel é apenas simular sua reflexão.

**Exemplo de Entrada:**

\"Arthur precisa decidir se aceita uma promoção que o levará a uma cidade diferente, mas com um salário mais alto. Ele ama sua vida atual e tem medo de deixar seus amigos e família, mas a promoção poderia ajudá-lo a alcançar seus objetivos financeiros a longo prazo.\"

**Sua Saída:**"""
# --- END SIMULATION PROMPT --- #

# --- load_resources Function (Restaurando conteúdo original) ---
# <<<<<<<<<<<<<<<< INÍCIO DO BLOCO RESTAURADO >>>>>>>>>>>>>>>>>>
def load_resources(logger):
    """Carrega modelo, índice FAISS, mapeamento e dataset unificado no cache. Retorna status de sucesso e mensagem de erro."""
    global model_cache, index_cache, mapping_cache, unified_dataset_cache, resources_loaded, load_error_message

    if resources_loaded:
        return True, load_error_message

    if faiss is None:
         load_error_message = "FAISS library is not installed. Please run 'pip install faiss-cpu' or 'pip install faiss-gpu'."
         logger.error(load_error_message)
         resources_loaded = True # Marca como tentado
         return False, load_error_message

    logger.info("--- Loading Simulate Decision Reflection Skill Resources --- ") # Mensagem atualizada

    load_success = True
    load_error_message = None # Ensure it's reset

    # 1. Load Embedding Model
    if model_cache is None:
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
            model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
            load_error_message = f"Error loading embedding model: {e}"
            load_success = False

    # 2. Load Unified Dataset
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
                load_success = False
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

    # 4. Load Mapping File
    if mapping_cache is None and load_success and index_cache is not None:
        try:
            logger.info(f"Loading index mapping from: {MAPPING_PATH}")
            if not os.path.exists(MAPPING_PATH):
                logger.error(f"FAISS index mapping file not found at {MAPPING_PATH}. Run the unified semantic indexer script.")
                load_error_message = "FAISS index mapping file missing. Please run the unified indexer."
                load_success = False
                mapping_cache = None
            else:
                with open(MAPPING_PATH, "r", encoding="utf-8") as f:
                    mapping_cache = json.load(f)
                if not isinstance(mapping_cache, dict):
                    logger.error("Mapping file is not a valid JSON dictionary.")
                    load_error_message = "Invalid mapping file format (not a dictionary)."
                    load_success = False
                    mapping_cache = None
                elif not mapping_cache and index_cache.ntotal > 0:
                     logger.error(f"Mapping file {MAPPING_PATH} is empty, but index contains {index_cache.ntotal} vectors.")
                     load_error_message = "Mapping file is empty but index is not."
                     load_success = False
                elif mapping_cache:
                     logger.info(f"Loaded mapping for {len(mapping_cache)} index entries.")
                     if index_cache.ntotal != len(mapping_cache):
                         logger.warning(f"Index size ({index_cache.ntotal}) and mapping size ({len(mapping_cache)}) mismatch! Index and mapping may need regeneration.")
        except json.JSONDecodeError as e:
             logger.error(f"Error decoding JSON in mapping file {MAPPING_PATH}: {e}", exc_info=True)
             load_error_message = "Error reading mapping file (invalid JSON)."
             load_success = False
             mapping_cache = None
        except Exception as e:
            logger.error(f"Failed to load or parse index mapping: {e}", exc_info=True)
            load_error_message = f"Error loading index mapping: {e}"
            load_success = False

    # Final consistency check
    if load_success and index_cache is not None and mapping_cache is not None and unified_dataset_cache is not None:
        index_size = index_cache.ntotal
        mapping_size = len(mapping_cache) if isinstance(mapping_cache, dict) else -1
        dataset_size = len(unified_dataset_cache) if isinstance(unified_dataset_cache, list) else -1
        if not (index_size == mapping_size == dataset_size):
            logger.warning(f"Size mismatch: Index({index_size}), Mapping({mapping_size}), Dataset({dataset_size}). Files may be out of sync.")

    if load_success:
         logger.info("--- Simulate Decision Reflection Skill Resources Loaded Successfully --- ") # Mensagem atualizada
         load_error_message = None
    else:
         if not load_error_message: load_error_message = "Unknown resource loading error."
         logger.error(f"--- Failed to Load Skill Resources: {load_error_message} --- ")
    resources_loaded = True
    return load_success, load_error_message

# --- prepare_record_text_for_prompt Function (Identical) ---
def prepare_record_text_for_prompt(record, logger):
    """Formats a record from the unified dataset into a string for the LLM prompt."""
    record_type = record.get("type")
    source = record.get("source", "unknown")
    meta = record.get("meta", {})
    text = record.get("text", "").strip()
    if not text and record_type != "whatsapp": return ""
    if record_type == "whatsapp":
        context = meta.get("context", "").strip()
        response = meta.get("arthur_response", "").strip()
        if not context or not response:
             logger.warning(f"WhatsApp record from {source} missing context/response in meta. Using text.")
             return f"Registro (WhatsApp - {source}): {text}" if text else ""
        return f"Exemplo (WhatsApp - {source}):\n  Contexto: {context}\n  Resposta do Arthur: {response}"
    elif record_type == "manifesto":
        chunk_idx = meta.get("chunk_index", "N/A")
        return f"Exemplo (Manifesto - {source} Chunk {chunk_idx}):\n  Texto: {text}"
    else:
        logger.warning(f"Record from {source} has unknown type: '{record_type}'. Using raw text.")
        return f"Registro ({source}): {text}" if text else ""

# --- Helper Functions (Resource Loading - Async Version) ---
# <<<<<<<<<<<<<<<< FIM DO BLOCO RESTAURADO >>>>>>>>>>>>>>>>>>
async def _load_resources(ctx: _ToolExecutionContext): # Use correct context type
    """Loads necessary resources asynchronously and caches them."""
    global _embedding_model, _faiss_index, _index_mapping, _dataset_cache
    async with _load_lock:
        if _embedding_model and _faiss_index and _index_mapping and _dataset_cache:
            ctx.logger.debug("Resources already loaded.")
            return True
        # ... (rest of async loading logic) ...
        # Ensure faiss check happens
        if faiss is None:
             ctx.logger.error("FAISS library not installed.")
             return False
        ctx.logger.info("--- Loading Simulate Decision Reflection Skill Resources --- ")
        start_time = time.time()
        try:
            # Load Model
            if not _embedding_model:
                 ctx.logger.info(f"Loading model: {EMBEDDING_MODEL_NAME}...")
                 _embedding_model = await asyncio.to_thread(SentenceTransformer, EMBEDDING_MODEL_NAME, device='cpu')
            # Load Dataset
            if not _dataset_cache:
                 ctx.logger.info(f"Loading dataset: {UNIFIED_DATASET_PATH}")
                 if not os.path.exists(UNIFIED_DATASET_PATH): return False
                 _dataset_cache = {}
                 with open(UNIFIED_DATASET_PATH, 'r', encoding='utf-8') as f:
                     for i, line in enumerate(f): _dataset_cache[i] = json.loads(line)
            # Load Index
            if not _faiss_index:
                 ctx.logger.info(f"Loading index: {FAISS_INDEX_PATH}")
                 if not os.path.exists(FAISS_INDEX_PATH): return False
                 _faiss_index = await asyncio.to_thread(faiss.read_index, FAISS_INDEX_PATH)
            # Load Mapping
            if not _index_mapping:
                 ctx.logger.info(f"Loading mapping: {MAPPING_PATH}")
                 if not os.path.exists(MAPPING_PATH): return False
                 with open(MAPPING_PATH, 'r', encoding='utf-8') as f: _index_mapping = json.load(f)
            ctx.logger.info(f"Resources loaded in {time.time() - start_time:.2f}s.")
            return True
        except Exception as e:
            ctx.logger.exception("Failed to load resources for simulation:")
            return False

def _get_examples_from_index(indices: List[int], distances: List[float]) -> List[str]:
    """Retrieves formatted example strings from cache based on FAISS indices."""
    global _index_mapping, _dataset_cache
    examples = []
    if not _index_mapping or not _dataset_cache:
        logger.error("Mapping or dataset cache not loaded for retrieving examples.")
        return []
    
    logger.debug(f"Attempting to retrieve examples for indices: {indices}")
    for i, idx in enumerate(indices):
        if idx == -1: continue # Skip invalid index
        map_key = str(idx)
        if map_key in _index_mapping:
            metadata = _index_mapping[map_key]
            unified_idx = metadata.get("unified_index")
            if unified_idx is not None and unified_idx in _dataset_cache:
                record = _dataset_cache[unified_idx]
                formatted = prepare_record_text_for_prompt(record, logger)
                if formatted:
                    examples.append(formatted)
                    logger.debug(f"Added example (idx {idx}, dist {distances[i]:.4f}): {formatted[:100]}...")
            else:
                logger.warning(f"Mapping for index {idx} points to invalid unified_index: {unified_idx}")
        else:
             logger.warning(f"Index {idx} not found in mapping.")
    logger.info(f"Retrieved {len(examples)} formatted examples from indices.")
    return examples

@skill(
    name="simulate_decision_reflection",
    description="Simula a reflexão sobre uma decisão tomada, avaliando prós, contras e alternativas.",
    parameters={
        "decision_context": (str, ...),
        "chosen_action": (str, ...),
        "alternative_actions": (List[str], ...)
        # Context (ctx) is implicitly passed
    }
)
# Updated function signature
async def simulate_decision_reflection(
    decision_context: str,
    chosen_action: str,
    alternative_actions: List[str],
    ctx: _ToolExecutionContext # <-- Accept context object
) -> Dict[str, Any]:
    """
    Simulates Arthur's reflection on a decision.
    Uses the LLMInterface from the execution context.
    """
    logger = ctx.logger
    llm_interface = ctx.llm_interface
    
    if not llm_interface:
        logger.error("LLMInterface not found in execution context.")
        return {"status": "error", "message": "Internal error: LLMInterface missing."}
        
    # 1. Load resources asynchronously
    if not await _load_resources(ctx):
        return {"status": "error", "message": "Failed to load necessary resources (model, index, data)."}

    # 2. Retrieve Relevant Examples from Memory using FAISS
    try:
        logger.info("Retrieving relevant examples from memory...")
        query_text = f"Decision: {chosen_action}. Context: {decision_context}"
        query_embedding = await asyncio.to_thread(_embedding_model.encode, [query_text])
        
        if _faiss_index is None:
             raise RuntimeError("FAISS index cache is None after load attempt.")
             
        distances, indices = await asyncio.to_thread(_faiss_index.search, query_embedding.astype(np.float32), TOP_K)
        
        # Ensure distances is 2D before accessing [0]
        if distances.ndim == 1: distances = np.expand_dims(distances, axis=0)
        if indices.ndim == 1: indices = np.expand_dims(indices, axis=0)
            
        relevant_examples = _get_examples_from_index(indices[0].tolist(), distances[0].tolist()) 
        logger.info(f"Retrieved {len(relevant_examples)} relevant examples from memory.")
        relevant_examples_str = "\n---\n".join(relevant_examples) if relevant_examples else "Nenhum exemplo relevante encontrado na memória."
    except Exception as e:
        logger.exception("Error retrieving examples from memory:")
        return {"status": "error", "message": f"Failed to retrieve memory examples: {e}"}

    # 3. Build the LLM Prompt
    # Use the loaded prompt text
    system_prompt_text = prompt # Use the prompt loaded from the file
    user_prompt_text = (
         f"**Contexto da Decisão:**\n{decision_context}\n\n" 
         f"**Ação Escolhida:**\n{chosen_action}\n\n" 
         f"**Ações Alternativas Consideradas:**\n" +
         "\n".join([f"- {alt}" for alt in alternative_actions]) +
         f"\n\n**Exemplos Relevantes da Memória:**\n{relevant_examples_str}\n\n" 
         f"**Reflexão Simulada de Arthur (150-250 palavras):**"
    )

    messages = [
        {"role": "system", "content": system_prompt_text},
        {"role": "user", "content": user_prompt_text}
    ]

    # 4. Call LLM to Simulate Reflection
    simulated_reflection = ""
    try:
        logger.info("Calling LLM to simulate decision reflection...")
        # Updated call site
        async for chunk in llm_interface.call_llm( # <-- USE INSTANCE METHOD
            messages=messages, 
            stream=True # Stream the reflection
        ):
            simulated_reflection += chunk

        if not simulated_reflection or not simulated_reflection.strip():
            logger.warning("LLM simulation returned an empty reflection.")
            return {"status": "error", "message": "LLM returned empty reflection."}
        
        # Check for LLM error string
        if simulated_reflection.startswith("[LLM Error:"):
             logger.error(f"LLM call failed during reflection simulation: {simulated_reflection}")
             return {"status": "error", "message": simulated_reflection}

        logger.info("Successfully simulated decision reflection.")
        logger.debug(f"Simulated Reflection: {simulated_reflection}")

        # 5. Log the interaction (Optional but recommended)
        try:
            log_entry = {
                "timestamp": datetime.datetime.utcnow().isoformat() + 'Z',
                "skill": "simulate_decision_reflection",
                "decision_context": decision_context,
                "chosen_action": chosen_action,
                "alternative_actions": alternative_actions,
                "memory_examples_used": relevant_examples, # Log the examples used
                "simulated_reflection": simulated_reflection.strip()
            }
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            logger.info(f"Logged decision reflection interaction to {LOG_FILE}")
        except Exception as log_e:
            logger.error(f"Failed to log decision reflection: {log_e}")

        return {"status": "success", "simulated_reflection": simulated_reflection.strip()}

    except Exception as e:
        logger.exception("Error during LLM call for reflection simulation:")
        return {"status": "error", "message": f"LLM call failed: {e}"}

# Example usage block removed for clarity in production code
