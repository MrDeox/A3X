import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
import datetime

try:
    import faiss
except ImportError:
    faiss = None

from a3x.core.skills import skill
from a3x.core.config import LLAMA_MODEL_PATH
from a3x.core.llm_interface import call_llm

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

# --- START SIMULATION PROMPT (Refined by LLM Analysis) --- #
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
async def _load_resources(ctx):
    """Loads necessary resources (model, index, data) asynchronously and caches them."""
    global _embedding_model, _faiss_index, _index_mapping, _dataset_cache
    async with _load_lock:
        if _embedding_model and _faiss_index and _index_mapping and _dataset_cache:
            ctx.logger.debug("Resources already loaded.")
            return True

        ctx.logger.info("--- Loading Simulate Decision Reflection Skill Resources --- ")
        start_time = time.time()

        try:
            # 1. Load Embedding Model
            if not _embedding_model:
                ctx.logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
                model_load_start = time.time()
                # Run synchronous model loading in a separate thread
                _embedding_model = await asyncio.to_thread(SentenceTransformer, EMBEDDING_MODEL_NAME, device='cpu')
                ctx.logger.info(f"Embedding model loaded in {time.time() - model_load_start:.2f}s.")

            # 2. Load Dataset Cache (if needed for mapping)
            if not _dataset_cache:
                ctx.logger.info(f"Loading unified dataset from: {UNIFIED_DATASET_PATH}")
                if not os.path.exists(UNIFIED_DATASET_PATH):
                    ctx.logger.error(f"Unified dataset file not found: {UNIFIED_DATASET_PATH}")
                    return False
                _dataset_cache = {}
                with open(UNIFIED_DATASET_PATH, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        try:
                            record = json.loads(line)
                            _dataset_cache[i] = record # Store record by line index
                        except json.JSONDecodeError:
                            ctx.logger.warning(f"Skipping invalid JSON line {i+1} in {UNIFIED_DATASET_PATH}")
                ctx.logger.info(f"Loaded {len(_dataset_cache)} records from unified dataset.")

            # 3. Load FAISS Index
            if not _faiss_index:
                ctx.logger.info(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
                if not os.path.exists(FAISS_INDEX_PATH):
                    ctx.logger.error(f"FAISS index file not found: {FAISS_INDEX_PATH}")
                    return False
                _faiss_index = await asyncio.to_thread(faiss.read_index, FAISS_INDEX_PATH)
                ctx.logger.info(f"FAISS index loaded. Contains {_faiss_index.ntotal} vectors.")
                if _dataset_cache and _faiss_index.ntotal != len(_dataset_cache):
                     ctx.logger.warning(f"Mismatch: Index has {_faiss_index.ntotal} vectors, dataset cache has {len(_dataset_cache)} records.")

            # 4. Load Index Mapping
            if not _index_mapping:
                ctx.logger.info(f"Loading index mapping from: {MAPPING_PATH}")
                if not os.path.exists(MAPPING_PATH):
                    ctx.logger.error(f"Index mapping file not found: {MAPPING_PATH}")
                    # Allow proceeding without mapping, but log error
                    _index_mapping = {}
                else:
                    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
                        _index_mapping = json.load(f)
                    ctx.logger.info(f"Loaded mapping for {len(_index_mapping)} index entries.")
                    # Convert string keys back to int if necessary (JSON saves keys as strings)
                    _index_mapping = {int(k): v for k, v in _index_mapping.items()}

            ctx.logger.info(f"--- Simulate Decision Reflection Skill Resources Loaded Successfully (Total: {time.time() - start_time:.2f}s) --- ")
            return True

        except Exception as e:
            ctx.logger.exception("Error loading resources for simulate_decision_reflection:")
            # Reset potentially partially loaded resources on error
            _embedding_model = _faiss_index = _index_mapping = _dataset_cache = None
            return False

def _get_examples_from_index(indices: List[int], distances: List[float]) -> List[str]:
    """Retrieves and formats examples from the dataset cache based on FAISS indices."""
    examples = []
    if not _dataset_cache or not _index_mapping:
        logger.error("Dataset cache or index mapping not loaded. Cannot retrieve examples.")
        return examples

    for i, idx in enumerate(indices):
        if idx == -1: continue # Skip invalid index from FAISS search

        # <<< MODIFICAÇÃO: Assumir que as chaves no mapping SÃO os índices FAISS (inteiros) >>>
        # mapped_info = _index_mapping.get(str(idx)) # Original (se chaves fossem strings)
        mapped_info = _index_mapping.get(idx) # Correto se chaves são inteiros

        if not mapped_info:
            logger.warning(f"No mapping found for index {idx}. Skipping example.")
            continue

        # Use o índice FAISS diretamente como chave para o cache do dataset (se for 0-based)
        cache_key = idx
        record = _dataset_cache.get(cache_key)

        if record:
            source = mapped_info.get("source", "unknown_source")
            record_type = record.get("type", "unknown_type") # <<< Adicionado para fallback mais genérico >>>

            # Format based on type (whatsapp or manifesto)
            if record_type == "whatsapp" and record.get("meta", {}).get("context") and record.get("meta", {}).get("arthur_response"):
                inp = record["meta"]["context"]
                resp = record["meta"]["arthur_response"]
                examples.append(f"Exemplo de Decisão/Resposta (WhatsApp: {source}):\nINPUT:\n\"\"\"\n{inp}\n\"\"\"\nRESPOSTA_ARTHUR:\n\"\"\"\n{resp}\n\"\"\"")
            elif record_type == "manifesto":
                 # Simple text chunk from manifesto
                 text = record.get("text", "")
                 chunk_idx = record.get("meta", {}).get("chunk_index", "N/A") # Get chunk index safely
                 examples.append(f"Exemplo de Manifesto/Nota ({source} - Chunk {chunk_idx}):\n\"\"\"\n{text}\n\"\"\"")
            else:
                 # Fallback for other types or incomplete records
                 text = record.get("text", record.get("input", "Conteúdo indisponível"))
                 examples.append(f"Exemplo ({record_type} - {source}):\n\"\"\"\n{text}\n\"\"\"")
        else:
            logger.warning(f"Record not found in dataset cache for index {idx} (mapped key {cache_key}). Skipping example.")

    return examples

# --- Skill Definition --- #

@skill(
    name="simulate_decision_reflection",
    description="Simula como Arthur refletiria sobre um problema ou questão e tomaria uma decisão, usando Chain-of-Thought e buscando exemplos no histórico unificado.",
    parameters={
        "user_input": (str, ...), # Input é a questão/problema
        "ctx": (Context, None)
    }
)
async def simulate_decision_reflection(ctx, user_input: str):
    """
    Simulates Arthur's decision-making reflection process based on semantic search.
    Args:
        ctx: The skill execution context (provides logger, llm_call).
        user_input: The question or problem to reflect upon.

    Returns:
        A dictionary containing either 'simulated_reflection' or 'error'.
    """
    logger = ctx.logger
    logger.info(f"Executing simulate_decision_reflection for input: '{user_input[:100]}...'")

    # 1. Load resources (uses cache)
    # <<< MODIFICADO: Chamar a versão async >>>
    if not await _load_resources(ctx):
        logger.error("Aborting reflection simulation due to resource load failure.")
        return {"error": "Failed to load necessary resources for simulation."}

    # Check essential resources after attempting load
    if _embedding_model is None: return {"error": "Embedding model not loaded."}
    if _faiss_index is None or not _index_mapping or not _dataset_cache:
        warning_msg = "(Não foi possível simular a reflexão: dados históricos insuficientes ou não carregados. Verifique os arquivos de índice, mapeamento e dataset unificado.)"
        logger.warning(f"Data for simulation missing or incomplete: {warning_msg}")
        return {"simulated_reflection": warning_msg}

    # 2. Generate embedding for user_input
    try:
        logger.debug(f"Generating embedding for: '{user_input}'")
        # <<< MODIFICADO: Usar a versão async do encode >>>
        input_embedding = await asyncio.to_thread(
            _embedding_model.encode, [user_input], convert_to_numpy=True
        )
        input_embedding = input_embedding.astype(np.float32)
        if input_embedding.ndim == 1: input_embedding = np.expand_dims(input_embedding, axis=0)
    except Exception as e:
        logger.error(f"Failed to generate input embedding: {e}", exc_info=True)
        return {"error": "Failed to generate input embedding."}

    # 3. Search FAISS index
    retrieved_faiss_indices = []
    distances = [] # Initialize distances
    if _faiss_index.ntotal > 0:
        try:
            k = min(TOP_K, _faiss_index.ntotal)
            logger.debug(f"Searching FAISS index ({_faiss_index.ntotal} vectors) for {k} nearest neighbors...")
            # <<< MODIFICADO: Usar a versão async da busca >>>
            distances, retrieved_faiss_indices = await asyncio.to_thread(
                _faiss_index.search, input_embedding, k
            )
            retrieved_faiss_indices = retrieved_faiss_indices[0]
            distances = distances[0] # Store distances
            valid_indices_mask = retrieved_faiss_indices != -1
            retrieved_faiss_indices = retrieved_faiss_indices[valid_indices_mask]
            distances = distances[valid_indices_mask]
            logger.debug(f"Retrieved {len(retrieved_faiss_indices)} valid indices: {retrieved_faiss_indices}")
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}", exc_info=True)
            retrieved_faiss_indices = []
            distances = []
    else:
        logger.warning("FAISS index is empty. Cannot search.")

    # 4. Retrieve and Format Examples
    examples_for_prompt = []
    if retrieved_faiss_indices is not None and len(retrieved_faiss_indices) > 0:
        logger.info(f"Retrieving and formatting {len(retrieved_faiss_indices)} examples...")
        try:
            # <<< MODIFICADO: Passar distances para a função auxiliar >>>
            examples_for_prompt = _get_examples_from_index(retrieved_faiss_indices.tolist(), distances.tolist())
        except Exception as e:
            logger.error(f"Error retrieving/formatting examples: {e}", exc_info=True)
            examples_for_prompt = []

    # Prepare examples string
    if not examples_for_prompt:
        logger.info("No relevant examples found for the prompt.")
        examples_str = "Nenhum exemplo similar encontrado no histórico."
    else:
        examples_str = "\n\n---\n\n".join(examples_for_prompt)

    # 5. Construct the prompt using the f-string defined at the top
    final_prompt = prompt.format(user_input=user_input, examples_str=examples_str) # Format the prompt string here

    ctx.logger.debug("Prompt construído. Chamando LLM para simulação...")
    # ctx.logger.debug(f"Prompt para LLM (simulação):\\n{final_prompt[:500]}...") # Log inicial do prompt

    # 6. Call LLM and Accumulate Response
    try:
        logger.info("Calling LLM to simulate decision reflection (streaming)...")
        simulated_reflection_content = ""
        llm_start_time = time.time()
        # Removed debug prints
        try:
            # <<< Use call_llm directly >>>
            # We assume the LLM will follow the prompt and provide the full reflection.
            # Stream=True is used, but we accumulate the result.
            async for chunk in call_llm(messages=[{"role": "user", "content": final_prompt}], stream=True, timeout=180): # Increased timeout
                simulated_reflection_content += chunk

            # Check if the response is empty after streaming
            if not simulated_reflection_content or not simulated_reflection_content.strip():
                ctx.logger.error("LLM returned an empty simulation response.")
                raise ValueError("LLM returned empty response")
            # Check for explicit error markers that might come from call_llm or its handling
            if simulated_reflection_content.startswith("[LLM Call Error:"):
                ctx.logger.error(f"LLM call failed: {simulated_reflection_content}")
                raise ValueError(simulated_reflection_content) # Propagate error

            llm_duration = time.time() - llm_start_time
            logger.info(f"LLM simulation finished. Total length: {len(simulated_reflection_content)}. Duration: {llm_duration:.3f}s")
        except Exception as e:
            logger.exception("Error during LLM call or processing:")
            return {"error": f"Erro ao chamar LLM ou processar resposta: {e}"}

        logger.info(f"LLM simulation finished. Total length: {len(simulated_reflection_content)}")

        # Log the result
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "input": user_input,
            "simulation": simulated_reflection_content.strip(),
            "model": LLAMA_MODEL_PATH, # Log which model was used
            "retrieved_indices": retrieved_faiss_indices.tolist() if retrieved_faiss_indices is not None else [],
            "examples_used_count": len(examples_for_prompt)
        }
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            logger.debug(f"Decision reflection log saved to {LOG_FILE}")
            logger.info(f"[DIAGNÓSTICO] Tentativa de escrita do log concluída para: {LOG_FILE}")
        except Exception as log_e:
            logger.error(f"Failed to write decision reflection log to {LOG_FILE}: {log_e}", exc_info=True)

        return {"simulated_reflection": simulated_reflection_content.strip()}

    except Exception as e: # Catch potential errors during the overall process
        logger.error(f"Unexpected error during simulate_decision_reflection: {e}", exc_info=True)
        return {"error": f"Unexpected error during simulation: {e}"}

# Example usage block removed for clarity in production code
