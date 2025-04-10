"""
Skill para simular o processo de decisão e reflexão de Arthur sobre um tópico.
Utiliza embeddings e busca por similaridade com FAISS no dataset unificado.
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio # Necessário para async def
import logging # Necessário para logger
try:
    import faiss
except ImportError:
    # Permite a importação do módulo, mas a skill falhará se FAISS for necessário.
    faiss = None
from a3x.core.tools import skill

# --- Configuration (Reused from simulate_arthur_response) ---
try:
    # Assume que o arquivo da skill está em a3x/skills/simulate/
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
except NameError:
    # Fallback se __file__ não estiver definido (ex: execução interativa)
    PROJECT_ROOT = os.path.abspath('.')

# Caminhos relativos à raiz do projeto - USANDO RECURSOS UNIFICADOS
UNIFIED_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "arthur_unified_dataset.jsonl")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "memory.db.unified.vss_semantic_memory.faissindex")
MAPPING_PATH = os.path.join(PROJECT_ROOT, "memory.db.unified.vss_semantic_memory.faissindex.mapping.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5 # Número de exemplos similares a recuperar

# --- Resource Caching (Reused) ---
# Caches para evitar recarregar recursos em cada chamada da skill na mesma execução do agente
model_cache = None
index_cache = None
mapping_cache = None
unified_dataset_cache = None
resources_loaded = False
load_error_message = None

# --- load_resources Function (Adaptado de simulate_arthur_response) ---
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

# --- Skill Definition --- #

@skill(
    name="simulate_decision_reflection",
    description="Simula como Arthur refletiria sobre um problema ou questão e tomaria uma decisão, usando Chain-of-Thought e buscando exemplos no histórico unificado.",
    parameters={
        "user_input": (str, ...), # Input é a questão/problema
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
    load_success, error_msg = load_resources(logger)
    if faiss is None: return {"error": "FAISS library not installed."}
    if not load_success:
        # Use the specific error message from loading if available
        final_error_msg = error_msg or "Failed to load necessary resources for simulation."
        logger.error(f"Aborting reflection simulation due to resource load failure: {final_error_msg}")
        return {"error": final_error_msg}

    # Check essential resources after attempting load
    if model_cache is None: return {"error": "Embedding model not loaded."}
    if index_cache is None or not mapping_cache or not unified_dataset_cache:
        # Handle case where resources loaded but might be empty/missing critical parts
        warning_msg = "(Não foi possível simular a reflexão: dados históricos insuficientes ou não carregados. Verifique os arquivos de índice, mapeamento e dataset unificado.)"
        logger.warning(f"Data for simulation missing or incomplete: {warning_msg}")
        return {"simulated_reflection": warning_msg}

    # 2. Generate embedding for user_input
    try:
        logger.debug(f"Generating embedding for: '{user_input}'")
        input_embedding = model_cache.encode([user_input], convert_to_numpy=True).astype(np.float32)
        if input_embedding.ndim == 1: input_embedding = np.expand_dims(input_embedding, axis=0)
    except Exception as e:
        logger.error(f"Failed to generate input embedding: {e}", exc_info=True)
        return {"error": "Failed to generate input embedding."}

    # 3. Search FAISS index
    retrieved_faiss_indices = []
    if index_cache.ntotal > 0:
        try:
            k = min(TOP_K, index_cache.ntotal)
            logger.debug(f"Searching FAISS index ({index_cache.ntotal} vectors) for {k} nearest neighbors...")
            distances, retrieved_faiss_indices = index_cache.search(input_embedding, k)
            retrieved_faiss_indices = retrieved_faiss_indices[0]
            # distances = distances[0] # Distances not used currently, commented out
            valid_indices_mask = retrieved_faiss_indices != -1
            retrieved_faiss_indices = retrieved_faiss_indices[valid_indices_mask]
            logger.debug(f"Retrieved {len(retrieved_faiss_indices)} valid indices: {retrieved_faiss_indices}")
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}", exc_info=True); retrieved_faiss_indices = []
    else: logger.warning("FAISS index is empty. Cannot search.")

    # 4. Retrieve and Format Examples
    examples_for_prompt = []
    if retrieved_faiss_indices is not None and len(retrieved_faiss_indices) > 0:
        logger.info(f"Retrieving and formatting {len(retrieved_faiss_indices)} examples...")
        try:
            for i, faiss_index in enumerate(retrieved_faiss_indices):
                faiss_index_str = str(faiss_index)
                metadata = mapping_cache.get(faiss_index_str)
                if metadata is None: continue # Skip if no mapping found
                record_index = int(faiss_index)
                if 0 <= record_index < len(unified_dataset_cache):
                    full_record = unified_dataset_cache[record_index]
                    record_text = prepare_record_text_for_prompt(full_record, logger)
                    if record_text: examples_for_prompt.append(record_text)
                else: logger.warning(f"Retrieved index {record_index} out of bounds for dataset size {len(unified_dataset_cache)}.")
        except Exception as e:
            logger.error(f"Error retrieving/formatting examples: {e}", exc_info=True); examples_for_prompt = []

    # Prepare examples string
    if not examples_for_prompt:
        logger.info("No relevant examples found for the prompt.")
        examples_str = "Nenhum exemplo similar encontrado no histórico."
    else:
        examples_str = "\n\n---\n\n".join(examples_for_prompt)

    # 5. Construct Chain-of-Thought Prompt
    prompt = f"""
Você é uma instância simulada da mente de Arthur. Seu objetivo é refletir sobre a questão abaixo como Arthur faria, usando um processo de raciocínio encadeado. Considere os exemplos do histórico e siga os passos:

Questão:
{user_input}

Exemplos do histórico (manifestos e conversas):
{examples_str}

Agora reflita passo a passo como Arthur faria:

## Análise do Contexto
(Identifique os aspectos importantes da questão baseando-se na entrada e nos exemplos, se houver)

## Raciocínio Interno
(Explique como Arthur tende a pensar sobre esse tipo de problema ou conceito, considerando sua filosofia de adaptabilidade, resiliência, modularidade, antifragilidade, etc., como visto nos exemplos)

## Critérios de Decisão
(Defina quais princípios ou fatores Arthur consideraria mais importantes para tomar uma decisão ou formar uma opinião sobre isso. Ex: Eficiência, Autonomia, Aprendizado, Impacto, Ética, Simplicidade, Ação)

## Escolha e Decisão
(Descreva qual seria a decisão, ação ou opinião de Arthur sobre a questão, sendo o mais direto possível)

## Justificativa Final
(Resuma por que essa seria a melhor decisão ou perspectiva segundo a lógica de Arthur, conectando com os critérios e o raciocínio)
"""

    # 6. Call LLM and Accumulate Response
    try:
        logger.info("Calling LLM to simulate decision reflection (streaming)...")
        simulated_reflection_content = ""
        # Assuming ctx.llm_call is an async generator that yields text chunks
        async for chunk in ctx.llm_call(prompt):
            simulated_reflection_content += chunk

        # Process accumulated response
        if not simulated_reflection_content or not simulated_reflection_content.strip():
             logger.warning("LLM returned empty response after streaming.")
             return {"simulated_reflection": "[Simulação falhou: O modelo não gerou reflexão]"}

        # Check if the response contains the error marker potentially yielded by the llm_call wrapper
        if simulated_reflection_content.startswith("[LLM Call Error:"):
            logger.error(f"LLM call failed within stream: {simulated_reflection_content}")
            return {"error": simulated_reflection_content}

        logger.info(f"LLM simulation finished. Total length: {len(simulated_reflection_content)}")

        # Placeholder for future: Save call details to agent memory/experience buffer
        # logger.debug("TODO: Save decision simulation to memory.")

        return {"simulated_reflection": simulated_reflection_content.strip()}

    except Exception as e:
        logger.error(f"LLM call failed during simulation stream: {e}", exc_info=True)
        return {"error": f"LLM call failed: {e}"} 