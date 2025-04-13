# Start of new file 
import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
import datetime
import itertools

# Ensure skill decorator is imported correctly based on current structure
try:
    from a3x.core.skills import skill
    from a3x.core.context import Context
    from a3x.core.llm_interface import call_llm
except ImportError:
    # Fallback if structure is different or running standalone
    def skill(**kwargs):
        def decorator(func):
            return func
        return decorator
    async def call_llm(*args, **kwargs):
        # Dummy LLM call for standalone testing
        yield "Regra geral de teste (sem LLM real)\n"
    Context = Any

logger = logging.getLogger(__name__)

# Constants - Using the unified log for reading, new file for writing generalized rules
LEARNING_LOG_DIR = "memory/learning_logs"
CONSOLIDATED_HEURISTIC_LOG_FILE = os.path.join(LEARNING_LOG_DIR, "learned_heuristics_consolidated.jsonl")
GENERALIZED_HEURISTICS_LOG_FILE = os.path.join(LEARNING_LOG_DIR, "generalized_heuristics.jsonl")

# Placeholder for imports - these might be needed depending on implementation
# from sklearn.cluster import DBSCAN, KMeans
# from sklearn.metrics.pairwise import cosine_similarity

try:
    import faiss
except ImportError:
    faiss = None
    logging.warning("FAISS library not found. Indexing of generalized rules will not work.")

# --- Dependencies --- 
# Embeddings
try:
    from a3x.core.embeddings import get_embedding
    EMBEDDING_FUNCTION_AVAILABLE = True
except ImportError:
    try:
        from sentence_transformers import SentenceTransformer
        DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2' 
        model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        async def get_embedding_st(text: str, **kwargs) -> Optional[List[float]]:
            try:
                embedding = model.encode([text], convert_to_numpy=True)[0]
                return embedding.tolist()
            except Exception as e:
                 logging.error(f"SentenceTransformer embedding failed: {e}")
                 return None
        get_embedding = get_embedding_st
        EMBEDDING_FUNCTION_AVAILABLE = True
        logging.info(f"Using SentenceTransformer model '{DEFAULT_EMBEDDING_MODEL}' for embeddings.")
    except ImportError:
        async def get_embedding(text: str, **kwargs) -> Optional[List[float]]:
            logging.error("No embedding function available. Cannot perform semantic generalization.")
            return None
        EMBEDDING_FUNCTION_AVAILABLE = False

# Clustering & Similarity (sklearn)
try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    DBSCAN = None
    cosine_similarity = None
    np = None
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn or numpy not found. Cannot perform DBSCAN clustering.")

# --- Prompt for Generalizing Heuristics (Simplified based on user request) --- 
GENERALIZATION_PROMPT_TEMPLATE = """
# Generalização de Heurísticas

Abaixo está um grupo de heurísticas consolidadas (regras aprendidas de sucessos ou falhas passadas), agrupadas por similaridade semântica.

Analise o grupo e extraia UMA ÚNICA REGRA GERAL (ou um insight chave) que capture o padrão principal. Seja direto e acionável.

**Heurísticas Agrupadas:**
{heuristic_list_str}

**Regra Geral (ou Insight Chave):**
"""

# <<< ADDED: DBSCAN parameters >>>
DBSCAN_EPS = 0.3 # Max distance for cosine similarity (1 - cosine_sim). Lower = more similar needed.
DBSCAN_MIN_SAMPLES = 2 # Minimum number of heuristics to form a cluster for generalization

# <<< REMOVED old _read_recent_heuristics >>>
# <<< REMOVED old _simple_grouping >>>
# <<< REMOVED old _generate_rule_from_group >>>
# <<< REMOVED old _add_generalized_rule_to_memory >>>

# <<< ADDED: New helper functions for DBSCAN approach >>>
def _read_consolidated_heuristics(log_file_path: Path) -> List[Dict[str, Any]]:
    """Reads representative/unique heuristics from the consolidated log file."""
    valid_heuristics = []
    if not log_file_path.exists():
        logger.warning(f"Consolidated heuristic log file not found at {log_file_path}")
        return []
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        entry = json.loads(line.strip())
                        # Filter for non-redundant entries
                        if entry.get("status") in ["representative", "unique"] and entry.get("heuristic"):
                             # Keep original index for reference if needed, though less critical now
                             entry["_original_consolidated_index"] = i 
                             valid_heuristics.append(entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line {i+1} in {log_file_path.name}")
    except Exception as e:
        logger.exception(f"Error reading consolidated heuristic log file {log_file_path}: {e}")
    logger.info(f"Read {len(valid_heuristics)} non-redundant heuristics from {log_file_path.name}")
    return valid_heuristics

async def _generate_embeddings_for_consolidation(heuristics: List[Dict[str, Any]]) -> Optional[Tuple[np.ndarray, List[int]]]:
    """Generates embeddings specifically for the consolidation task."""
    if not EMBEDDING_FUNCTION_AVAILABLE or not SKLEARN_AVAILABLE:
        logger.error("Dependencies for embedding generation or numpy are missing.")
        return None
        
    texts = [h.get("heuristic", "") for h in heuristics]
    embeddings_list = []
    valid_indices = [] # Indices within the input `heuristics` list
    dimension = None

    logger.info(f"Generating embeddings for {len(texts)} consolidated heuristics...")
    for i, text in enumerate(texts):
        if not text:
            logger.warning(f"Skipping empty heuristic text at index {i}")
            continue
        try:
            emb = await get_embedding(text)
            if emb:
                if dimension is None: dimension = len(emb)
                if len(emb) == dimension:
                    embeddings_list.append(emb)
                    valid_indices.append(i)
                else:
                    logger.warning(f"Dimension mismatch at index {i} ({len(emb)} vs {dimension}). Skipping.")
            else:
                logger.warning(f"Failed to get embedding for heuristic at index {i}")
        except Exception as e:
            logger.exception(f"Error generating embedding for heuristic at index {i}: {e}")

    if not embeddings_list or dimension is None:
        logger.error("Could not generate any valid embeddings for consolidation.")
        return None
        
    logger.info(f"Successfully generated {len(embeddings_list)} embeddings for consolidation (dimension {dimension}).")
    return np.array(embeddings_list).astype('float32'), valid_indices

def _cluster_heuristics_dbscan(embeddings: np.ndarray, eps: float, min_samples: int) -> Tuple[np.ndarray, int]:
    """Clusters heuristic embeddings using DBSCAN with cosine distance."""
    if not SKLEARN_AVAILABLE or DBSCAN is None:
        logger.error("DBSCAN (sklearn) is not available for clustering.")
        return np.array([]), 0
        
    logger.info(f"Clustering {embeddings.shape[0]} embeddings using DBSCAN (eps={eps}, min_samples={min_samples}, metric='cosine')...")
    # DBSCAN uses distance, cosine_similarity gives similarity.
    # metric='cosine' uses 1 - cosine_similarity as distance.
    # So eps should be set based on the maximum ALLOWED distance (e.g., 0.3 means similarity >= 0.7)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    logger.info(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points.")
    
    return labels, n_clusters

async def _generate_rule_from_cluster(cluster_heuristics: List[Dict[str, Any]], llm_url: Optional[str]) -> Optional[str]:
    """Uses LLM to generate a general rule from a cluster of specific heuristics."""
    if len(cluster_heuristics) < DBSCAN_MIN_SAMPLES:
        return None # Should not happen if called correctly
        
    heuristic_texts = [f"- ({h.get('type', 'N/A')}) {h['heuristic']}" for h in cluster_heuristics]
    heuristic_list_str = "\n".join(heuristic_texts)

    prompt = GENERALIZATION_PROMPT_TEMPLATE.format(heuristic_list_str=heuristic_list_str)
    prompt_messages = [{"role": "user", "content": prompt}]

    logger.info(f"Sending cluster of {len(cluster_heuristics)} heuristics to LLM for generalization...")
    llm_response_str = ""
    try:
        # Using stream=False as we expect a single concise rule
        async for chunk in call_llm(prompt_messages, llm_url=llm_url, stream=False, temperature=0.5, max_tokens=150):
             llm_response_str += chunk
        generalized_rule = llm_response_str.strip().strip('"')
        if generalized_rule and len(generalized_rule) > 10:
            logger.info(f"LLM generated generalized rule: {generalized_rule}")
            return generalized_rule
        else:
             logger.warning(f"LLM response for generalization seems empty or invalid: '{llm_response_str}'")
             return None
    except Exception as e:
        logger.exception(f"Error during generalization LLM call: {e}")
        return None

async def _log_generalized_heuristic(rule_text: str, source_cluster_texts: List[str]):
    """Appends a new generalized heuristic to its specific JSONL file."""
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + 'Z',
        "generalized_heuristic": rule_text,
        "cluster_examples": source_cluster_texts, # Store the source texts
        "embedding_source": "consolidated_log"
    }
    try:
        os.makedirs(LEARNING_LOG_DIR, exist_ok=True)
        log_file_path = Path(GENERALIZED_HEURISTICS_LOG_FILE)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        logger.info(f"Logged generalized heuristic to {log_file_path.name}")
    except Exception as e:
        logger.exception(f"Failed to write generalized heuristic to log file: {e}")

# <<< END ADDED: New helper functions >>>

# <<< UPDATED: Skill definition >>>
@skill(
    name="generalize_heuristics",
    description="Analisa heurísticas consolidadas, agrupa semanticamente via DBSCAN e gera regras gerais.",
    parameters={
        "dbscan_eps": (Optional[float], DBSCAN_EPS),
        "min_cluster_size": (Optional[int], DBSCAN_MIN_SAMPLES)
    }
)
# <<< UPDATED: Function signature and logic >>>
async def generalize_heuristics(dbscan_eps: Optional[float] = None, min_cluster_size: Optional[int] = None, ctx: Optional[Context] = None) -> Dict[str, Any]:
    """Analyzes consolidated heuristics, clusters them semantically, and generates general rules."""
    log_prefix = "[GeneralizeHeuristics Skill]"
    eps = dbscan_eps or DBSCAN_EPS
    min_samples = min_cluster_size or DBSCAN_MIN_SAMPLES
    logger.info(f"{log_prefix} Iniciando generalização semântica (eps={eps}, min_samples={min_samples}).")

    # Check dependencies
    if not all([EMBEDDING_FUNCTION_AVAILABLE, SKLEARN_AVAILABLE]):
        msg = "Dependências ausentes (embeddings, sklearn). Não é possível generalizar."
        logger.error(f"{log_prefix} {msg}")
        return {"status": "error", "data": {"message": msg}}

    generated_rules_count = 0
    processed_clusters = 0
    try:
        workspace_root = Path(getattr(ctx, 'workspace_root', '.'))
        input_log_path = workspace_root / CONSOLIDATED_HEURISTIC_LOG_FILE
        llm_url = getattr(ctx, 'llm_url', None) if ctx else None

        # 1. Read CONSOLIDATED heuristics
        valid_heuristics = _read_consolidated_heuristics(input_log_path)
        if len(valid_heuristics) < min_samples:
            msg = f"Não há heurísticas consolidadas suficientes ({len(valid_heuristics)}) para formar clusters (mínimo: {min_samples})."
            logger.info(f"{log_prefix} {msg}")
            return {"status": "success", "action": "generalization_skipped", "data": {"message": msg}}

        # 2. Generate Embeddings
        embedding_result = await _generate_embeddings_for_consolidation(valid_heuristics)
        if embedding_result is None:
            msg = "Falha ao gerar embeddings para as heurísticas consolidadas."
            logger.error(f"{log_prefix} {msg}")
            return {"status": "error", "data": {"message": msg}}
            
        embeddings_array, valid_indices = embedding_result
        # Ensure the valid_heuristics list matches the embeddings array
        # This should be guaranteed if _generate_embeddings returns valid_indices correctly
        if len(valid_indices) != embeddings_array.shape[0]:
             logger.error("Mismatch between valid heuristic count and embedding count after generation.")
             return {"status": "error", "data": {"message": "Internal error during embedding generation."}}
        heuristics_for_clustering = [valid_heuristics[i] for i in valid_indices]

        # 3. Cluster using DBSCAN
        labels, n_clusters = _cluster_heuristics_dbscan(embeddings_array, eps=eps, min_samples=min_samples)
        if n_clusters == 0:
             msg = "Nenhum cluster significativo encontrado pela análise DBSCAN."
             logger.info(f"{log_prefix} {msg}")
             return {"status": "success", "action": "generalization_skipped", "data": {"message": msg}}

        # 4. Process each cluster (excluding noise label -1)
        unique_labels = set(labels) - {-1}
        logger.info(f"{log_prefix} Processando {len(unique_labels)} clusters encontrados...")
        for k in unique_labels:
            cluster_member_indices = [i for i, label in enumerate(labels) if label == k]
            if len(cluster_member_indices) >= min_samples:
                processed_clusters += 1
                cluster_heuristics = [heuristics_for_clustering[i] for i in cluster_member_indices]
                cluster_texts = [h["heuristic"] for h in cluster_heuristics]
                
                # 5. Synthesize rule
                generalized_rule = await _generate_rule_from_cluster(cluster_heuristics, llm_url)
                
                if generalized_rule:
                    # 6. Log the new generalized rule
                    await _log_generalized_heuristic(generalized_rule, cluster_texts)
                    generated_rules_count += 1
            else:
                 # Should be filtered by min_samples in DBSCAN, but good practice to check
                 logger.debug(f"Skipping cluster {k} with size {len(cluster_member_indices)} (below min_samples {min_samples}).")
                 
        msg = f"Processo de generalização concluído. {generated_rules_count} novas regras gerais geradas a partir de {processed_clusters} clusters significativos."
        logger.info(f"{log_prefix} {msg}")
        return {"status": "success", "action": "generalization_completed", "data": {"message": msg, "generated_rules_count": generated_rules_count, "clusters_processed": processed_clusters}}

    except Exception as e:
        logger.exception(f"{log_prefix} Erro durante a generalização semântica de heurísticas:")
        return {"status": "error", "action": "generalization_error", "data": {"message": f"Erro: {e}"}}

# <<< UPDATED: Example Test Block >>>
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    async def run_main_test():
        print("\n--- Running Semantic Generalize Heuristics Test --- ")
        # Ensure CONSOLIDATED_HEURISTIC_LOG_FILE exists with representative/unique entries
        if not os.path.exists(LEARNING_LOG_DIR):
             os.makedirs(LEARNING_LOG_DIR)
        # Example: Use the output from the consolidate_heuristics test if available,
        # or create a dummy consolidated file here.
        dummy_consolidated_heuristics = [
            {"timestamp": "2023-10-27T09:00:00Z", "type": "failure", "heuristic": "Verifique se o arquivo existe antes de tentar abri-lo para leitura.", "status": "representative", "context_snapshot": {}},
            {"timestamp": "2023-10-27T15:00:00Z", "type": "failure", "heuristic": "Cuidado com espaços vs tabs em YAML para evitar erros de parse.", "status": "representative", "context_snapshot": {}},
            {"timestamp": "2023-10-28T10:00:00Z", "type": "success", "heuristic": "Usar list_dir antes de delete_file previne erros.", "status": "unique", "context_snapshot": {}},
            # Add more for better testing if needed
        ]
        input_path = Path(CONSOLIDATED_HEURISTIC_LOG_FILE)
        with open(input_path, 'w', encoding='utf-8') as f:
            for entry in dummy_consolidated_heuristics:
                f.write(json.dumps(entry) + '\n')
        logger.info(f"Created dummy consolidated heuristic file: {input_path.name}")
        
        # Clear previous generalized rules log for clean test
        generalized_log_path = Path(GENERALIZED_HEURISTICS_LOG_FILE)
        if generalized_log_path.exists(): os.remove(generalized_log_path)

        result = await generalize_heuristics() # Use default params
        print("\n--- Generalization Result --- ")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Check output file
        if generalized_log_path.exists():
            print(f"\n--- Generalized Heuristics Log ({generalized_log_path.name}) Content: ---")
            with open(generalized_log_path, 'r', encoding='utf-8') as f_out:
                for line in f_out:
                    print(line.strip()) 
        else:
             print(f"\n--- Generalized Heuristics Log ({generalized_log_path.name}) was not created. ---")

    import asyncio
    # Only run test if dependencies seem available
    if all([EMBEDDING_FUNCTION_AVAILABLE, SKLEARN_AVAILABLE]):
        asyncio.run(run_main_test())
    else:
        print("Skipping generalize_heuristics test due to missing dependencies (embeddings or sklearn).") 