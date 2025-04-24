import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
import datetime
import itertools

# --- Dependencies --- 
# Embeddings (using the project's utility or directly)
try:
    # Try project's utility first
    from a3x.core.embeddings import get_embedding
    EMBEDDING_FUNCTION_AVAILABLE = True
except ImportError:
    # Fallback to trying sentence-transformers directly
    try:
        from sentence_transformers import SentenceTransformer
        # Load a default model - consider making this configurable
        # Using a smaller, faster model suitable for CPU inference if needed
        DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2' 
        model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        async def get_embedding_st(text: str, **kwargs) -> Optional[List[float]]:
            try:
                # SentenceTransformer encode might be synchronous
                # Run in executor if in async context (TBD based on integration)
                # For simplicity here, assuming direct call is okay or handled upstream
                embedding = model.encode([text], convert_to_numpy=True)[0]
                return embedding.tolist()
            except Exception as e:
                 logging.error(f"SentenceTransformer embedding failed: {e}")
                 return None
        get_embedding = get_embedding_st # Assign the fallback function
        EMBEDDING_FUNCTION_AVAILABLE = True
        logging.info(f"Using SentenceTransformer model '{DEFAULT_EMBEDDING_MODEL}' for embeddings.")
    except ImportError:
        # Define a dummy embedding function if nothing is available
        async def get_embedding(text: str, **kwargs) -> Optional[List[float]]:
            logging.error("No embedding function available (a3x.core.embeddings or sentence-transformers). Cannot perform semantic consolidation.")
            return None
        EMBEDDING_FUNCTION_AVAILABLE = False

# Similarity Matrix (sklearn)
try:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    cosine_similarity = None
    np = None
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn or numpy not found. Cannot calculate similarity matrix.")

# Graph Processing (networkx)
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    NETWORKX_AVAILABLE = False
    logging.warning("networkx library not found. Cannot perform graph-based clustering.")

# Skill definition utilities
try:
    from a3x.core.skills import skill
    from a3x.core.context import Context
    from a3x.core.learning_logs import save_learning_log
except ImportError:
    # Fallback for standalone testing
    def skill(**kwargs):
        def decorator(func):
            return func
        return decorator
    Context = Any
# --- End Dependencies ---

logger = logging.getLogger(__name__)

# --- Configuration ---
# How many recent raw heuristic logs to consider
MAX_LOGS_TO_PROCESS = 100
# Output file for the consolidated summary
# Paths relative to project root
LEARNING_LOG_DIR = "data/memory/learning_logs" # UPDATED PATH
CONSOLIDATED_LOG_FILE = os.path.join(LEARNING_LOG_DIR, "learned_heuristics_consolidated.jsonl")
RAW_HEURISTIC_LOG_FILE = os.path.join(LEARNING_LOG_DIR, "learned_heuristics.jsonl") # Source file

# Threshold for considering heuristics similar enough to group
DEFAULT_SIMILARITY_THRESHOLD = 0.90 # Adjust based on model and desired granularity

def _read_heuristics(log_file_path: Path) -> List[Dict[str, Any]]:
    """Reads all valid heuristic entries from the JSONL log file."""
    heuristics = []
    if not log_file_path.exists():
        logger.warning(f"Heuristic log file not found at {log_file_path}")
        return []
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        entry = json.loads(line.strip())
                        # Basic validation: ensure it has a heuristic text and type
                        if entry.get("heuristic") and entry.get("type") in ["success", "failure"]:
                             # Add an original index for later reference
                             entry["_original_index"] = i
                             heuristics.append(entry)
                        # Optionally handle generalized rules if needed
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line {i+1} in {log_file_path.name}")
    except Exception as e:
        logger.exception(f"Error reading heuristic log file {log_file_path}: {e}")
    logger.info(f"Read {len(heuristics)} heuristics from {log_file_path.name}")
    return heuristics

async def _generate_embeddings(heuristics: List[Dict[str, Any]]) -> Optional[Tuple[np.ndarray, List[int]]]: # Return valid indices too
    """Generates embeddings for a list of heuristics."""
    if not EMBEDDING_FUNCTION_AVAILABLE or not SKLEARN_AVAILABLE:
        logger.error("Dependencies for embedding generation or numpy are missing.")
        return None
        
    texts = [h.get("heuristic", "") for h in heuristics]
    embeddings_list = []
    valid_indices = []
    dimension = None

    logger.info(f"Generating embeddings for {len(texts)} heuristics...")
    for i, text in enumerate(texts):
        original_heuristic_index = heuristics[i].get('_original_index', i) # Get original index if available
        if not text:
            logger.warning(f"Skipping empty heuristic text at original index {original_heuristic_index}")
            continue
        try:
            emb = get_embedding(text)
            if emb:
                if dimension is None: dimension = len(emb)
                if len(emb) == dimension:
                    embeddings_list.append(emb)
                    valid_indices.append(i) # Track index within the current list `heuristics`
                else:
                    logger.warning(f"Dimension mismatch for embedding at original index {original_heuristic_index} ({len(emb)} vs {dimension}). Skipping.")
            else:
                logger.warning(f"Failed to get embedding for heuristic at original index {original_heuristic_index}")
        except Exception as e:
            logger.exception(f"Error generating embedding for heuristic at original index {original_heuristic_index}: {e}")

    if not embeddings_list or dimension is None:
        logger.error("Could not generate any valid embeddings.")
        return None
        
    logger.info(f"Successfully generated {len(embeddings_list)} embeddings with dimension {dimension}.")
    # Return embeddings and the indices within the `heuristics` list they correspond to
    return np.array(embeddings_list).astype('float32'), valid_indices

def _find_similar_clusters(embeddings: np.ndarray, threshold: float) -> List[Set[int]]:
    """Finds clusters of similar heuristics using graph components based on cosine similarity."""
    if not SKLEARN_AVAILABLE or not NETWORKX_AVAILABLE or cosine_similarity is None or nx is None:
        logger.error("Dependencies for clustering (sklearn, networkx) are missing.")
        return []
        
    logger.info(f"Calculating similarity matrix for {embeddings.shape[0]} embeddings...")
    # Calculate cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    logger.debug(f"Similarity matrix shape: {sim_matrix.shape}")

    # Build graph
    logger.info(f"Building similarity graph with threshold >= {threshold}...")
    graph = nx.Graph()
    num_nodes = sim_matrix.shape[0]
    graph.add_nodes_from(range(num_nodes))

    # Add edges where similarity exceeds the threshold
    edge_count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes): # Avoid self-loops and duplicates
            if sim_matrix[i, j] >= threshold:
                graph.add_edge(i, j, weight=sim_matrix[i, j])
                edge_count += 1
                
    logger.info(f"Graph built with {num_nodes} nodes and {edge_count} edges.")

    # Find connected components (clusters)
    clusters = list(nx.connected_components(graph))
    logger.info(f"Found {len(clusters)} connected components (clusters).")
    # Filter out single-element clusters as they are not redundant
    non_singleton_clusters = [cluster for cluster in clusters if len(cluster) > 1]
    logger.info(f"Found {len(non_singleton_clusters)} clusters with potential redundancy (size > 1).")
    
    return non_singleton_clusters

def _select_representative(cluster_indices: Set[int], heuristics_subset: List[Dict[str, Any]]) -> int:
    """Selects a representative heuristic index (relative to the subset) from a cluster."""
    # Strategy: Choose the most recent heuristic in the cluster
    latest_time = None
    representative_subset_idx = -1

    for subset_idx in cluster_indices: # subset_idx refers to index within embeddings/heuristics_subset
         if subset_idx >= len(heuristics_subset):
              logger.warning(f"Cluster index {subset_idx} out of bounds for heuristics subset (size {len(heuristics_subset)}). Skipping.")
              continue
              
         heuristic = heuristics_subset[subset_idx]
         timestamp_str = heuristic.get("timestamp")
         original_idx = heuristic.get("_original_index", "N/A") # For logging
         
         try:
             if timestamp_str:
                 # Clean timestamp before parsing: Remove trailing 'Z' if present
                 cleaned_timestamp_str = timestamp_str.rstrip('Z')
                 current_time = datetime.datetime.fromisoformat(cleaned_timestamp_str)
                 if current_time.tzinfo:
                     current_time = current_time.astimezone(datetime.timezone.utc).replace(tzinfo=None)
                 
                 if latest_time is None or current_time > latest_time:
                     latest_time = current_time
                     representative_subset_idx = subset_idx
             elif representative_subset_idx == -1: # If no timestamp, pick the first one as fallback
                 representative_subset_idx = subset_idx
         except Exception as e:
              logger.warning(f"Could not parse timestamp '{timestamp_str}' for heuristic at original index {original_idx}: {e}. Skipping for representative selection.")
              if representative_subset_idx == -1: # Fallback if parsing fails
                  representative_subset_idx = subset_idx
                  
    if representative_subset_idx == -1:
         # Should not happen if cluster is not empty, but fallback
         representative_subset_idx = next(iter(cluster_indices))
         logger.warning(f"Could not reliably select representative for cluster {cluster_indices}, using first element's subset index: {representative_subset_idx}")

    return representative_subset_idx

@skill(
    name="consolidate_heuristics",
    description="Analisa heurísticas aprendidas, identifica redundâncias semânticas e gera um log consolidado.",
    parameters={
        "similarity_threshold": {"type": Optional[float], "description": "O limiar de similaridade para considerar heurísticas como redundantes.", "default": DEFAULT_SIMILARITY_THRESHOLD}
    }
)
async def consolidate_heuristics(ctx: Any, similarity_threshold: Optional[float] = None) -> Dict[str, Any]:
    """Consolidates learned heuristics by identifying and marking semantic duplicates."""
    log_prefix = "[ConsolidateHeuristics Skill]"
    threshold = similarity_threshold or DEFAULT_SIMILARITY_THRESHOLD
    logger.info(f"{log_prefix} Iniciando consolidação de heurísticas com limiar >= {threshold}")

    # Check dependencies
    if not all([EMBEDDING_FUNCTION_AVAILABLE, SKLEARN_AVAILABLE, NETWORKX_AVAILABLE]):
        msg = "Dependências ausentes (embeddings, sklearn, networkx). Não é possível consolidar."
        logger.error(f"{log_prefix} {msg}")
        return {"status": "error", "data": {"message": msg}}

    try:
        workspace_root = Path(getattr(ctx, 'workspace_root', '.'))
        input_log_path = workspace_root / RAW_HEURISTIC_LOG_FILE
        output_log_path = workspace_root / CONSOLIDATED_LOG_FILE
        # <<< CORREÇÃO: Garantir que o diretório exista >>>
        output_log_dir = output_log_path.parent
        try:
            output_log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"{log_prefix} Diretório de log garantido: {output_log_dir}")
        except OSError as e:
            logger.error(f"{log_prefix} Falha ao criar diretório de log {output_log_dir}: {e}")
            return {"status": "error", "data": {"message": f"Falha ao criar diretório de log: {e}"}}
        # <<< FIM CORREÇÃO >>>

        # 1. Read heuristics
        all_heuristics = _read_heuristics(input_log_path)
        if not all_heuristics:
            msg = "Nenhuma heurística encontrada para consolidar."
            logger.info(f"{log_prefix} {msg}")
            return {"status": "success", "data": {"message": msg, "consolidated_count": 0, "redundant_count": 0}}

        # 2. Generate Embeddings for valid heuristics
        embeddings_result = await _generate_embeddings(all_heuristics)
        if embeddings_result is None:
             msg = "Falha ao gerar embeddings para as heurísticas."
             logger.error(f"{log_prefix} {msg}")
             return {"status": "error", "data": {"message": msg}}
             
        embeddings_array, valid_list_indices = embeddings_result
        # Create the subset of heuristics that have embeddings
        valid_heuristics_subset = [all_heuristics[i] for i in valid_list_indices]

        # 3. Find Clusters based on the embeddings of the valid subset
        # Indices returned by _find_similar_clusters refer to positions within embeddings_array/valid_heuristics_subset
        clusters_in_subset = _find_similar_clusters(embeddings_array, threshold)

        # 4. Mark Redundancy
        # Keep track of which original heuristics (_original_index) are marked redundant or representative
        redundancy_map: Dict[int, int] = {} # Map from redundant original_index -> representative original_index
        representatives: Set[int] = set() # Set of representative original_indices
        processed_original_indices: Set[int] = set() # Track processed original indices

        for cluster_subset_indices in clusters_in_subset:
            # Select representative index relative to the subset
            representative_subset_idx = _select_representative(cluster_subset_indices, valid_heuristics_subset)
            if representative_subset_idx == -1 or representative_subset_idx >= len(valid_heuristics_subset):
                 logger.error(f"Invalid representative index {representative_subset_idx} selected for cluster {cluster_subset_indices}. Skipping cluster.")
                 continue
                 
            # Get the original index of the representative
            representative_original_idx = valid_heuristics_subset[representative_subset_idx].get("_original_index")
            if representative_original_idx is None:
                 logger.error(f"Representative heuristic (subset index {representative_subset_idx}) missing _original_index. Skipping cluster.")
                 continue
            
            representatives.add(representative_original_idx)
            processed_original_indices.add(representative_original_idx)

            # Mark others in the cluster as redundant
            for subset_idx in cluster_subset_indices:
                if subset_idx != representative_subset_idx:
                    if subset_idx >= len(valid_heuristics_subset):
                         logger.warning(f"Cluster index {subset_idx} out of bounds when marking redundancy. Skipping.")
                         continue
                    redundant_original_idx = valid_heuristics_subset[subset_idx].get("_original_index")
                    if redundant_original_idx is not None:
                        redundancy_map[redundant_original_idx] = representative_original_idx
                        processed_original_indices.add(redundant_original_idx)
                    else:
                         logger.warning(f"Redundant heuristic (subset index {subset_idx}) missing _original_index. Cannot mark.")

        # 5. Write Consolidated Log
        consolidated_count = 0
        redundant_count = len(redundancy_map)
        unique_count = 0
        skipped_count = 0
        logger.info(f"{log_prefix} Escrevendo log consolidado em {output_log_path.name}...")
        try:
            os.makedirs(output_log_path.parent, exist_ok=True)
            with open(output_log_path, 'w', encoding='utf-8') as f_out:
                for heuristic in all_heuristics:
                    original_idx = heuristic.pop("_original_index") # Remove temporary index

                    if original_idx in redundancy_map:
                        heuristic["status"] = "redundant"
                        heuristic["consolidated_into_original_index"] = redundancy_map[original_idx]
                    elif original_idx in representatives:
                        heuristic["status"] = "representative"
                        consolidated_count += 1
                    elif original_idx not in processed_original_indices:
                        # Check if it was skipped due to missing embedding
                        was_valid = original_idx in [all_heuristics[i].get('_original_index') for i in valid_list_indices]
                        if was_valid:
                             heuristic["status"] = "unique"
                             consolidated_count += 1 
                             unique_count +=1
                        else:
                             heuristic["status"] = "skipped_no_embedding" # Mark explicitly
                             skipped_count += 1
                    else:
                         # Should not happen if logic is correct (processed but not redundant/representative)
                         logger.warning(f"Heuristic with original index {original_idx} has unclear status. Marking as skipped.")
                         heuristic["status"] = "skipped_unclear"
                         skipped_count += 1
                        
                    f_out.write(json.dumps(heuristic, ensure_ascii=False) + '\n')
            logger.info(f"{log_prefix} Log consolidado escrito. {consolidated_count} mantidas ({unique_count} únicas + {len(representatives)} representantes), {redundant_count} redundantes, {skipped_count} puladas.")
        except IOError as e:
             logger.exception(f"{log_prefix} Erro ao escrever arquivo de log consolidado: {e}")
             return {"status": "error", "data": {"message": f"Erro ao escrever log consolidado: {e}"}}

        msg = f"Consolidação concluída. {consolidated_count} mantidas, {redundant_count} redundantes, {skipped_count} puladas. Log salvo em {output_log_path.name}."
        return {"status": "success", "data": {"message": msg, "consolidated_count": consolidated_count, "redundant_count": redundant_count, "skipped_count": skipped_count}}

    except Exception as e:
        logger.exception(f"{log_prefix} Erro inesperado durante a consolidação de heurísticas:")
        return {"status": "error", "data": {"message": f"Erro inesperado: {e}"}}

# Example Test Block
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    async def run_main_test():
        print("\n--- Running Consolidate Heuristics Test --- ")
        # 1. Create a richer dummy heuristic log file
        if not os.path.exists(LEARNING_LOG_DIR):
             os.makedirs(LEARNING_LOG_DIR)
        dummy_heuristics = [
            # Group 1: File reading errors
            {"timestamp": "2023-10-26T10:00:00Z", "type": "failure", "heuristic": "Ao ler arquivos, trate exceções FileNotFoundError.", "context_snapshot": {}},
            {"timestamp": "2023-10-27T09:00:00Z", "type": "failure", "heuristic": "Verifique se o arquivo existe antes de tentar abri-lo para leitura.", "context_snapshot": {}},
            {"timestamp": "2023-10-26T11:00:00Z", "type": "failure", "heuristic": "Se read_file falhar com 'arquivo não encontrado', verifique o path.", "context_snapshot": {}},
            # Group 2: YAML editing
            {"timestamp": "2023-10-27T15:00:00Z", "type": "failure", "heuristic": "Cuidado com espaços vs tabs em YAML para evitar erros de parse.", "context_snapshot": {}},
            {"timestamp": "2023-10-26T14:00:00Z", "type": "failure", "heuristic": "A indentação é crucial ao editar arquivos YAML.", "context_snapshot": {}},
             # Unique
            {"timestamp": "2023-10-28T10:00:00Z", "type": "success", "heuristic": "Usar list_dir antes de delete_file previne erros.", "context_snapshot": {}},
        ]
        input_path = Path(RAW_HEURISTIC_LOG_FILE)
        from a3x.core.learning_logs import log_heuristic_with_traceability
        # Limpa o arquivo antes de inserir exemplos
        if input_path.exists():
            input_path.unlink()
        for i, entry in enumerate(dummy_heuristics):
            plan_id = f"consolidate-seed-plan-{i+1}"
            execution_id = f"consolidate-seed-exec-{i+1}"
            log_heuristic_with_traceability(entry, plan_id, execution_id, validation_status="seed")
        logger.info(f"Created dummy heuristic file: {input_path.name}")

        # 2. Run the skill
        # Use a high threshold first to see unique items, then lower to see consolidation
        # result_high_thresh = await consolidate_heuristics(similarity_threshold=0.98)
        # print("\n--- Consolidation Result (High Threshold) --- ")
        # print(json.dumps(result_high_thresh, indent=2, ensure_ascii=False))

        result_lower_thresh = await consolidate_heuristics(similarity_threshold=0.85) # Lower threshold
        print("\n--- Consolidation Result (Lower Threshold) --- ")
        print(json.dumps(result_lower_thresh, indent=2, ensure_ascii=False))

        # 3. Check output file (optional)
        output_path = Path(CONSOLIDATED_LOG_FILE)
        if output_path.exists():
            print(f"\n--- Consolidated Log ({output_path.name}) Content: ---")
            with open(output_path, 'r', encoding='utf-8') as f_out:
                for line in f_out:
                    print(line.strip())
        else:
             print(f"\n--- Consolidated Log ({output_path.name}) was not created. ---")

    import asyncio
    # Only run test if dependencies seem available
    if all([EMBEDDING_FUNCTION_AVAILABLE, SKLEARN_AVAILABLE, NETWORKX_AVAILABLE]):
        asyncio.run(run_main_test())
    else:
        print("Skipping consolidate_heuristics test due to missing dependencies (embeddings, sklearn, or networkx).") 