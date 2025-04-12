# Start of new file 
import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import datetime

# --- Dependencies --- 
# FAISS for vector search
try:
    import faiss
except ImportError:
    faiss = None # Indicate FAISS is not available
    logging.warning("FAISS library not found. Semantic search in generalized rules will not work.")

# Embeddings (using the project's utility)
try:
    from a3x.core.embeddings import get_embedding
except ImportError:
    # Define a dummy embedding function if the real one isn't available
    async def get_embedding(text: str, **kwargs) -> Optional[List[float]]:
        logging.error("Embedding function 'a3x.core.embeddings.get_embedding' not found. Cannot perform semantic search.")
        return None

# Skill definition utilities
try:
    from a3x.core.skill_interface import skill
    from a3x.core.context import Context
except ImportError:
    # Fallback for standalone testing
    def skill(**kwargs):
        def decorator(func):
            return func
        return decorator
    Context = Any
# --- End Dependencies ---

logger = logging.getLogger(__name__)

# Constants
LEARNING_LOG_DIR = "memory/learning_logs"
GENERALIZED_RULES_FILE = os.path.join(LEARNING_LOG_DIR, "generalized_rules.jsonl")
FAISS_INDEX_FILE = os.path.join(LEARNING_LOG_DIR, "generalized_rules.faissindex")

# Global cache for index and metadata (simple version)
# In a real application, consider more robust caching or a dedicated memory manager
_index_cache = {
    "index": None,
    "metadata": None, # List of dicts from JSONL, index corresponds to FAISS ID
    "index_mtime": None,
    "metadata_mtime": None
}

async def _load_index_and_metadata(index_path: Path, metadata_path: Path) -> Tuple[Optional[Any], Optional[List[Dict]]]:
    """Loads FAISS index and corresponding metadata, using simple mtime caching."""
    global _index_cache
    index = None
    metadata = []

    # Check metadata file first
    if not metadata_path.exists() or not index_path.exists():
        logger.debug(f"Index ({index_path.name}) or metadata ({metadata_path.name}) file not found.")
        _index_cache["index"] = None
        _index_cache["metadata"] = None
        _index_cache["index_mtime"] = None
        _index_cache["metadata_mtime"] = None
        return None, None

    current_metadata_mtime = metadata_path.stat().st_mtime
    current_index_mtime = index_path.stat().st_mtime

    # Check cache validity
    if (
        _index_cache["index"] is not None and
        _index_cache["metadata"] is not None and
        _index_cache["metadata_mtime"] == current_metadata_mtime and
        _index_cache["index_mtime"] == current_index_mtime
    ):
        logger.debug("Using cached generalized rules index and metadata.")
        return _index_cache["index"], _index_cache["metadata"]

    # Load metadata (JSONL)
    logger.info(f"Loading generalized rules metadata from {metadata_path}...")
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        metadata.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in {metadata_path.name}: {line.strip()}")
        logger.info(f"Loaded {len(metadata)} rules from metadata.")
    except Exception as e:
        logger.exception(f"Error loading metadata file {metadata_path}: {e}")
        return None, None # Cannot proceed without metadata

    # Load FAISS index
    logger.info(f"Loading FAISS index from {index_path}...")
    if not faiss:
        logger.error("FAISS library is not installed. Cannot load index.")
        return None, metadata # Return metadata only, search won't work
        
    try:
        index = faiss.read_index(str(index_path))
        logger.info(f"FAISS index loaded successfully. Contains {index.ntotal} vectors.")
        if index.ntotal != len(metadata):
             logger.warning(f"FAISS index size ({index.ntotal}) does not match metadata size ({len(metadata)}). Search results might be inconsistent.")
             # Decide how to handle mismatch: error out, proceed with caution?
             # For now, proceed but log warning.
    except Exception as e:
        logger.exception(f"Error loading FAISS index {index_path}: {e}")
        # If index fails to load, clear cache to force reload next time
        _index_cache["index"] = None
        _index_cache["index_mtime"] = None
        return None, metadata # Return metadata only

    # Update cache
    _index_cache["index"] = index
    _index_cache["metadata"] = metadata
    _index_cache["index_mtime"] = current_index_mtime
    _index_cache["metadata_mtime"] = current_metadata_mtime

    return index, metadata

@skill(
    name="consult_generalized_rules",
    description="Busca regras gerais aprendidas relevantes para o objetivo atual usando busca semântica vetorial (FAISS).",
    parameters=[
        {"name": "objective", "type": "string", "description": "O objetivo atual para o qual buscar regras relevantes."},
        {"name": "top_k", "type": "int", "description": "Número máximo de regras a retornar.", "optional": True, "default": 3},
        {"name": "ctx", "type": "Context", "description": "Objeto de contexto da execução.", "optional": True}
    ]
)
async def consult_generalized_rules(objective: str, top_k: int = 3, ctx: Optional[Context] = None) -> Dict[str, Any]:
    """Searches for relevant generalized rules using semantic vector search."""
    log_prefix = "[ConsultGeneralizedRules Skill]"
    logger.info(f"{log_prefix} Consultando regras gerais para objetivo '{objective[:50]}...' (top_k={top_k})" )

    if not faiss:
        msg = "FAISS library not available. Cannot perform semantic search."
        logger.error(f"{log_prefix} {msg}")
        return {"status": "error", "data": {"message": msg}}

    relevant_rules = []
    try:
        workspace_root = Path(getattr(ctx, 'workspace_root', '.'))
        index_path = workspace_root / FAISS_INDEX_FILE
        metadata_path = workspace_root / GENERALIZED_RULES_FILE

        # 1. Load index and metadata (uses caching)
        index, metadata = await _load_index_and_metadata(index_path, metadata_path)

        if index is None or not metadata:
            msg = "Índice FAISS ou metadados de regras gerais não disponíveis ou vazios."
            logger.info(f"{log_prefix} {msg}")
            return {"status": "success", "data": {"rules": [], "message": msg}}
        
        if index.ntotal == 0:
            msg = "Índice de regras gerais está vazio."
            logger.info(f"{log_prefix} {msg}")
            return {"status": "success", "data": {"rules": [], "message": msg}}

        # 2. Get embedding for the objective
        logger.debug(f"{log_prefix} Gerando embedding para o objetivo...")
        query_embedding = await get_embedding(objective)

        if query_embedding is None:
            msg = "Falha ao gerar embedding para o objetivo. Não é possível realizar a busca."
            logger.error(f"{log_prefix} {msg}")
            return {"status": "error", "data": {"message": msg}}
        
        # Ensure embedding is in the correct format (numpy array, float32)
        try:
             import numpy as np
        except ImportError:
             logger.error("Numpy library not found. Cannot perform FAISS search.")
             return {"status": "error", "data": {"message": "Numpy not installed."}}
        query_vector = np.array([query_embedding]).astype('float32') # FAISS expects a 2D array

        # 3. Perform FAISS search
        logger.debug(f"{log_prefix} Realizando busca FAISS com top_k={top_k}...")
        # Adjust k if it's larger than the number of items in the index
        actual_k = min(top_k, index.ntotal)
        distances, indices = index.search(query_vector, actual_k)

        if indices.size == 0 or distances.size == 0:
             logger.info(f"{log_prefix} Busca FAISS não retornou resultados.")
             return {"status": "success", "data": {"rules": [], "message": "Nenhuma regra similar encontrada."}}

        # 4. Retrieve rules from metadata using indices
        found_indices = indices[0] # search returns a 2D array of indices
        found_distances = distances[0]
        
        for i, idx in enumerate(found_indices):
            # FAISS can return -1 for indices if k > ntotal, even after adjusting k in some cases?
            # Also check bounds against actual metadata length
            if idx < 0 or idx >= len(metadata): 
                logger.warning(f"{log_prefix} Índice FAISS inválido ({idx}) retornado para k={actual_k}, ntotal={index.ntotal}, metadata_len={len(metadata)}. Ignorando.")
                continue
            # Assuming metadata list index corresponds to FAISS ID
            rule_entry = metadata[idx]
            rule_text = rule_entry.get("rule")
            if rule_text:
                # Convert distance to similarity (assuming L2 distance normalized embeddings or just as ranking score)
                # A simple inverse or 1-dist/(1+dist) can work for ranking
                similarity_score = 1.0 / (1.0 + float(found_distances[i]))
                relevant_rules.append({
                    "rule": rule_text,
                    "similarity_score": similarity_score
                })
                logger.debug(f"{log_prefix} Regra relevante encontrada (Índice: {idx}, Score: {similarity_score:.4f}): {rule_text[:80]}...")
            else:
                 logger.warning(f"{log_prefix} Metadado para índice {idx} não contém texto da regra.")

        logger.info(f"{log_prefix} Consulta concluída. {len(relevant_rules)} regras gerais relevantes encontradas.")

        # Sort by similarity score (higher is better)
        relevant_rules.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Return only the rule texts for now, unless score is needed downstream
        rule_texts_only = [r["rule"] for r in relevant_rules]

        return {
            "status": "success",
            "data": {"rules": rule_texts_only}
        }

    except FileNotFoundError:
         logger.warning(f"{log_prefix} Arquivo de índice ou metadados não encontrado.")
         return {"status": "success", "data": {"rules": [], "message": "Arquivos de regras gerais não encontrados."}}
    except Exception as e:
        logger.exception(f"{log_prefix} Erro ao consultar regras gerais:")
        return {"status": "error", "data": {"message": f"Erro ao consultar regras gerais: {e}"}}

# Example Test Block
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # Requires dummy log files and potentially a FAISS index

    async def run_main_test():
        print("\n--- Running Consult Generalized Rules Test --- ")
        # 1. Create dummy generalized rules file
        if not os.path.exists(LEARNING_LOG_DIR):
             os.makedirs(LEARNING_LOG_DIR)
        dummy_rules = [
            {"timestamp": datetime.datetime.utcnow().isoformat()+"Z", "rule": "Sempre valide os parâmetros das skills antes de executar.", "source_heuristic_count": 5},
            {"timestamp": datetime.datetime.utcnow().isoformat()+"Z", "rule": "Use list_dir para verificar a existência de arquivos antes de lê-los.", "source_heuristic_count": 3},
            {"timestamp": datetime.datetime.utcnow().isoformat()+"Z", "rule": "Ao editar arquivos YAML, preste atenção especial à indentação.", "source_heuristic_count": 4},
            {"timestamp": datetime.datetime.utcnow().isoformat()+"Z", "rule": "Para criar diretórios aninhados, use 'mkdir -p' no terminal.", "source_heuristic_count": 2}
        ]
        with open(GENERALIZED_RULES_FILE, 'w', encoding='utf-8') as f:
            for rule in dummy_rules:
                f.write(json.dumps(rule) + '\n')
        logger.info(f"Created dummy rules file: {GENERALIZED_RULES_FILE}")

        # 2. Create dummy FAISS index (requires embeddings)
        faiss_index_created = False
        if faiss:
            logger.info("Attempting to create dummy FAISS index...")
            rule_texts = [r["rule"] for r in dummy_rules]
            embeddings = []
            dimension = None
            for text in rule_texts:
                 emb = await get_embedding(text)
                 if emb:
                     if dimension is None:
                         # Assuming get_embedding returns a list of floats
                         dimension = len(emb)
                         logger.info(f"Deduced embedding dimension: {dimension}")
                     if len(emb) == dimension:
                         embeddings.append(emb)
                     else:
                          logger.warning(f"Skipping embedding with wrong dimension ({len(emb)} vs {dimension}) for text: {text[:30]}")
                          # Add a zero vector of the correct dimension as placeholder
                          embeddings.append([0.0] * dimension)
                 else:
                     logger.error(f"Failed to get embedding for rule: {text[:30]}...")
                     # Use a placeholder dimension if none was set yet
                     if dimension is None: dimension = 384 # Default fallback
                     embeddings.append([0.0] * dimension) # Add dummy if fails

            if embeddings and dimension:
                try:
                    import numpy as np
                    embeddings_np = np.array(embeddings).astype('float32')
                    # Simple index, e.g., IndexFlatL2 or IndexFlatIP for cosine similarity
                    # Using IndexFlatIP for cosine similarity if embeddings are normalized
                    # Assuming embeddings ARE normalized by the get_embedding function
                    index = faiss.IndexFlatIP(dimension)
                    index.add(embeddings_np)
                    faiss.write_index(index, FAISS_INDEX_FILE)
                    logger.info(f"Created dummy FAISS index ({index.ntotal} vectors): {FAISS_INDEX_FILE}")
                    faiss_index_created = True
                except Exception as idx_e:
                    logger.exception(f"Failed to create dummy FAISS index: {idx_e}")
            else:
                 logger.error("Could not generate embeddings or determine dimension for dummy index.")
        else:
             logger.warning("FAISS not installed, skipping dummy index creation.")

        # 3. Run the skill test (only if index was created)
        if faiss_index_created:
            test_objective = "Como garantir que a edição de arquivos de configuração seja segura?"
            # Use a dummy context if needed
            class DummyCtx: workspace_root = '.'
            dummy_ctx = DummyCtx()

            result = await consult_generalized_rules(objective=test_objective, top_k=2, ctx=dummy_ctx)
            print("\n--- Consultation Result --- ")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
             print("\nSkipping consultation test as FAISS index was not created.")

    import asyncio
    # Only run test if faiss seems available
    if faiss:
        asyncio.run(run_main_test())
    else:
        print("FAISS not found, skipping consult_generalized_rules test.") 