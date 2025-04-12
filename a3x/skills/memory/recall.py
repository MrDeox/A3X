import sqlite3
import struct  # Para converter vetor float em bytes
import logging
from typing import Dict, Any

# Importar utils - Use absolute paths
from a3x.core.db_utils import get_db_connection
from a3x.core.embeddings import get_embedding, EMBEDDING_DIM  # Importa dimensão também
from a3x.core.semantic_memory_backend import search_index, init_index
import os
from a3x.core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)  # Usar logger específico do módulo

# Definir caminho base do índice FAISS
# É melhor centralizar isso, mas por enquanto definimos aqui
DEFAULT_INDEX_BASE_PATH = os.path.join(PROJECT_ROOT, "a3x", "memory", "indexes", "semantic_memory")

def skill_recall_memory(action_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Busca no índice FAISS externo por conteúdos similares à query fornecida.

    Args:
        action_input (dict): Dicionário contendo a query e opções.
            Exemplo:
                {"query": "Qual o lembrete?", "max_results": 5}

    Returns:
        dict: Dicionário padronizado com status e resultados.
            Os resultados são os metadados armazenados (incluindo o 'content')
            e a distância L2.
    """
    logger.info("\n[Skill: Recall Memory (FAISS)]")
    logger.debug(f"  Action Input: {action_input}")

    # Case-insensitive check for 'query' key
    query = None
    for key, value in action_input.items():
        if key.lower() == "query":
            query = value
            logger.info(f"  Found query using case-insensitive key: '{key}'")
            break

    max_results = action_input.get("max_results", 3)  # Default 3 se não fornecido

    if not query:
        logger.error(
            "  Falha ao recuperar memória: Parâmetro 'query' obrigatório ausente."
        )
        return {
            "status": "error",
            "action": "recall_memory_failed",
            "data": {"message": "Parâmetro 'query' obrigatório ausente."},
        }

    # Validação básica de max_results
    try:
        max_results = int(max_results)
        if max_results <= 0:
            logger.warning("  'max_results' inválido, usando default 3.")
            max_results = 3
        elif max_results > 50:  # Limitar um pouco mais alto que antes?
            logger.warning("  'max_results' muito alto, limitando a 50.")
            max_results = 50
        logger.info(f"  Max results definido como: {max_results}")
    except ValueError:
        logger.warning("  'max_results' não é um inteiro válido, usando default 3.")
        max_results = 3

    # 1. Gerar Embedding para a Query
    logger.info(f"  Gerando embedding para a query: '{query}'")
    query_embedding_list = None  # Inicializa para o except
    current_embedding_dim = EMBEDDING_DIM
    try:
        query_embedding_np = get_embedding(query)
        if query_embedding_np is None:
            raise ValueError("Falha ao gerar embedding para a query (retornou None).")

        # Opcional: Verificar dimensão contra o índice existente
        temp_index = init_index(DEFAULT_INDEX_BASE_PATH, embedding_dim=len(query_embedding_np))
        if temp_index is None:
            logger.warning(f"Não foi possível inicializar/carregar o índice FAISS em {DEFAULT_INDEX_BASE_PATH} para verificação de dimensão.")
            # Prosseguir com cautela
        elif temp_index.d != len(query_embedding_np):
            logger.error(f"Dimensão do embedding da query ({len(query_embedding_np)}) difere da dimensão do índice FAISS ({temp_index.d}).")
            return {
                "status": "error",
                "action": "recall_memory_failed",
                "data": {"message": f"Inconsistência na dimensão do embedding da query (esperado {temp_index.d}, obtido {len(query_embedding_np)})."},
            }
        del temp_index # Liberar referência

        query_embedding_list = query_embedding_np.tolist() # backend espera List[float]
        logger.debug(
            f"  Embedding da query gerado (Dim: {len(query_embedding_list)}).")
        

    except Exception as emb_err:
        logger.exception(f"  Erro ao gerar/preparar embedding para a query '{query}':")
        return {
            "status": "error",
            "action": "recall_memory_failed",
            "data": {"message": f"Erro ao preparar query embedding: {emb_err}"},
        }

    # 2. Buscar no Índice FAISS Externo
    try:
        logger.info(f"  Buscando no índice FAISS ({DEFAULT_INDEX_BASE_PATH}) por {max_results} resultados...")
        search_results = search_index(
            index_path_base=DEFAULT_INDEX_BASE_PATH,
            query_embedding=query_embedding_list,
            top_k=max_results
        )
        # search_index já loga se o índice não existe ou erros internos
        # Retorna lista de dicts: {'distance': float, 'metadata': dict}

        # 3. Formatar e retornar os resultados
        if search_results:
            # Os resultados já estão no formato desejado (distância + metadados)
            # Apenas precisamos garantir que 'content' está nos metadados (deve estar, se save.py estiver correto)
            formatted_results = []
            for res in search_results:
                if isinstance(res, dict) and 'metadata' in res and isinstance(res['metadata'], dict):
                    formatted_results.append({
                        "content": res['metadata'].get("content", "<Conteúdo Ausente>"),
                        "metadata": res['metadata'], # Inclui o metadata completo
                        "distance": round(res.get('distance', 999.0), 4)
                    })
                else:
                    logger.warning(f"Resultado inesperado do search_index: {res}")

            log_msg = f"[Skill: Recall Memory (FAISS)] Encontrados {len(formatted_results)} resultados relevantes."
            logger.info(log_msg)

            return {
                "status": "success",
                "action": "memory_recalled",
                "data": {"message": log_msg, "results": formatted_results},
            }
        else:
            log_msg = f"[Skill: Recall Memory (FAISS)] Nenhuma informação relevante encontrada na memória FAISS para a consulta: '{query}'"
            logger.info(log_msg)
            return {
                "status": "success",
                "action": "memory_recalled",
                "data": {"message": log_msg, "results": []},
            }

    except Exception as e:
        # Captura erros inesperados na chamada search_index ou formatação
        logger.exception(
            f"[Skill: Recall Memory (FAISS)] Erro inesperado ao buscar/processar memória FAISS: {e}"
        )
        return {
            "status": "error",
            "action": "recall_memory_failed",
            "data": {"message": f"Erro inesperado ao buscar/processar memória FAISS: {e}"},
        }
