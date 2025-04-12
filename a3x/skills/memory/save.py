import sqlite3
import json
import struct  # Para converter vetor float em bytes
import logging
from typing import Dict, Any

# Importar utils - Use absolute paths
from a3x.core.db_utils import get_db_connection
from a3x.core.embeddings import get_embedding, EMBEDDING_DIM  # Importa dimensão também
from a3x.core.semantic_memory_backend import add_to_index, init_index
import os
from a3x.core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)  # Usar logger específico do módulo

# Definir caminho base do índice FAISS
# É melhor centralizar isso, mas por enquanto definimos aqui
DEFAULT_INDEX_BASE_PATH = os.path.join(PROJECT_ROOT, "a3x", "memory", "indexes", "semantic_memory")


def skill_save_memory(action_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Salva um conteúdo textual e seu embedding no índice FAISS externo.
    Metadata inclui o conteúdo original e qualquer metadata adicional fornecida.

    Args:
        action_input (dict): Dicionário contendo os dados a salvar.
            Exemplo:
                {"content": "Lembrete importante.", "metadata": {"source": "user"}}

    Returns:
        dict: Dicionário padronizado com status.
    """
    logger.info("\n[Skill: Save Memory (FAISS)]")
    logger.debug(f"  Action Input: {action_input}")

    content_to_save = action_input.get("content")
    provided_metadata = action_input.get("metadata", {})  # Garante dict

    if not content_to_save:
        logger.error(
            "  Falha ao salvar memória: Parâmetro 'content' obrigatório ausente."
        )
        return {
            "status": "error",
            "action": "save_memory_failed",
            "data": {"message": "Parâmetro 'content' obrigatório ausente."},
        }

    # 1. Gerar Embedding
    logger.info(
        f"  Gerando embedding para o conteúdo: '{content_to_save[:50]}...'"
    )  # Log trecho
    embedding_np = None  # Inicializa para usar no except
    current_embedding_dim = EMBEDDING_DIM
    try:
        embedding_np = get_embedding(content_to_save)
        if embedding_np is None:
            raise ValueError("Falha ao gerar embedding (retornou None).")

        # Opcional: Verificar dimensão contra o índice existente
        # Isso pode adicionar latência, mas garante consistência
        temp_index = init_index(DEFAULT_INDEX_BASE_PATH, embedding_dim=len(embedding_np))
        if temp_index is None:
            logger.warning(f"Não foi possível inicializar/carregar o índice FAISS em {DEFAULT_INDEX_BASE_PATH} para verificação.")
            # Prosseguir mesmo assim? Ou retornar erro? Por enquanto, prosseguir.
        elif temp_index.d != len(embedding_np):
            logger.error(f"Dimensão do embedding gerado ({len(embedding_np)}) difere da dimensão do índice FAISS existente ({temp_index.d}).")
            return {
                "status": "error",
                "action": "save_memory_failed",
                "data": {"message": f"Inconsistência na dimensão do embedding (esperado {temp_index.d}, obtido {len(embedding_np)})."},
            }
        del temp_index  # Liberar referência

        embedding_list = embedding_np.tolist()  # backend espera List[float]
        logger.debug(
            f"  Embedding gerado (Dim: {len(embedding_list)})."
        )

    except Exception as e:
        # Captura genérica para erros no get_embedding ou carga do modelo
        logger.exception(
            f"  Erro inesperado durante geração/preparação do embedding para '{content_to_save[:50]}...':"
        )
        return {
            "status": "error",
            "action": "save_memory_failed",
            "data": {"message": f"Erro inesperado ao gerar embedding: {e}"},
        }

    # 2. Preparar Metadados
    final_metadata = {
        "content": content_to_save,
        "source": "memory_save_skill",
        **(provided_metadata if isinstance(provided_metadata, dict) else {})
    }
    logger.debug(f"  Metadados finais a serem salvos: {final_metadata}")

    # 3. Salvar no Índice FAISS Externo
    try:
        add_to_index(
            index_path_base=DEFAULT_INDEX_BASE_PATH,
            embedding=embedding_list,
            metadata=final_metadata,
            embedding_dim=len(embedding_list)
        )
        # add_to_index já loga sucesso/erro internamente

        logger.info(
            f"  Memória salva com sucesso no índice FAISS em '{DEFAULT_INDEX_BASE_PATH}'."
        )
        return {
            "status": "success",
            "action": "memory_saved",
            "data": {
                "message": f"Informação salva na memória vetorial (FAISS).",
            },
        }

    except Exception as faiss_err:
        # add_to_index pode levantar exceções se algo muito errado ocorrer
        # (embora tente tratar erros internamente e logar)
        logger.exception(f"  Erro inesperado ao chamar add_to_index: {faiss_err}")
        return {
            "status": "error",
            "action": "save_memory_failed",
            "data": {"message": f"Erro inesperado ao salvar no índice FAISS: {faiss_err}"},
        }
