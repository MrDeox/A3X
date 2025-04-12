import logging
import os
from typing import Any, List, Dict, Optional

# A3X Core Imports
from a3x.core.skills import skill
from a3x.core.embeddings import get_embedding
from a3x.core.semantic_memory_backend import add_to_index # <<< IMPORT FAISS BACKEND ADD FUNCTION

logger = logging.getLogger(__name__)

# Defina o caminho base para o índice FAISS aqui ou obtenha de uma configuração
# Usando um caminho relativo ao diretório raiz do projeto (assumindo que a execução ocorre lá)
DEFAULT_INDEX_BASE_PATH = "a3x/memory/indexes/semantic_memory"
DEFAULT_EMBEDDING_DIM = 768 # Assumed default, should ideally come from config

@skill(
    name="index_memory_chunk", 
    description="Adiciona um chunk de texto à memória semântica FAISS para futura recuperação vetorial.",
    parameters={
        "content": (str, ...),              # str, Obrigatório
        "source": (str, "unknown"),       # str, Opcional com default "unknown"
        "tags": (List[str], [])           # List[str], Opcional com default []
    } 
)
def index_memory_chunk(ctx: Any, content: str, source: str = "unknown", tags: Optional[List[str]] = None) -> str:
    """Adiciona um chunk de texto e metadados associados ao índice vetorial FAISS."""
    
    if tags is None:
        tags = [] # Ensure tags is a list if not provided
        
    logger.info(f"Iniciando indexação de chunk de memória. Source: {source}, Tags: {tags}, Content: '{content[:100]}...'")
    
    try:
        # 1. Gerar embedding para o conteúdo
        logger.debug(f"Gerando embedding para o conteúdo...")
        embedding = get_embedding(content)
        if embedding is None:
            logger.error(f"Falha ao gerar embedding para o conteúdo: '{content[:100]}...'")
            return "Erro ao gerar embedding para indexação."
        logger.debug(f"Embedding gerado com sucesso (dim={len(embedding)})." )

        # 2. Preparar metadados
        metadata = {
            "content": content,
            "source": source,
            "tags": tags
            # O backend `add_to_index` adicionará o _vector_id automaticamente
        }
        logger.debug(f"Metadados preparados: {metadata}")

        # 3. Adicionar ao índice FAISS usando o backend
        # Use o workspace_root do contexto se disponível, senão o default.
        # NOTA: Isso assume que o código está sendo executado da raiz do projeto A3X
        # Uma solução mais robusta seria passar o workspace root explicitamente ou usar uma config.
        workspace_root_path = getattr(ctx, 'workspace_root', None) 
        if workspace_root_path and os.path.isdir(workspace_root_path):
             index_path = os.path.join(workspace_root_path, DEFAULT_INDEX_BASE_PATH)
        else:
             # Fallback se ctx não tem workspace_root ou não é válido
             # Isso pode não ser ideal se executado de diretórios inesperados
             logger.warning(f"Workspace root não encontrado ou inválido no contexto. Usando caminho relativo padrão: {DEFAULT_INDEX_BASE_PATH}")
             index_path = DEFAULT_INDEX_BASE_PATH

        logger.info(f"Adicionando ao índice FAISS em: {index_path}")
        
        # Chamar a função do backend para adicionar e salvar
        add_to_index(
            index_path_base=index_path, 
            embedding=embedding, 
            metadata=metadata,
            embedding_dim=DEFAULT_EMBEDDING_DIM # Passa a dimensão caso o índice precise ser criado
        )
        
        # Se add_to_index não levantou exceção, consideramos sucesso.
        # (A função add_to_index já loga sucesso/falha interna)
        return "Chunk adicionado com sucesso à memória vetorial."

    except ImportError as e:
        logger.error(f"Erro ao importar dependência necessária (possivelmente FAISS ou Embeddings): {e}", exc_info=True)
        return "Erro interno: Dependência de indexação não encontrada."
    except Exception as e:
        logger.error(f"Erro inesperado durante a indexação do chunk: {e}", exc_info=True)
        return f"Erro inesperado durante indexação: {e}" 