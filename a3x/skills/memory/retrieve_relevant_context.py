import logging
import os
from typing import Any, List, Dict 

# A3X Core Imports
from a3x.core.skills import skill
from a3x.core.embeddings import get_embedding
from a3x.core.semantic_memory_backend import search_index # <<< IMPORT FAISS BACKEND

logger = logging.getLogger(__name__)

# Defina o caminho base para o índice FAISS aqui ou obtenha de uma configuração
# Usando um caminho relativo ao diretório raiz do projeto (assumindo que a execução ocorre lá)
# Ajuste se necessário
DEFAULT_INDEX_BASE_PATH = "a3x/memory/indexes/semantic_memory"

@skill(
    name="introspect", 
    description="Responde perguntas sobre o que o A³X aprendeu ou sobre seu estado interno, buscando na memória semântica vetorial (FAISS) por contexto relevante.",
    parameters={
        "question": {"type": str, "description": "A pergunta a ser respondida."},
        "top_k": {"type": int, "description": "Número máximo de chunks relevantes a buscar.", "default": 3}
    } 
)
def introspect(ctx: Any, question: str, top_k: int = 3) -> str:
    """Responde perguntas sobre o agente buscando na memória semântica."""
    
    logger.info(f"Iniciando introspecção para pergunta: '{question[:100]}...' (top_k={top_k})")
    
    try:
        # 1. Gerar embedding para a pergunta (objetivo)
        logger.debug(f"Gerando embedding para a pergunta...")
        query_embedding = get_embedding(question) # Usa a 'question' como query
        if query_embedding is None:
            logger.error(f"Falha ao gerar embedding para a pergunta: '{question[:100]}...'")
            return "Erro ao gerar embedding para a busca na memória."
        logger.debug(f"Embedding gerado com sucesso (dim={len(query_embedding)})." )

        # 2. Buscar no índice FAISS usando o backend
        index_path = os.path.join(ctx.workspace_root, DEFAULT_INDEX_BASE_PATH) if hasattr(ctx, 'workspace_root') else DEFAULT_INDEX_BASE_PATH
        logger.info(f"Buscando no índice FAISS em: {index_path}")
        search_results: List[Dict[str, Any]] = search_index(
            index_path_base=index_path, 
            query_embedding=query_embedding, 
            top_k=top_k
        )

        # 3. Processar e formatar resultados
        if not search_results:
            logger.info("Nenhum resultado encontrado na busca vetorial FAISS para a introspecção.")
            return "Não encontrei informações relevantes em minha memória sobre isso."

        # Extrai o conteúdo de cada resultado
        content_chunks = []
        for result in search_results:
            metadata = result.get("metadata", {})
            content = metadata.get("content")
            if content:
                content_chunks.append(str(content))
            else:
                logger.warning(f"Resultado da busca sem 'content' na metadata: {metadata}")
        
        if not content_chunks:
             logger.warning("Busca vetorial retornou resultados, mas nenhum tinha 'content' na metadata.")
             return "Encontrei registros relacionados, mas sem conteúdo textual para exibir."

        logger.info(f"Recuperados {len(content_chunks)} chunks de contexto relevantes via FAISS para a introspecção.")
        # Retorna uma string formatada como uma resposta baseada na memória
        response_header = "Com base no que me lembro:\n---"
        return f"{response_header}\n" + "\n---\\n".join(content_chunks)

    except ImportError as e:
        logger.error(f"Erro ao importar dependência necessária (possivelmente FAISS ou Embeddings): {e}", exc_info=True)
        return "Erro interno: Dependência de memória não encontrada."
    except Exception as e:
        logger.error(f"Erro inesperado durante a busca vetorial FAISS para introspecção: {e}", exc_info=True)
        return f"Erro inesperado ao acessar minha memória: {e}"

# Código antigo baseado em SQLite e extensão removido. 