import sqlite3
import struct  # Para converter vetor float em bytes
import logging
from typing import Dict, Any

# Importar utils
from core.db_utils import get_db_connection
from core.embeddings import get_embedding, EMBEDDING_DIM  # Importa dimensão também

logger = logging.getLogger(__name__)  # Usar logger específico do módulo


def skill_recall_memory(action_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Busca na memória semântica (tabela semantic_memory via índice VSS)
    por conteúdos similares à query fornecida.

    Args:
        action_input (dict): Dicionário contendo a query e opções.
            Exemplo:
                {"query": "Qual o lembrete?", "max_results": 5}

    Returns:
        dict: Dicionário padronizado com status e resultados.
    """
    logger.info("\n[Skill: Recall Memory (ReAct)]")
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
        elif max_results > 20:  # Limitar para não sobrecarregar
            logger.warning("  'max_results' muito alto, limitando a 20.")
            max_results = 20
        logger.info(f"  Max results definido como: {max_results}")
    except ValueError:
        logger.warning("  'max_results' não é um inteiro válido, usando default 3.")
        max_results = 3

    # 1. Gerar Embedding para a Query
    logger.info(f"  Gerando embedding para a query: '{query}'")
    query_embedding_blob = None  # Inicializa para o except
    current_embedding_dim = EMBEDDING_DIM  # Usar dimensão importada
    try:
        query_embedding_np = get_embedding(query)
        if query_embedding_np is None:
            raise ValueError("Falha ao gerar embedding para a query (retornou None).")

        if current_embedding_dim is None:
            # Tenta forçar o carregamento
            from core.embeddings import _load_model_internal

            _load_model_internal()
            from core.embeddings import (
                EMBEDDING_DIM as current_embedding_dim,
            )  # Tenta de novo

            if current_embedding_dim is None:
                raise ValueError(
                    "EMBEDDING_DIM não está definida em core.embeddings para recall."
                )

        # Converter para BLOB
        format_string = f"<{current_embedding_dim}f"  # Usa a dimensão obtida
        query_embedding_blob = struct.pack(format_string, *query_embedding_np)
        logger.debug(
            f"  Embedding da query gerado e convertido para BLOB ({len(query_embedding_blob)} bytes, Dim: {current_embedding_dim})."
        )

    except (ImportError, ValueError, struct.error) as emb_err:
        logger.exception(f"  Erro ao gerar/empacotar embedding para a query '{query}':")
        return {
            "status": "error",
            "action": "recall_memory_failed",
            "data": {"message": f"Erro ao preparar query embedding: {emb_err}"},
        }
    except Exception as e:
        logger.exception(
            f"  Erro inesperado ao gerar embedding para a query '{query}':"
        )
        return {
            "status": "error",
            "action": "recall_memory_failed",
            "data": {"message": f"Erro inesperado ao gerar query embedding: {e}"},
        }

    # 2. Buscar no Banco de Dados usando VSS
    conn = None  # Inicializa conn como None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Verificar se a tabela VSS existe
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vss_semantic_memory'"
        )
        vss_table_exists = cursor.fetchone()

        if not vss_table_exists:
            logger.warning(
                "  Índice VSS 'vss_semantic_memory' não encontrado. A busca semântica não pode ser realizada."
            )
            return {  # Retorna erro, mas finally será executado
                "status": "error",
                "action": "recall_memory_failed",
                "data": {"message": "Índice de busca semântica não está disponível."},
            }

        # Verificar se a tabela principal está vazia
        cursor.execute("SELECT COUNT(*) FROM semantic_memory")
        memory_count = cursor.fetchone()[0]
        if memory_count == 0:
            logger.info("  Tabela 'semantic_memory' está vazia. Pulando busca VSS.")
            return {  # Retorna sucesso, mas finally será executado
                "status": "success",
                "action": "memory_recalled",
                "data": {
                    "message": "Memória semântica está vazia. Nada a ser recuperado.",
                    "results": [],
                },
            }

        logger.info(f"  Buscando no índice VSS por {max_results} resultados...")
        cursor.execute(
            """
            SELECT
                sm.rowid,      -- Índice 0
                sm.content,    -- Índice 1
                vss.distance   -- Índice 2
            FROM vss_semantic_memory vss
            JOIN semantic_memory sm ON vss.rowid = sm.id -- Usa sm.id para JOIN
            WHERE vss_search(
                vss.embedding,
                vss_search_params(?, ?) -- Passa embedding e max_results aqui
            )
            ORDER BY vss.distance ASC;
            """,
            (query_embedding_blob, max_results),  # Parâmetros para vss_search_params
        )

        results = cursor.fetchall()  # Fetchall retorna uma lista de tuplas

        # 3. Formatar e retornar os resultados
        if results:
            formatted_results = []
            try:
                for row in results:
                    # ACESSAR POR ÍNDICE NUMÉRICO
                    result_id = row[0]
                    result_content = row[1]
                    result_distance = row[2]
                    formatted_results.append(
                        {
                            "id": result_id,
                            "content": result_content,
                            "distance": round(result_distance, 4),
                        }
                    )
                log_msg = f"[Skill: Recall Memory (ReAct)] Encontrados e formatados {len(results)} resultados relevantes."
                logger.info(log_msg)

            except IndexError as e:
                logger.error(
                    f"[Skill: Recall Memory ERROR] IndexError ao acessar coluna no resultado: {e}. Linha: {row}",
                    exc_info=True,
                )
                return {
                    "status": "success",
                    "action": "memory_recalled",
                    "data": {
                        "message": f"Erro ao formatar resultados da memória: {e}",
                        "results": [],
                    },
                }
            except Exception as e:
                logger.error(
                    f"[Skill: Recall Memory ERROR] Erro inesperado ao formatar resultados: {e}. Linha: {row}",
                    exc_info=True,
                )
                return {
                    "status": "success",
                    "action": "memory_recalled",
                    "data": {
                        "message": f"Erro inesperado ao formatar resultados: {e}",
                        "results": [],
                    },
                }

            # Se o loop for concluído com sucesso:
            return {
                "status": "success",
                "action": "memory_recalled",
                "data": {"message": log_msg, "results": formatted_results},
            }
        else:
            log_msg = f"[Skill: Recall Memory (ReAct)] Nenhuma informação relevante encontrada na memória para a consulta: '{query}'"
            logger.info(log_msg)
            return {
                "status": "success",
                "action": "memory_recalled",
                "data": {"message": log_msg, "results": []},
            }

    except sqlite3.OperationalError as e:
        logger.error(
            f"[Skill: Recall Memory (ReAct)] Erro operacional de banco de dados ao buscar memória: {e}",
            exc_info=True,
        )
        if "no such table: vss_semantic_memory" in str(e):
            logger.error(
                "[Skill: Recall Memory (ReAct)] A tabela VSS 'vss_semantic_memory' parece não existir ou não foi criada corretamente."
            )
            return {
                "status": "error",
                "action": "recall_memory_failed",
                "data": {
                    "message": "Erro de banco de dados: Tabela de busca vetorial não encontrada."
                },
            }
        return {
            "status": "error",
            "action": "recall_memory_failed",
            "data": {"message": f"Erro operacional de banco de dados: {e}"},
        }
    except Exception as e:
        logger.error(
            f"[Skill: Recall Memory (ReAct)] Erro inesperado ao buscar memória: {e}"
        )
        logger.exception(e)
        return {
            "status": "error",
            "action": "recall_memory_failed",
            "data": {"message": f"Erro inesperado ao buscar memória: {e}"},
        }
    finally:
        if conn:
            conn.close()
            logger.debug(
                "[Skill: Recall Memory (ReAct)] Conexão com o banco de dados fechada no finally."
            )
