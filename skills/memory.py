import sqlite3
import os
from datetime import datetime
import json
import struct # Para converter vetor float em bytes
import numpy as np # A função get_embedding retorna numpy array
import logging

# Define o caminho do banco de dados na raiz do projeto
DATABASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'memory.db'))

# Importar utils (ajuste o caminho se memory.py estiver em outro lugar)
from core.db_utils import get_db_connection
from core.embeddings import get_embedding, EMBEDDING_DIM # Importa dimensão também

logger = logging.getLogger(__name__) # Usar logger da skill

def _initialize_memory_db():
    """Cria a tabela 'knowledge' se ela não existir."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Trigger para atualizar updated_at (Opcional, mas bom)
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS update_knowledge_updated_at
            AFTER UPDATE ON knowledge FOR EACH ROW
            BEGIN
                UPDATE knowledge SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
            END;
        ''')
        conn.commit()
        conn.close()
        print("[Memory] Banco de dados inicializado.")
    except sqlite3.Error as e:
        print(f"[Memory Error] Erro ao inicializar banco de dados: {e}")

# Chama a inicialização quando o módulo é carregado
_initialize_memory_db()

def skill_remember_info(entities: dict, original_command: str, intent: str = None, history: list = None) -> dict:
    """Armazena informações na memória do assistente."""
    print("\n[Skill: Remember Info]")
    print(f"  Entidades recebidas: {entities}")
    if history:
        print(f"  Histórico recebido (últimos turnos): {history[-4:]}") # Mostra parte do histórico
    else:
        print("  Nenhum histórico fornecido")

    info = entities.get("info")
    if not info:
        return {"status": "error", "action": "remember_info_failed", "data": {"message": "Não entendi qual informação armazenar."}}

    # AVISO: Esta skill é um placeholder para compatibilidade. Use 'save_memory' no fluxo ReAct.
    logger.warning(f"[Skill: Remember Info WARN] Chamada para skill legacy 'remember_info'. Use 'save_memory' para o fluxo ReAct. Info: {info}")
    # REMOVIDO: store_info(info)
    return {"status": "error", "action": "remember_info_legacy", "data": {"message": f"AVISO: A skill 'remember_info' é legacy. Use 'save_memory' no fluxo ReAct para '{info}'."}}

def skill_recall_info(entities: dict, original_command: str, intent: str = None, history: list = None) -> dict:
    """Recupera informações da memória do assistente."""
    print("\n[Skill: Recall Info]")
    print(f"  Entidades recebidas: {entities}")
    if history:
        print(f"  Histórico recebido (últimos turnos): {history[-4:]}") # Mostra parte do histórico
    else:
        print("  Nenhum histórico fornecido")

    info = entities.get("info")
    if not info:
        return {"status": "error", "action": "recall_info_failed", "data": {"message": "Não entendi qual informação recuperar."}}

    # AVISO: Esta skill é um placeholder para compatibilidade. Use 'recall_memory' no fluxo ReAct.
    logger.warning(f"[Skill: Recall Info WARN] Chamada para skill legacy 'recall_info'. Use 'recall_memory' para o fluxo ReAct. Info: {info}")
    # REMOVIDO: retrieved_info = recall_info(info)
    # Retorna erro indicando para usar a skill correta
    return {"status": "error", "action": "recall_info_legacy", "data": {"message": f"AVISO: A skill 'recall_info' é legacy. Use 'recall_memory' no fluxo ReAct para buscar sobre '{info}'."}}

# Outras funções de memória podem ser adicionadas aqui (ex: deletar, listar chaves)

# <<< NOVA FUNÇÃO skill_save_memory >>>
def skill_save_memory(action_input: dict) -> dict:
    """
    Salva um conteúdo textual e seu embedding na tabela semantic_memory.
    Também insere no índice VSS se disponível.

    Args:
        action_input (dict): Dicionário contendo os dados a salvar.
            Exemplo:
                {"content": "Lembrete importante.", "metadata": {"source": "user"}}

    Returns:
        dict: Dicionário padronizado com status.
    """
    logger.info("\n[Skill: Save Memory (ReAct)]")
    logger.debug(f"  Action Input: {action_input}")

    content_to_save = action_input.get("content")
    metadata_dict = action_input.get("metadata") # Pode ser None

    if not content_to_save:
        logger.error("  Falha ao salvar memória: Parâmetro 'content' obrigatório ausente.")
        return {"status": "error", "action": "save_memory_failed", "data": {"message": "Parâmetro 'content' obrigatório ausente."}}

    # 1. Gerar Embedding
    logger.info(f"  Gerando embedding para o conteúdo: '{content_to_save[:50]}...'") # Log trecho
    embedding_np = None # Inicializa para usar no except do pack_err
    current_embedding_dim = None # Inicializa para usar no except do pack_err
    try:
        # get_embedding agora retorna numpy array, precisamos converter
        embedding_np = get_embedding(content_to_save) # Usa a função de core/embeddings.py
        if embedding_np is None:
             raise ValueError("Falha ao gerar embedding (retornou None).")

        # Converter o numpy array de floats para BLOB (bytes)
        # <<< OBTER DIMENSÃO AQUI >>>
        from core.embeddings import EMBEDDING_DIM as current_embedding_dim # Reimporta localmente
        if current_embedding_dim is None:
             # Tenta forçar o carregamento se ainda não ocorreu (embora get_embedding devesse ter feito)
             from core.embeddings import _load_model_internal
             _load_model_internal() # Assume que _load_model_internal() está acessível ou define EMBEDDING_DIM globalmente
             from core.embeddings import EMBEDDING_DIM as current_embedding_dim # Tenta de novo
             if current_embedding_dim is None:
                  raise ValueError("EMBEDDING_DIM não está definida em core.embeddings mesmo após tentativa de carga.")

        format_string = f'<{current_embedding_dim}f' # Usa a dimensão obtida
        embedding_blob = struct.pack(format_string, *embedding_np)
        logger.debug(f"  Embedding gerado e convertido para BLOB ({len(embedding_blob)} bytes, Dim: {current_embedding_dim}).")
        # <<< FIM DA CORREÇÃO struct.error >>>

    except (ImportError, ValueError, struct.error) as pack_err:
        logger.exception(f"  Erro ao empacotar embedding (Dim:{current_embedding_dim}):")
        return {"status": "error", "action": "save_memory_failed", "data": {"message": f"Erro ao preparar embedding para salvar: {pack_err}"}}
    except Exception as e:
        # Captura genérica para erros no get_embedding ou carga do modelo
        logger.exception(f"  Erro inesperado durante geração/preparação do embedding para '{content_to_save[:50]}...':")
        return {"status": "error", "action": "save_memory_failed", "data": {"message": f"Erro inesperado ao gerar embedding: {e}"}}

    # 2. Converter Metadados para JSON string (se houver)
    metadata_json = None
    if metadata_dict:
        try:
            metadata_json = json.dumps(metadata_dict)
            logger.debug(f"  Metadados convertidos para JSON: {metadata_json}")
        except TypeError as e:
            logger.warning(f"  Metadados fornecidos não são serializáveis em JSON: {e}. Ignorando metadados.")
            metadata_json = None # Garante None se falhar

    # 3. Salvar no Banco de Dados
    conn = None
    semantic_rowid = None
    vss_inserted = False
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Inserir na tabela principal (semantic_memory)
        # Usar INSERT OR IGNORE para não dar erro se o 'content' já existir (devido ao UNIQUE)
        logger.info(f"  Inserindo/Ignorando conteúdo na tabela 'semantic_memory'...")
        cursor.execute("""
            INSERT OR IGNORE INTO semantic_memory (content, embedding, metadata)
            VALUES (?, ?, ?)
        """, (content_to_save, embedding_blob, metadata_json))

        # <<< CORREÇÃO: Inverter a lógica do if/else >>>
        if cursor.rowcount > 0: # INSERT aconteceu (rowcount > 0)
            semantic_rowid = cursor.lastrowid
            logger.info(f"[Skill: Save Memory (ReAct)] Conteúdo salvo com ID {semantic_rowid}.")
        else: # IGNORE aconteceu (rowcount == 0)
            # Se IGNORE foi ativado, o rowcount é 0, mas precisamos do ID existente
            cursor.execute("SELECT id FROM semantic_memory WHERE content = ?", (content_to_save,))
            existing_row = cursor.fetchone()
            if existing_row:
                # Acessar o ID pelo índice 0 da tupla retornada por fetchone()
                semantic_rowid = existing_row[0]
                logger.info(f"[Skill: Save Memory (ReAct)] Conteúdo já existe com ID {semantic_rowid}. Verificando/Atualizando VSS.")
            else:
                # Isso não deveria acontecer se o IGNORE foi por causa deste conteúdo
                logger.error("[Skill: Save Memory (ReAct)] Inconsistência: INSERT OR IGNORE retornou 0, mas SELECT não encontrou o conteúdo.")
                # Não fechar conn aqui, deixar o finally cuidar disso
                return {"status": "error", "action": "save_memory_failed", "data": {"message": "Erro de inconsistência no banco de dados ao verificar conteúdo duplicado."}}
        # <<< FIM DA CORREÇÃO >>>

        # Inserir no índice VSS (APENAS se VSS estiver ativo e tivermos rowid)
        if semantic_rowid is not None:
             cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vss_semantic_memory'")
             vss_table_exists = cursor.fetchone()

             if vss_table_exists:
                 try:
                     logger.debug(f"  Tentando inserir/ignorar no índice VSS (rowid: {semantic_rowid})...")
                     cursor.execute('''
                         INSERT OR IGNORE INTO vss_semantic_memory (rowid, embedding)
                         VALUES (?, ?)
                     ''', (semantic_rowid, embedding_blob))
                     if cursor.rowcount > 0:
                          logger.info(f"  Embedding inserido no índice VSS com rowid {semantic_rowid}.")
                          vss_inserted = True
                     else:
                          logger.info(f"  Embedding com rowid {semantic_rowid} já existia no índice VSS.")
                          # Considerar se deveríamos atualizar o embedding no VSS aqui?
                          # Por enquanto, apenas ignoramos se já existe.
                          vss_inserted = True # Consideramos sucesso mesmo que ignorado

                 except sqlite3.OperationalError as vss_err:
                     logger.error(f"  Erro operacional do SQLite ao inserir no índice VSS: {vss_err}")
                     # Não retornamos erro fatal aqui, a memória principal foi salva.
                 except Exception as vss_generic_err:
                     logger.exception(f"  Erro inesperado ao inserir no índice VSS:")
             else:
                 logger.debug("  Tabela VSS 'vss_semantic_memory' não encontrada. Pulando inserção no VSS.")

        conn.commit()
        logger.info(f"  Memória salva com sucesso (ID: {semantic_rowid}, VSS: {vss_inserted}).")
        return {"status": "success", "action": "memory_saved", "data": {"message": f"Informação salva na memória com ID {semantic_rowid}.", "rowid": semantic_rowid, "vss_updated": vss_inserted}}

    except sqlite3.Error as db_err:
        logger.error(f"  Erro de banco de dados ao salvar memória: {db_err}")
        if conn: conn.rollback()
        return {"status": "error", "action": "save_memory_failed", "data": {"message": f"Erro de DB: {db_err}"}}
    except Exception as generic_err:
        logger.exception("  Erro inesperado ao salvar memória:")
        if conn: conn.rollback()
        return {"status": "error", "action": "save_memory_failed", "data": {"message": f"Erro inesperado: {generic_err}"}}
    finally:
        if conn:
            conn.close()
# <<< FIM skill_save_memory >>>

# <<< NOVA FUNÇÃO skill_recall_memory >>>
def skill_recall_memory(action_input: dict) -> dict:
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

    query = action_input.get("query")
    max_results = action_input.get("max_results", 3) # Default 3 se não fornecido

    if not query:
        logger.error("  Falha ao recuperar memória: Parâmetro 'query' obrigatório ausente.")
        return {"status": "error", "action": "recall_memory_failed", "data": {"message": "Parâmetro 'query' obrigatório ausente."}}

    # Validação básica de max_results
    try:
        max_results = int(max_results)
        if max_results <= 0:
            logger.warning("  'max_results' inválido, usando default 3.")
            max_results = 3
        elif max_results > 20: # Limitar para não sobrecarregar
             logger.warning("  'max_results' muito alto, limitando a 20.")
             max_results = 20
        logger.info(f"  Max results definido como: {max_results}")
    except ValueError:
        logger.warning("  'max_results' não é um inteiro válido, usando default 3.")
        max_results = 3

    # 1. Gerar Embedding para a Query
    logger.info(f"  Gerando embedding para a query: '{query}'")
    query_embedding_blob = None # Inicializa para o except
    try:
        query_embedding_np = get_embedding(query)
        if query_embedding_np is None:
            raise ValueError("Falha ao gerar embedding para a query (retornou None).")

        # Converter para BLOB
        # <<< OBTER DIMENSÃO AQUI TAMBÉM >>>
        from core.embeddings import EMBEDDING_DIM as current_embedding_dim # Reimporta localmente
        if current_embedding_dim is None:
             # Tenta forçar o carregamento
             from core.embeddings import _load_model_internal
             _load_model_internal()
             from core.embeddings import EMBEDDING_DIM as current_embedding_dim # Tenta de novo
             if current_embedding_dim is None:
                 raise ValueError("EMBEDDING_DIM não está definida em core.embeddings para recall.")

        format_string = f'<{current_embedding_dim}f' # Usa a dimensão obtida
        query_embedding_blob = struct.pack(format_string, *query_embedding_np)
        logger.debug(f"  Embedding da query gerado e convertido para BLOB ({len(query_embedding_blob)} bytes, Dim: {current_embedding_dim}).")

    except (ImportError, ValueError, struct.error) as emb_err:
        logger.exception(f"  Erro ao gerar/empacotar embedding para a query '{query}':")
        return {"status": "error", "action": "recall_memory_failed", "data": {"message": f"Erro ao preparar query embedding: {emb_err}"}}
    except Exception as e:
        logger.exception(f"  Erro inesperado ao gerar embedding para a query '{query}':")
        return {"status": "error", "action": "recall_memory_failed", "data": {"message": f"Erro inesperado ao gerar query embedding: {e}"}}


    # 2. Buscar no Banco de Dados usando VSS
    conn = None # Inicializa conn como None
    try:
        # Mover get_db_connection para ANTES das checagens que podem retornar cedo
        conn = get_db_connection()
        cursor = conn.cursor()

        # Verificar se a tabela VSS existe antes de tentar buscar
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vss_semantic_memory'")
        vss_table_exists = cursor.fetchone()

        if not vss_table_exists:
             logger.warning("  Índice VSS 'vss_semantic_memory' não encontrado. A busca semântica não pode ser realizada.")
             # Não precisa mais fechar conn aqui, o finally cuidará disso.
             return { # Retorna erro, mas finally será executado
                  "status": "error", 
                  "action": "recall_memory_failed", 
                  "data": {"message": "Índice de busca semântica não está disponível."}
             }

        # <<< ADICIONAR VERIFICAÇÃO DE TABELA VAZIA AQUI >>>
        cursor.execute("SELECT COUNT(*) FROM semantic_memory")
        memory_count = cursor.fetchone()[0]
        if memory_count == 0:
             logger.info("  Tabela 'semantic_memory' está vazia. Pulando busca VSS.")
             # Não precisa mais fechar conn aqui.
             return { # Retorna sucesso, mas finally será executado
                "status": "success",
                "action": "memory_recalled",
                "data": {
                    "message": "Memória semântica está vazia. Nada a ser recuperado.",
                    "results": []
                }
            }
        # <<< FIM DA VERIFICAÇÃO >>>

        logger.info(f"  Buscando no índice VSS por {max_results} resultados...")
        cursor.execute(
            f"""
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
            """, # Certifique-se que não há '#' antes deste """
            (query_embedding_blob, max_results) # Parâmetros para vss_search_params
        )

        results = cursor.fetchall() # Fetchall retorna uma lista de tuplas (ou Rows)

        # 3. Formatar e retornar os resultados
        if results:
            formatted_results = []
            try:
                for row in results:
                    # <<< LOG REMOVIDO/COMENTADO pois vamos acessar por índice >>>
                    # logger.debug(f"[Skill: Recall Memory DEBUG] Tipo: {type(row)}, Conteúdo: {row}") 
                    # ACESSAR POR ÍNDICE NUMÉRICO (mais seguro)
                    result_id = row[0]
                    result_content = row[1]
                    result_distance = row[2]
                    formatted_results.append({
                        "id": result_id,
                        "content": result_content,
                        # "timestamp": None, # Timestamp removido da query e do resultado
                        "distance": round(result_distance, 4)
                    })
                log_msg = f"[Skill: Recall Memory (ReAct)] Encontrados e formatados {len(results)} resultados relevantes."
                logger.info(log_msg)
                # Logar os resultados formatados (opcional)
                # for res in formatted_results:
                #     logger.debug(f"  - ID: {res['id']}, Dist: {res['distance']}, Content: {res['content'][:100]}...")

            except IndexError as e:
                 # Logar o erro específico se o acesso por índice falhar
                 logger.error(f"[Skill: Recall Memory ERROR] IndexError ao acessar coluna no resultado: {e}. Linha: {row}", exc_info=True)
                 # Retorna sucesso, mas com mensagem de erro nos dados e lista vazia
                 return {
                    "status": "success", # Ou "error" se preferir tratar falha de formatação como erro
                    "action": "memory_recalled",
                    "data": {
                        "message": f"Erro ao formatar resultados da memória: {e}",
                        "results": []
                    }
                 }
            except Exception as e:
                 logger.error(f"[Skill: Recall Memory ERROR] Erro inesperado ao formatar resultados: {e}. Linha: {row}", exc_info=True)
                 return {
                    "status": "success", # Ou "error"
                    "action": "memory_recalled",
                    "data": {
                        "message": f"Erro inesperado ao formatar resultados: {e}",
                        "results": []
                    }
                 }

            # Se o loop for concluído com sucesso:
            return {
                "status": "success",
                "action": "memory_recalled",
                "data": {
                    "message": log_msg,
                    "results": formatted_results
                }
            }
        else:
            # ... (bloco else não muda)
            log_msg = f"[Skill: Recall Memory (ReAct)] Nenhuma informação relevante encontrada na memória para a consulta: '{query}'"
            logger.info(log_msg)
            return {
                "status": "success", # Ainda sucesso, mas sem resultados
                "action": "memory_recalled",
                "data": {
                    "message": log_msg,
                    "results": []
                }
            }

    except sqlite3.OperationalError as e:
        # Erro comum se a tabela VSS não existir ou a query estiver mal formada
        logger.error(f"[Skill: Recall Memory (ReAct)] Erro operacional de banco de dados ao buscar memória: {e}", exc_info=True)
        # Verifica se o erro é sobre a tabela VSS não existir
        if "no such table: vss_semantic_memory" in str(e):
             logger.error("[Skill: Recall Memory (ReAct)] A tabela VSS 'vss_semantic_memory' parece não existir ou não foi criada corretamente.")
             return {"status": "error", "error": "Erro de banco de dados: Tabela de busca vetorial não encontrada."}
        return {"status": "error", "error": f"Erro operacional de banco de dados: {e}"}
    except Exception as e:
        logger.error(f"[Skill: Recall Memory (ReAct)] Erro inesperado ao buscar memória: {e}")
        # Log completo do traceback para depuração
        logger.exception(e)
        # Retornar um erro genérico
        return {"status": "error", "action": "recall_memory_failed", "data": {"message": f"Erro inesperado ao buscar memória: {e}"}}
    finally:
        # Garantir que a conexão seja fechada se foi aberta
        if conn:
            conn.close()
            logger.debug("[Skill: Recall Memory (ReAct)] Conexão com o banco de dados fechada no finally.")
# <<< FIM skill_recall_memory >>> 