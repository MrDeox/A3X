import sqlite3
import json
import struct # Para converter vetor float em bytes
import logging
from typing import Dict, Any

# Importar utils
from core.db_utils import get_db_connection
from core.embeddings import get_embedding, EMBEDDING_DIM # Importa dimensão também

logger = logging.getLogger(__name__) # Usar logger específico do módulo

def skill_save_memory(action_input: Dict[str, Any]) -> Dict[str, Any]:
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
    current_embedding_dim = EMBEDDING_DIM # Usar dimensão importada
    embedding_blob = None
    try:
        # get_embedding agora retorna numpy array, precisamos converter
        embedding_np = get_embedding(content_to_save) # Usa a função de core/embeddings.py
        if embedding_np is None:
             raise ValueError("Falha ao gerar embedding (retornou None).")

        if current_embedding_dim is None:
            # Tenta forçar o carregamento se ainda não ocorreu (embora get_embedding devesse ter feito)
            # Isso indica um problema maior se ocorrer, mas tentamos contornar.
            from core.embeddings import _load_model_internal
            _load_model_internal()
            from core.embeddings import EMBEDDING_DIM as current_embedding_dim # Tenta de novo
            if current_embedding_dim is None:
                  raise ValueError("EMBEDDING_DIM não está definida em core.embeddings mesmo após tentativa de carga.")

        # Converter o numpy array de floats para BLOB (bytes)
        format_string = f'<{current_embedding_dim}f' # Usa a dimensão obtida
        embedding_blob = struct.pack(format_string, *embedding_np)
        logger.debug(f"  Embedding gerado e convertido para BLOB ({len(embedding_blob)} bytes, Dim: {current_embedding_dim}).")

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

        if cursor.rowcount > 0: # INSERT aconteceu
            semantic_rowid = cursor.lastrowid
            logger.info(f"[Skill: Save Memory (ReAct)] Conteúdo salvo com ID {semantic_rowid}.")
        else: # IGNORE aconteceu
            cursor.execute("SELECT id FROM semantic_memory WHERE content = ?", (content_to_save,))
            existing_row = cursor.fetchone()
            if existing_row:
                semantic_rowid = existing_row[0]
                logger.info(f"[Skill: Save Memory (ReAct)] Conteúdo já existe com ID {semantic_rowid}. Verificando/Atualizando VSS.")
            else:
                logger.error("[Skill: Save Memory (ReAct)] Inconsistência: INSERT OR IGNORE retornou 0, mas SELECT não encontrou o conteúdo.")
                return {"status": "error", "action": "save_memory_failed", "data": {"message": "Erro de inconsistência no banco de dados ao verificar conteúdo duplicado."}}

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
                          vss_inserted = True # Consideramos sucesso mesmo que ignorado

                 except sqlite3.OperationalError as vss_err:
                     logger.error(f"  Erro operacional do SQLite ao inserir no índice VSS: {vss_err}")
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
            logger.debug("[Skill: Save Memory (ReAct)] Conexão DB fechada.")
