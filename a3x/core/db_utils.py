import sqlite3 # Use standard library
# import pysqlite3 as sqlite3 # Use pysqlite3 backport
# import sqlite3 # Use standard library
# import pysqlite3 as sqlite3 # Removed custom pysqlite3
import os
import logging  # Usar logging é melhor que print para debug
import json
import asyncio
from typing import Optional, Dict

# Configurar logging básico
logging.basicConfig(level=logging.INFO, format="[%(levelname)s DB] %(message)s")
logger = logging.getLogger(__name__)

# Importa o caminho do DB do config
from a3x.core.config import DATABASE_PATH

logger.info(f"Using database path from config: {DATABASE_PATH}")

# <<< ADICIONAR CAMINHO DA EXTENSÃO >>>
# AJUSTE ESTE CAMINHO se você colocou as extensões em outro lugar!
VECTOR_EXTENSION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "lib", "vector0.so"
)
# VSS_EXTENSION_PATH = os.path.join(
#     os.path.dirname(os.path.dirname(__file__)), "lib", "vss0.so"
# ) # Removed VSS path
# <<< FIM DA ADIÇÃO >>>

_db_connections: Dict[str, sqlite3.Connection] = {}

def get_db_connection(db_path: Optional[str] = None):
    """Retorna uma conexão com o banco de dados SQLite. Usa db_path se fornecido, senão DATABASE_PATH."""
    target_db_path = db_path or DATABASE_PATH
    logger.debug(f"[DB_UTILS] get_db_connection called for path: {target_db_path}")
    if target_db_path in _db_connections and _db_connections[target_db_path]:
        # TODO: Check if connection is still valid?
        return _db_connections[target_db_path]
    conn = None
    try:
        # Use o path_to_use para conectar
        conn = sqlite3.connect(target_db_path, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        # <<< EXTENSION LOADING REMOVED FROM HERE >>>
        # try:
        #     # ... loading logic removed ...
        # except sqlite3.OperationalError as e:
        #     # ... exception handling removed ...
        # except Exception as e_load:
        #     # ... exception handling removed ...

        # <<< SECURITY DISABLE REMOVED FROM HERE >>>
        # conn.enable_load_extension(False)

        _db_connections[target_db_path] = conn
        return conn
    except sqlite3.Error as e:
        logger.error(f"Erro CRÍTICO ao conectar ao DB em '{target_db_path}': {e}")
        if conn:
            conn.close()
        raise
    # Não fechar a conexão aqui, quem chama é responsável por fechar


async def close_db_connection(conn):
    """Fecha a conexão com o banco de dados de forma assíncrona."""
    if conn:
        await asyncio.to_thread(conn.close) # Use to_thread for blocking IO
        logger.debug("Database connection closed.")


# <<< ADICIONADA: Função para inicializar o DB >>>
def initialize_database(db_path: Optional[str] = None):
    """Cria as tabelas necessárias no banco de dados se não existirem."""
    conn = None # Initialize conn
    try:
        conn = get_db_connection(db_path=db_path) # Pass db_path
        cursor = conn.cursor()
        logger.info(f"Inicializando/Verificando estrutura do banco de dados em '{db_path or DATABASE_PATH}'...")

        # --- Criação GARANTIDA de agent_state ---
        logger.info("Verificando/Criando tabela 'agent_state'...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_state (
                agent_id TEXT PRIMARY KEY, 
                history TEXT,              -- Store history as JSON text
                memory TEXT,               -- Store memory as JSON text
                last_code TEXT,            -- Keep for potential backward compatibility or other uses?
                last_lang TEXT,            -- Keep for potential backward compatibility or other uses?
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        logger.info("Verificando/Criando trigger 'update_agent_state_updated_at'...")
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS update_agent_state_updated_at
            AFTER UPDATE ON agent_state FOR EACH ROW
            BEGIN
                UPDATE agent_state SET updated_at = CURRENT_TIMESTAMP WHERE agent_id = OLD.agent_id;
            END;
        """
        )
        conn.commit()  # Commit IMEDIATO para agent_state
        logger.info("Tabela 'agent_state' e trigger verificados/criados.")
        # --- Fim Bloco agent_state ---

        # --- Criação Tabelas Semantic Memory ---
        logger.info("Verificando/Criando tabela 'semantic_memory'...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL UNIQUE, -- Texto original
                embedding BLOB NOT NULL,    -- Vetor embedding
                metadata TEXT,             -- JSON para infos extras
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        logger.info("Tabela 'semantic_memory' verificada/criada.")

        # --- Criação Tabela Experience Buffer ---
        logger.info("Verificando/Criando tabela 'experience_buffer'...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS experience_buffer (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context TEXT,              -- Prompt ou situação que levou à ação
                action TEXT,               -- Ação realizada (código, comando, resposta)
                outcome TEXT,              -- Resultado (success, failure, error_log, user_feedback)
                priority REAL DEFAULT 1.0, -- Prioridade calculada para amostragem (default 1.0)
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Quando ocorreu
                metadata TEXT              -- JSON para infos extras (skill usada, confiança, etc.)
            )
        """
        )
        logger.info("Tabela 'experience_buffer' verificada/criada.")
        # --- Fim Bloco Experience Buffer ---

        conn.commit()  # Commit final para knowledge, semantic_memory, VSS
        logger.info("Estrutura completa do banco de dados verificada/atualizada.")

    except sqlite3.Error as e:
        logger.error(f"Erro CRÍTICO durante a inicialização do banco de dados: {e}")
        # Considerar se deve levantar o erro ou tentar continuar
        raise
    finally:
        pass # Removed conn.close()


# Funções auxiliares para salvar/carregar estado (adicionar aqui)
def save_agent_state(agent_id: str, state: dict, db_path: Optional[str] = None):
    conn = None
    history_json = None
    memory_json = None
    try:
        # Serialize history and memory to JSON
        # Provide default empty list/dict if not present in state
        history_json = json.dumps(state.get('history', []))
        memory_json = json.dumps(state.get('memory', {}))
        
        conn = get_db_connection(db_path=db_path) # Pass db_path
        cursor = conn.cursor()
        cursor.execute(
            """
             INSERT OR REPLACE INTO agent_state (agent_id, history, memory, last_code, last_lang) 
             VALUES (?, ?, ?, ?, ?)
         """,
            (
                agent_id, 
                history_json, 
                memory_json,
                state.get("last_code"), # Keep saving these for now
                state.get("last_lang")
            ),
        )
        conn.commit()
        logger.info(f"Estado do agente '{agent_id}' (incluindo histórico e memória) salvo no DB.")
    except sqlite3.Error as e:
        logger.error(f"Erro ao salvar estado do agente '{agent_id}': {e}")
    except json.JSONDecodeError as json_err:
        logger.error(f"Erro ao serializar estado para JSON para agente '{agent_id}': {json_err}")
    finally:
        pass # Removed conn.close()


def load_agent_state(agent_id: str, db_path: Optional[str] = None) -> dict:
    conn = None
    # Initialize state with defaults for all expected keys, including agent_id
    state = {"agent_id": agent_id, "history": [], "memory": {}, "last_code": None, "last_lang": None}
    try:
        conn = get_db_connection(db_path=db_path) # Pass db_path
        cursor = conn.cursor()
        # Select all relevant columns from agent_state
        cursor.execute(
            """
             SELECT history, memory, last_code, last_lang 
             FROM agent_state 
             WHERE agent_id = ?
         """,
            (agent_id,),
        )
        row = cursor.fetchone()
        
        if row:
            history_json = row['history']
            memory_json = row['memory']
            
            # Attempt to load history from JSON
            if history_json:
                try:
                    state['history'] = json.loads(history_json)
                except json.JSONDecodeError as json_err:
                    logger.error(f"Erro ao desserializar histórico JSON para agente '{agent_id}': {json_err}")
                    # Keep default empty list on error
            
            # Attempt to load memory from JSON
            if memory_json:
                try:
                    state['memory'] = json.loads(memory_json)
                except json.JSONDecodeError as json_err:
                    logger.error(f"Erro ao desserializar memória JSON para agente '{agent_id}': {json_err}")
                    # Keep default empty dict on error

            # Load legacy/other fields
            state["last_code"] = row["last_code"] # Can be None if not previously saved
            state["last_lang"] = row["last_lang"]
            
            logger.info(f"Estado do agente '{agent_id}' (incluindo histórico e memória) carregado do DB.")
        else:
            logger.info(f"Nenhum estado salvo encontrado para agente '{agent_id}'. Usando estado padrão.")
            
    except sqlite3.OperationalError as e:
        # Error likely means the table or columns don't exist yet
        logger.error(
            f"Erro operacional ao carregar estado do agente '{agent_id}' (tabela existe?): {e}"
        )
    except sqlite3.Error as e:
        logger.error(f"Erro SQLite ao carregar estado do agente '{agent_id}': {e}")
    finally:
        pass # Removed conn.close()
    return state


# --- Funções para Experience Buffer --- #

# Renamed from record_experience
def add_episodic_record(context: str, action: str, outcome: str, metadata: dict | None = None):
    """Registra um ciclo de experiência completo (episódio) no buffer."""
    conn = None
    try:
        # Lógica inicial simples de prioridade: falhas/erros têm prioridade maior
        initial_priority = 2.0 if "failure" in outcome.lower() or "error" in outcome.lower() else 1.0

        metadata_json = json.dumps(metadata) if metadata else None

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO experience_buffer (context, action, outcome, priority, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (context, action, outcome, initial_priority, metadata_json)
        )
        conn.commit()
        logger.info(f"Experiência registrada no buffer (Priority: {initial_priority:.1f}).")

    except sqlite3.Error as e:
        logger.error(f"Erro ao registrar experiência: {e}")
    finally:
        pass


def sample_experiences(batch_size: int) -> list[sqlite3.Row]:
    """Amostra experiências do buffer, ponderadas pela prioridade."""
    conn = None
    experiences = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 1. Obter todas as experiências com suas prioridades
        cursor.execute("SELECT id, priority FROM experience_buffer ORDER BY timestamp DESC") # Limitar opcionalmente?
        all_experiences = cursor.fetchall()

        if not all_experiences:
            logger.info("Buffer de experiências vazio, nenhuma amostra retirada.")
            return []

        # 2. Preparar IDs e Pesos para amostragem
        ids = [row['id'] for row in all_experiences]
        # Normalizar prioridades para que somem 1 (se não forem todas 0)
        priorities = [max(0.0, row['priority']) for row in all_experiences] # Garante não negativo
        total_priority = sum(priorities)
        weights = [p / total_priority for p in priorities] if total_priority > 0 else None

        # 3. Amostrar IDs usando random.choices
        import random
        # Garante que não tentamos amostrar mais do que temos
        sample_size = min(batch_size, len(ids))
        sampled_ids = random.choices(ids, weights=weights, k=sample_size)

        # 4. Buscar as experiências completas correspondentes aos IDs amostrados
        # Usar placeholders para segurança e eficiência
        placeholders = ', '.join('?' * len(sampled_ids))
        query = f"SELECT * FROM experience_buffer WHERE id IN ({placeholders})"
        cursor.execute(query, sampled_ids)
        experiences = cursor.fetchall()

        logger.info(f"Amostradas {len(experiences)} experiências do buffer.")

    except sqlite3.Error as e:
        logger.error(f"Erro ao amostrar experiências: {e}")
    except ImportError:
        logger.error("Módulo 'random' não encontrado para amostragem.")
    finally:
        pass
    return experiences


# Função para buscar experiências baseadas em prioridade (usada pelo trainer)
def retrieve_recent_episodes(limit: int = 5) -> list[sqlite3.Row]:
    """Recupera os 'limit' episódios mais recentes do buffer de experiência."""
    conn = None
    episodes = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Seleciona todos os campos, ordenando pelo timestamp decrescente e limitando
        cursor.execute(
            f"""
            SELECT id, context, action, outcome, timestamp, metadata 
            FROM experience_buffer 
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,)
        )
        episodes = cursor.fetchall() # Retorna lista de Rows
        logger.info(f"Recuperados {len(episodes)} episódios recentes da memória.")

    except sqlite3.Error as e:
        logger.error(f"Erro ao recuperar episódios recentes: {e}")
        # Retorna lista vazia em caso de erro
    finally:
        pass
    return episodes


# --- Funções para Semantic Memory / VSS --- #

# ADDED: New function for semantic context retrieval
def retrieve_relevant_context(objective: str, top_k: int = 5) -> list[str]:
    """Recupera contextos relevantes da memória semântica usando sqlite-vec."""
    conn = None
    results = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Verifica se a tabela VEC existe
        cursor.execute("SELECT name FROM sqlite_master WHERE type=\'table\' AND name=\'vec_semantic_memory\'")
        vec_table_exists = cursor.fetchone()

        if not vec_table_exists:
            logger.warning("Tabela VEC \'vec_semantic_memory\' não encontrada. Não é possível realizar busca vetorial.")
            return []

        # 1. Gerar embedding para o objetivo
        from a3x.core.embeddings import get_embedding # Import local para evitar dependência circular ou carregar modelo cedo demais
        query_embedding = get_embedding(objective)
        if query_embedding is None:
            logger.error(f"Falha ao gerar embedding para o objetivo: \'{objective[:100]}...\'")
            return []

        # 2. Preparar query VEC
        # Convert embedding to JSON string for the query (sqlite-vec expects this)
        query_embedding_json = json.dumps(query_embedding)

        # Executar busca vetorial using a subquery for the LIMIT
        # Syntax for sqlite-vec uses MATCH
        vec_query = f"""
            SELECT
                sm.content,
                sm.metadata,
                vec_results.distance
            FROM
                ( -- Start Subquery --
                    SELECT
                        rowid,
                        distance
                    FROM vec_semantic_memory
                    WHERE embedding MATCH ? -- Use MATCH operator
                    ORDER BY distance       -- sqlite-vec orders by distance implicitly
                    LIMIT ?                 -- Apply LIMIT within the subquery
                ) vec_results -- End Subquery --
            JOIN
                semantic_memory sm ON vec_results.rowid = sm.id
            ORDER BY
                vec_results.distance -- Order the final joined results by distance
        """
        logger.debug(f"Executando busca VEC com subquery LIMIT={top_k}. Query: {vec_query}")
        # Pass query_embedding_json and top_k as parameters
        cursor.execute(vec_query, (query_embedding_json, top_k))
        rows = cursor.fetchall()

        results = [row['content'] for row in rows] # Retorna apenas o conteúdo por simplicidade
        logger.info(f"Recuperados {len(results)} contextos relevantes via VEC para o objetivo.")

    except ImportError:
        logger.error("Erro ao importar \'get_embedding\'. A função de embeddings está disponível?")
    except sqlite3.Error as e:
        logger.error(f"Erro SQLite durante a busca vetorial (VEC): {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Erro inesperado durante a busca vetorial (VEC): {e}", exc_info=True)
    finally:
        pass
    # return results # Don't return results from here, it's handled by the skill


# Comentado para evitar execução automática na importação.
# A inicialização será chamada explicitamente em assistant_cli.py
# initialize_database()
