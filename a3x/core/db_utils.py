import sqlite3
import os
import logging  # Usar logging é melhor que print para debug
import json

# Configurar logging básico
logging.basicConfig(level=logging.INFO, format="[%(levelname)s DB] %(message)s")
logger = logging.getLogger(__name__)

# Caminho para o DB (pode estar em config.py ou definido aqui)
# Garanta que aponta para o memory.db na raiz do projeto
try:
    # Tenta encontrar a raiz do projeto subindo diretórios
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    # Sobe até encontrar um diretório que NÃO seja 'core' ou que contenha um marcador (ex: .git, pyproject.toml)
    while os.path.basename(project_root) == "core":
        project_root = os.path.dirname(project_root)
        if not project_root or project_root == "/":  # Evita loop infinito
            project_root = os.path.dirname(current_dir)  # Volta um nível se der errado
            logger.warning(
                "Não foi possível determinar a raiz do projeto de forma confiável subindo diretórios."
            )
            break

    DATABASE_PATH = os.path.join(project_root, "memory.db")
    logger.info(f"Database path set to: {DATABASE_PATH}")
except Exception as e:
    logger.error(f"Error determining project root or database path: {e}")
    # Fallback path - pode não ser ideal
    DATABASE_PATH = os.path.join(os.getcwd(), "memory.db")
    logger.warning(f"Falling back to database path: {DATABASE_PATH}")

# <<< ADICIONAR CAMINHO DA EXTENSÃO >>>
# AJUSTE ESTE CAMINHO se você colocou as extensões em outro lugar!
VECTOR_EXTENSION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "lib", "vector0.so"
)
VSS_EXTENSION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "lib", "vss0.so"
)
# <<< FIM DA ADIÇÃO >>>


def get_db_connection():
    """Retorna uma conexão com o banco de dados com a extensão VSS carregada (se possível)."""
    conn = None  # Inicializa conn
    try:
        conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        # conn.enable_load_extension(True)  # Temporariamente desabilitado para contornar erro em Python compilado sem suporte a SQLite extensions
        conn.row_factory = sqlite3.Row

        # Tenta carregar extensões VSS
        # try:
        #     # Verifica se os arquivos existem
        #     if not os.path.exists(VECTOR_EXTENSION_PATH):
        #         logger.warning(f"Arquivo da extensão vector0 NÃO encontrado no caminho esperado: {VECTOR_EXTENSION_PATH}")
        #     if not os.path.exists(VSS_EXTENSION_PATH):
        #         logger.warning(f"Arquivo da extensão vss0 NÃO encontrado no caminho esperado: {VSS_EXTENSION_PATH}")

        #     # Carrega as extensões na ordem correta
        #     conn.load_extension(VECTOR_EXTENSION_PATH)
        #     logger.info("Extensão vector0 carregada com sucesso!")

        #     conn.load_extension(VSS_EXTENSION_PATH)
        #     logger.info("Extensão vss0 carregada com sucesso!")
        # except sqlite3.OperationalError as e:
        #      logger.warning(f"Falha ao carregar extensão sqlite-vss: {e}. Busca vetorial estará DESABILITADA.")
        # conn.enable_load_extension(False)

        return conn
    except sqlite3.Error as e:
        logger.error(f"Erro CRÍTICO ao conectar ao DB: {e}")
        if conn:
            conn.close()  # Tenta fechar se falhar após conectar
        raise  # Levanta o erro para indicar falha grave
    # Não fechar a conexão aqui, quem chama é responsável por fechar


def initialize_database():
    """Garante que todas as tabelas necessárias existam."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        logger.info("Inicializando/Verificando estrutura do banco de dados...")

        # --- Criação GARANTIDA de agent_state ---
        logger.info("Verificando/Criando tabela 'agent_state'...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_state (
                agent_id TEXT PRIMARY KEY, 
                last_code TEXT,
                last_lang TEXT,
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

        # Tenta criar tabela VSS (com try/except separado)
        try:
            cursor.execute("SELECT vss_version()")
            logger.info("Extensão VSS funcional detectada.")
            EMBEDDING_DIM = 768  # Para ibm-granite
            logger.info("Verificando/Criando tabela virtual 'vss_semantic_memory'...")
            cursor.execute(
                f"""
                 CREATE VIRTUAL TABLE IF NOT EXISTS vss_semantic_memory USING vss0(
                     embedding({EMBEDDING_DIM}) 
                 )
             """
            )
            logger.info("Tabela virtual 'vss_semantic_memory' verificada/criada.")
        except sqlite3.OperationalError as e:
            logger.warning(
                f"Não foi possível criar/verificar tabela VSS (extensão pode não estar carregada ou erro): {e}"
            )
        # --- Fim Bloco Semantic Memory ---

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
        if conn:
            conn.close()


# Funções auxiliares para salvar/carregar estado (adicionar aqui)
def save_agent_state(agent_id: str, state: dict):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
             INSERT OR REPLACE INTO agent_state (agent_id, last_code, last_lang, updated_at) 
             VALUES (?, ?, ?, CURRENT_TIMESTAMP)
         """,
            (agent_id, state.get("last_code"), state.get("last_lang")),
        )
        conn.commit()
        logger.info(f"Estado do agente '{agent_id}' salvo.")
    except sqlite3.Error as e:
        logger.error(f"Erro ao salvar estado do agente '{agent_id}': {e}")
        # Não levantar erro aqui, para o agente poder continuar mesmo se salvar falhar
    finally:
        if conn:
            conn.close()


def load_agent_state(agent_id: str) -> dict:
    conn = None
    state = {"last_code": None, "last_lang": None}
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Tenta ler a tabela, se ela não existir (devido a erro na inicialização), isso falhará
        cursor.execute(
            """
             SELECT last_code, last_lang 
             FROM agent_state 
             WHERE agent_id = ?
         """,
            (agent_id,),
        )
        row = cursor.fetchone()
        if row and row["last_code"] is not None:
            state["last_code"] = row["last_code"]
            state["last_lang"] = row["last_lang"]
            logger.info(f"Estado do agente '{agent_id}' carregado do DB.")
        else:
            logger.info(f"Nenhum estado salvo encontrado para agente '{agent_id}'.")
    except sqlite3.OperationalError as e:
        # Erro comum se a tabela agent_state não existir
        logger.error(
            f"Erro operacional ao carregar estado do agente '{agent_id}' (tabela existe?): {e}"
        )
    except sqlite3.Error as e:
        logger.error(f"Erro SQLite ao carregar estado do agente '{agent_id}': {e}")
    finally:
        if conn:
            conn.close()
    return state


# --- Funções para Experience Buffer --- #

def record_experience(context: str, action: str, outcome: str, metadata: dict | None = None):
    """Registra uma nova experiência no buffer."""
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
        if conn:
            conn.close()


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
        if conn:
            conn.close()
    return experiences


# Comentado para evitar execução automática na importação.
# A inicialização será chamada explicitamente em assistant_cli.py
# initialize_database()
