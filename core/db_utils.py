import sqlite3
import os
import json # Para serializar/desserializar dados complexos se necessário no futuro
import logging # Usar logging é melhor que print para debug
import time

# Configurar logging básico
logging.basicConfig(level=logging.INFO, format='[%(levelname)s DB Utils] %(message)s')
logger = logging.getLogger(__name__)

# Caminho para o DB (pode estar em config.py ou definido aqui)
# Garanta que aponta para o memory.db na raiz do projeto
try:
    # Tenta encontrar a raiz do projeto subindo diretórios
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    # Sobe até encontrar um diretório que NÃO seja 'core' ou que contenha um marcador (ex: .git, pyproject.toml)
    while os.path.basename(project_root) == 'core':
        project_root = os.path.dirname(project_root)
        if not project_root or project_root == '/': # Evita loop infinito
            project_root = os.path.dirname(current_dir) # Volta um nível se der errado
            logger.warning("Não foi possível determinar a raiz do projeto de forma confiável subindo diretórios.")
            break

    DATABASE_PATH = os.path.join(project_root, 'memory.db')
    logger.info(f"Database path set to: {DATABASE_PATH}")
except Exception as e:
    logger.error(f"Error determining project root or database path: {e}")
    # Fallback path - pode não ser ideal
    DATABASE_PATH = os.path.join(os.getcwd(), 'memory.db')
    logger.warning(f"Falling back to database path: {DATABASE_PATH}")

# <<< ADICIONAR CAMINHO DA EXTENSÃO >>>
# AJUSTE ESTE CAMINHO se você colocou o sqlite-vss.so em outro lugar!
SQLITE_VSS_EXTENSION_PATH = os.path.join(os.path.dirname(__file__), 'sqlite-vss.so')
# <<< FIM DA ADIÇÃO >>>

def get_db_connection():
    """Retorna uma conexão com o banco de dados com a extensão VSS carregada (se disponível)."""
    try:
        logger.debug(f"Tentando conectar ao banco de dados em: {DATABASE_PATH}")
        conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False) # check_same_thread pode ser útil
        conn.row_factory = sqlite3.Row
        logger.debug("Conexão SQLite estabelecida.")

        # <<< CARREGAR EXTENSÃO VSS >>>
        conn.enable_load_extension(True)
        logger.debug("Carregamento de extensão habilitado.")
        try:
             if not os.path.exists(SQLITE_VSS_EXTENSION_PATH):
                  # Log mais informativo
                  logger.error(f"Arquivo da extensão sqlite-vss NÃO encontrado no caminho esperado: {SQLITE_VSS_EXTENSION_PATH}")
                  raise FileNotFoundError(f"Extensão sqlite-vss não encontrada em {SQLITE_VSS_EXTENSION_PATH}")

             logger.debug(f"Tentando carregar extensão de: {SQLITE_VSS_EXTENSION_PATH}")
             conn.load_extension(SQLITE_VSS_EXTENSION_PATH)
             logger.info("Extensão sqlite-vss carregada com sucesso.") # Mudado para INFO
        except FileNotFoundError as fnf_err:
             # Log do erro específico de não encontrar o arquivo
             logger.warning(f"{fnf_err}. Busca vetorial estará DESABILITADA.")
        except sqlite3.OperationalError as e:
             # Log de erro operacional (ex: arquitetura errada, problema na compilação)
             logger.error(f"Falha ao carregar extensão sqlite-vss em '{SQLITE_VSS_EXTENSION_PATH}': {e}")
             logger.warning("Busca vetorial estará DESABILITADA.")
             # Não levanta erro aqui, permite continuar sem VSS
        except Exception as load_err:
            # Captura outros erros inesperados durante o carregamento
            logger.exception(f"Erro inesperado ao carregar a extensão VSS: {load_err}")
            logger.warning("Busca vetorial estará DESABILITADA.")

        # Desabilitar por segurança após tentar carregar
        conn.enable_load_extension(False)
        logger.debug("Carregamento de extensão desabilitado.")
        # <<< FIM CARREGAR EXTENSÃO >>>

        return conn
    except sqlite3.Error as e:
        logger.error(f"Erro CRÍTICO ao conectar/configurar o DB em {DATABASE_PATH}: {e}")
        raise # Levanta o erro para indicar falha na conexão

def initialize_database():
    """Garante que todas as tabelas necessárias existam, incluindo as de memória semântica."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        logger.info("Inicializando/Verificando estrutura do banco de dados...")

        # Primeiro, verifica se a tabela agent_state existe
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_state'")
        if not cursor.fetchone():
            logger.info("Criando tabela 'agent_state'...")
            cursor.execute('''
                CREATE TABLE agent_state (
                    agent_id TEXT PRIMARY KEY,
                    last_code TEXT,
                    last_lang TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            logger.info("Tabela 'agent_state' criada com sucesso.")

        # Trigger para atualizar 'updated_at' automaticamente
        cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name='update_agent_state_updated_at'")
        if not cursor.fetchone():
            logger.info("Criando trigger 'update_agent_state_updated_at'...")
            cursor.execute('''
                CREATE TRIGGER update_agent_state_updated_at
                AFTER UPDATE ON agent_state
                FOR EACH ROW
                BEGIN
                    UPDATE agent_state SET updated_at = CURRENT_TIMESTAMP WHERE agent_id = OLD.agent_id;
                END;
            ''')
            logger.info("Trigger 'update_agent_state_updated_at' criado com sucesso.")

        # Tabela para memória semântica
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='semantic_memory'")
        if not cursor.fetchone():
            logger.info("Criando tabela 'semantic_memory'...")
            cursor.execute('''
                CREATE TABLE semantic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL UNIQUE,
                    embedding BLOB NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            logger.info("Tabela 'semantic_memory' criada com sucesso.")

        # Trigger para semantic_memory
        cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name='update_semantic_memory_timestamp'")
        if not cursor.fetchone():
            logger.info("Criando trigger 'update_semantic_memory_timestamp'...")
            cursor.execute('''
                CREATE TRIGGER update_semantic_memory_timestamp
                AFTER UPDATE ON semantic_memory
                FOR EACH ROW
                BEGIN
                    UPDATE semantic_memory SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
                END;
            ''')
            logger.info("Trigger 'update_semantic_memory_timestamp' criado com sucesso.")

        # Tabela VSS (se a extensão estiver disponível)
        try:
            cursor.execute("SELECT vss_version()")
            vss_version = cursor.fetchone()[0]
            logger.info(f"Extensão VSS funcional detectada (Versão: {vss_version}).")

            EMBEDDING_DIM = 768
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vss_semantic_memory'")
            if not cursor.fetchone():
                logger.info("Criando tabela virtual 'vss_semantic_memory'...")
                cursor.execute(f'''
                    CREATE VIRTUAL TABLE vss_semantic_memory USING vss0(
                        embedding({EMBEDDING_DIM})
                    )
                ''')
                logger.info("Tabela virtual 'vss_semantic_memory' criada com sucesso.")
        except sqlite3.OperationalError as e:
            if "no such function: vss_version" in str(e):
                logger.warning("Extensão VSS não parece estar carregada ou funcional. Tabela VSS não criada.")
            else:
                logger.warning(f"Não foi possível criar/verificar tabela VSS (extensão carregada, mas erro): {e}")

        conn.commit()
        logger.info("Banco de dados inicializado/verificado com sucesso.")
    except sqlite3.Error as e:
        logger.error(f"Erro CRÍTICO durante a inicialização do banco de dados: {e}")
        raise  # Levanta o erro para indicar falha na inicialização
    except Exception as general_e:
        logger.exception(f"Erro inesperado durante a inicialização do banco de dados: {general_e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.debug("Conexão SQLite fechada.")

# Funções auxiliares para salvar/carregar estado (adicionar aqui)
def save_agent_state(agent_id: str, state: dict):
     """Salva ou atualiza o estado (last_code, last_lang) de um agente no DB."""
     # <<< VERIFICAR SE state TEM AS CHAVES ESPERADAS >>>
     if not all(k in state for k in ('last_code', 'last_lang')):
          logger.warning(f"Tentativa de salvar estado incompleto para agente '{agent_id}'. Estado: {state}")
          # Decide se retorna erro ou salva o que tem. Por enquanto, salva o que tem.
          pass # Permite salvar mesmo que incompleto

     conn = get_db_connection()
     if conn is None:
         logger.error(f"Falha ao obter conexão com DB para salvar estado (agente: {agent_id}).")
         return
     try:
         cursor = conn.cursor()
         # Usar INSERT ON CONFLICT DO UPDATE para ser mais explícito
         cursor.execute('''
             INSERT INTO agent_state (agent_id, last_code, last_lang, updated_at)
             VALUES (?, ?, ?, CURRENT_TIMESTAMP)
             ON CONFLICT(agent_id) DO UPDATE SET
                 last_code = excluded.last_code,
                 last_lang = excluded.last_lang,
                 updated_at = CURRENT_TIMESTAMP;
         ''', (agent_id, state.get('last_code'), state.get('last_lang')))
         conn.commit()
         logger.info(f"Estado do agente '{agent_id}' salvo.") # Log INFO
     except sqlite3.Error as e:
         logger.error(f"Erro ao salvar estado para agente '{agent_id}': {e}")
     finally:
         if conn:
             conn.close()

def load_agent_state(agent_id: str) -> dict:
     """Carrega o estado (last_code, last_lang) de um agente do DB. Retorna defaults se não encontrado."""
     conn = get_db_connection()
     # <<< Inicializa com defaults >>>
     state = {'last_code': None, 'last_lang': None}
     if conn is None:
        logger.error(f"Falha ao obter conexão com DB para carregar estado (agente: {agent_id}).")
        return state # Retorna defaults

     try:
         cursor = conn.cursor()
         cursor.execute('''
             SELECT last_code, last_lang
             FROM agent_state
             WHERE agent_id = ?
         ''', (agent_id,))
         row = cursor.fetchone()
         if row:
              # Não precisa verificar se last_code é None, o dict já tem None como default
              state['last_code'] = row['last_code']
              state['last_lang'] = row['last_lang']
              logger.info(f"Estado do agente '{agent_id}' carregado.")
         else:
              logger.info(f"Nenhum estado salvo encontrado para agente '{agent_id}'. Usando defaults.")
     except sqlite3.Error as e:
         logger.error(f"Erro ao carregar estado para agente '{agent_id}': {e}")
     finally:
         if conn:
             conn.close()
     return state

# Comentado para evitar execução automática na importação.
# A inicialização será chamada explicitamente em assistant_cli.py
# initialize_database() 