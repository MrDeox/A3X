import os
from dotenv import load_dotenv

# <<< ADDED: Define project_root >>>
# Assume config.py is in the 'core' directory, so go up one level
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Carregar variáveis de ambiente do arquivo .env
load_dotenv(os.path.join(project_root, '.env')) # Load from project root

# URL do servidor LLAMA (Ollama, LM Studio, etc.)
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1/chat/completions") # Default if not set

# Chave da API Tavily (se usar busca web com Tavily)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)

# Configurações de Execução de Código
PYTHON_EXEC_TIMEOUT = 10 # Segundos máximos para execução de código
# FIREJAIL_PROFILE = os.path.join(os.path.dirname(__file__), 'python.profile') # Path relative to config.py - REMOVED/COMMENTED

# Configurações do Agente ReAct
MAX_REACT_ITERATIONS = int(os.getenv("MAX_REACT_ITERATIONS", "10"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))
MAX_META_DEPTH = 3 # Profundidade máxima para chamadas recursivas de auto-correção/reflexão

# Configurações de Histórico (se aplicável fora do agente)
MAX_HISTORY_TURNS = 5 # Keep one definition

# Outras configurações globais podem ser adicionadas aqui
# Ex: Nível de log, caminhos de diretório padrão, etc.

# Configuração do histórico
# MAX_HISTORY_TURNS = 5  # Quantos pares (usuário + assistente) lembrar - REMOVED DUPLICATE 

# Configurações do Servidor LLM (Llama.cpp)
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "nokey") # 'nokey' ou sua chave se o servidor exigir
LLAMA_DEFAULT_HEADERS = {"Content-Type": "application/json"}
# Adicionar Authorization se LLAMA_API_KEY for definida
if LLAMA_API_KEY and LLAMA_API_KEY.lower() not in ['none', 'nokey', '']:
    LLAMA_DEFAULT_HEADERS["Authorization"] = f"Bearer {LLAMA_API_KEY}"

# Configurações do Banco de Dados (SQLite)
# DATABASE_PATH = os.getenv("DATABASE_PATH", "a3x_memory.db") # Deprecated or unused?
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", os.path.join(project_root, "memory.db")) # <<< Usa project_root >>>

# Configurações de Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Configurações de Ferramentas Externas
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Outras configurações (se necessário)
# ... 