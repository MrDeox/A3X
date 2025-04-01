import os
from dotenv import load_dotenv
import logging

# <<< ADDED: Define project_root >>>
# Assume config.py is in the 'core' directory, so go up one level
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# <<< EXPORTING project_root as PROJECT_ROOT >>>
PROJECT_ROOT = project_root

# Carregar variáveis de ambiente do arquivo .env
load_dotenv(os.path.join(project_root, '.env')) # Load from project root

# <<< ADDED BACK Commented LLAMA_SERVER_URL for reference/potential use >>>
# URL do servidor LLAMA (Ollama, LM Studio, etc.)
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1/chat/completions") # Default if not set

# <<< ADDED: Define LLM Provider >>>
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "llama.cpp") # Default to llama.cpp

# Chave da API Tavily (se usar busca web com Tavily)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)

# Configurações de Execução de Código
PYTHON_EXEC_TIMEOUT = 10 # Segundos máximos para execução de código
# FIREJAIL_PROFILE = os.path.join(os.path.dirname(__file__), 'python.profile') # Path relative to config.py - REMOVED/COMMENTED - REMOVED

# Configurações do Agente ReAct
MAX_REACT_ITERATIONS = int(os.getenv("MAX_REACT_ITERATIONS", "10"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5")) # Keep one definition
MAX_META_DEPTH = 3 # Profundidade máxima para chamadas recursivas de auto-correção/reflexão

# Configurações de Histórico (se aplicável fora do agente) - REMOVED DUPLICATE
# MAX_HISTORY_TURNS = 5 # Keep one definition

# Outras configurações globais podem ser adicionadas aqui
# Ex: Nível de log, caminhos de diretório padrão, etc.

# Configuração do histórico - REMOVED DUPLICATE COMMENT
# MAX_HISTORY_TURNS = 5  # Quantos pares (usuário + assistente) lembrar - REMOVED DUPLICATE

# Configurações do Servidor LLM (Llama.cpp)
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "nokey") # 'nokey' ou sua chave se o servidor exigir
LLAMA_DEFAULT_HEADERS = {"Content-Type": "application/json"} # Keep first definition
# Adicionar Authorization se LLAMA_API_KEY for definida
if LLAMA_API_KEY and LLAMA_API_KEY.lower() not in ['none', 'nokey', '']:
    LLAMA_DEFAULT_HEADERS["Authorization"] = f"Bearer {LLAMA_API_KEY}"

# Configurações do Banco de Dados (SQLite)
# DATABASE_PATH = os.getenv("DATABASE_PATH", "a3x_memory.db") # Deprecated or unused? - REMOVED
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", os.path.join(project_root, "memory.db")) # <<< Usa project_root >>>

# Configurações de Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # Keep first definition
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# <<< REMOVING TAVILY CONFIG BLOCK >>>
# Configurações de Ferramentas Externas
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# TAVILY_ENABLED = os.getenv("TAVILY_ENABLED", "False").lower() == "true"

# Outras configurações (se necessário)
# ...

# --- General Configuration --- REMOVED BLOCK START
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
# DB_FILE = os.getenv('DB_FILE', os.path.join(PROJECT_ROOT, 'memory.db')) # <<< Define DB_FILE >>> REMOVED (Using MEMORY_DB_PATH)

# --- Agent Configuration ---
AGENT_STATE_ID = os.getenv('AGENT_STATE_ID', '1') # Default agent state ID - Kept this definition assuming it's used
# MAX_REACT_ITERATIONS = int(os.getenv('MAX_REACT_ITERATIONS', '10')) - REMOVED DUPLICATE
# ... (rest of the file)

# --- Configurações do Servidor LLM (llama.cpp) ---
# Caminho para o executável do servidor llama.cpp (se não estiver no PATH)
# LLAMA_SERVER_EXECUTABLE = os.environ.get("LLAMA_SERVER_EXECUTABLE", os.path.join(PROJECT_ROOT, "llama.cpp/build/bin/llama-server")) # Corrigido para apontar para o build <<< COMMENTED OUT - Using main binary directly - REMOVED
# Caminho para o arquivo do modelo GGUF
LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", os.path.join(project_root, "models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")) # <<< Using Llama 3, ensure path relative to project_root (corrected variable name)
# Argumentos adicionais para passar ao servidor llama.cpp
# Ex: número de camadas GPU, tamanho do contexto, etc.
# IMPORTANTE: Garanta que o tamanho do contexto (-c) aqui seja consistente ou maior que o usado nos prompts.
# LLAMA_SERVER_ARGS = os.environ.get("LLAMA_SERVER_ARGS", "-c 4096 --temp 0.2 --top-p 0.95 --repeat-penalty 1.15 --ngl 28") # <-- Novos args para OpenChat <<< COMMENTED OUT - Args passed directly to main binary - REMOVED
# URL completa do endpoint de chat do servidor llama.cpp
# LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1/chat/completions") <<< COMMENTED OUT - Not using HTTP server - REMOVED
# Cabeçalhos HTTP adicionais (ex: para autenticação, se necessário) - REMOVED DUPLICATE
# LLAMA_DEFAULT_HEADERS = {
#     "Content-Type": "application/json",
#     # "Authorization": f"Bearer {os.getenv('LLAMA_API_KEY')}" # Exemplo se API key fosse necessária
# }

# --- Configurações de Banco de Dados e Memória ---
# DB_FILE = os.environ.get("DB_FILE", "memory.db") - REMOVED DUPLICATE

# Fallback máximo de tokens para respostas do LLM
MAX_TOKENS_FALLBACK = int(os.environ.get("MAX_TOKENS_FALLBACK", 1024))

# Tamanho do contexto do modelo LLM (deve ser compatível com o modelo carregado)
CONTEXT_SIZE = int(os.environ.get("CONTEXT_SIZE", 4096)) 