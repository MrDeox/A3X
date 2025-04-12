import os
from dotenv import load_dotenv
import platform # Add platform import
from pathlib import Path

# Define project root based on this file's location
PROJECT_ROOT = Path(__file__).parent.parent.parent # Go up 3 levels from core/config.py

# Carregar variáveis de ambiente do arquivo .env
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))  # Load from project root

# <<< ADDED BACK Commented LLAMA_SERVER_URL for reference/potential use >>>
# URL do servidor LLAMA (Ollama, LM Studio, etc.)
LLAMA_SERVER_URL = os.getenv(
    "LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1/chat/completions"
)  # Default if not set

# <<< ADDED: Define LLM Provider >>>
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "llama.cpp")  # Default to llama.cpp

# Chave da API Tavily (se usar busca web com Tavily)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)

# Configurações de Execução de Código
PYTHON_EXEC_TIMEOUT = 10  # Segundos máximos para execução de código
# FIREJAIL_PROFILE = os.path.join(os.path.dirname(__file__), 'python.profile') # Path relative to config.py - REMOVED/COMMENTED - REMOVED

# Configurações do Agente ReAct
MAX_REACT_ITERATIONS = int(os.getenv("MAX_REACT_ITERATIONS", "10"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))  # Keep one definition
MAX_META_DEPTH = (
    3  # Profundidade máxima para chamadas recursivas de auto-correção/reflexão
)

# Configurações de Histórico (se aplicável fora do agente) - REMOVED DUPLICATE
# MAX_HISTORY_TURNS = 5 # Keep one definition

# Outras configurações globais podem ser adicionadas aqui
# Ex: Nível de log, caminhos de diretório padrão, etc.

# Configuração do histórico - REMOVED DUPLICATE COMMENT
# MAX_HISTORY_TURNS = 5  # Quantos pares (usuário + assistente) lembrar - REMOVED DUPLICATE

# Configurações do Servidor LLM (Llama.cpp)
LLAMA_API_KEY = os.getenv(
    "LLAMA_API_KEY", "nokey"
)  # 'nokey' ou sua chave se o servidor exigir
LLAMA_DEFAULT_HEADERS = {"Content-Type": "application/json"}  # Keep first definition
# Adicionar Authorization se LLAMA_API_KEY for definida
if LLAMA_API_KEY and LLAMA_API_KEY.lower() not in ["none", "nokey", ""]:
    LLAMA_DEFAULT_HEADERS["Authorization"] = f"Bearer {LLAMA_API_KEY}"

# Configurações do Banco de Dados (SQLite)
# DATABASE_PATH = os.getenv("DATABASE_PATH", "a3x_memory.db") # Deprecated or unused? - REMOVED
MEMORY_DB_PATH = os.getenv(
    "MEMORY_DB_PATH", os.path.join(PROJECT_ROOT, "a3x", "memory.db") # <<< Corrigido para apontar para dentro de a3x/ >>>
)

# Configurações de Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # Keep first definition
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

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
AGENT_STATE_ID = os.getenv(
    "AGENT_STATE_ID", "1"
)  # Default agent state ID - Kept this definition assuming it's used
# MAX_REACT_ITERATIONS = int(os.getenv('MAX_REACT_ITERATIONS', '10')) - REMOVED DUPLICATE
# ... (rest of the file)

# --- Configurações do Servidor LLM (llama.cpp) ---
# Caminho para o executável do servidor llama.cpp (se não estiver no PATH)
# LLAMA_SERVER_EXECUTABLE = os.environ.get("LLAMA_SERVER_EXECUTABLE", os.path.join(PROJECT_ROOT, "llama.cpp/build/bin/llama-server")) # Corrigido para apontar para o build <<< COMMENTED OUT - Using main binary directly - REMOVED
# Caminho para o arquivo do modelo GGUF
LLAMA_MODEL_PATH = os.environ.get(
    "LLAMA_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "gemma-3-4b-it-Q4_K_M.gguf"), # Correct full path using project_root
)
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

# --- Configurações de Treinamento QLoRA ---
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "google/gemma-2b")
QLORA_R = int(os.getenv("QLORA_R", "8"))
QLORA_ALPHA = int(os.getenv("QLORA_ALPHA", "16"))
QLORA_DROPOUT = float(os.getenv("QLORA_DROPOUT", "0.05"))
TRAINING_OUTPUT_DIR = os.getenv("TRAINING_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "a3x_training_output/qlora_adapters"))
TRAINING_BATCH_SIZE = int(os.getenv("TRAINING_BATCH_SIZE", "1"))
TRAINING_GRAD_ACCUMULATION = int(os.getenv("TRAINING_GRAD_ACCUMULATION", "4"))
TRAINING_EPOCHS = int(os.getenv("TRAINING_EPOCHS", "1"))
TRAINING_LEARNING_RATE = float(os.getenv("TRAINING_LEARNING_RATE", "2e-4"))
# Outras configurações de BnB/LoRA podem ser adicionadas aqui se necessário
# Ex: BNB_COMPUTE_DTYPE = os.getenv("BNB_COMPUTE_DTYPE", "bfloat16")
# Ex: LORA_TARGET_MODULES = os.getenv("LORA_TARGET_MODULES", '["q_proj","v_proj"]') # Ler como string e parsear

# --- FIM Configurações QLoRA ---

# Garantir que o diretório de output do treino exista
try:
    os.makedirs(TRAINING_OUTPUT_DIR, exist_ok=True)
except OSError as e:
    print(f"Erro ao criar diretório de output do treinamento {TRAINING_OUTPUT_DIR}: {e}")

# --- LLM Server (llama.cpp) Configuration ---
LLAMA_CPP_DIR = str(PROJECT_ROOT / "llama.cpp")
# Correct path: Directly join PROJECT_ROOT with the relative path to the binary
LLAMA_SERVER_BINARY_NAME = "llama-server"
LLAMA_SERVER_BINARY = str(PROJECT_ROOT / "llama.cpp/build/bin" / LLAMA_SERVER_BINARY_NAME) if platform.system() != "Windows" else str(PROJECT_ROOT / "llama.cpp/build/bin/Release" / f"{LLAMA_SERVER_BINARY_NAME}.exe")
LLAMA_SERVER_MODEL_DIR = str(PROJECT_ROOT / "models")
LLAMA_SERVER_MODEL_NAME = "google_gemma-3-4b-it-Q4_K_S.gguf" # Corrected model name
LLAMA_SERVER_MODEL_PATH = os.getenv('LLAMA_SERVER_MODEL_PATH', str(Path(LLAMA_SERVER_MODEL_DIR) / LLAMA_SERVER_MODEL_NAME))
LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_PORT = 8080
LLAMA_SERVER_URL_BASE = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
LLAMA_HEALTH_ENDPOINT = f"{LLAMA_SERVER_URL_BASE}/health"
LLAMA_CHAT_ENDPOINT = f"{LLAMA_SERVER_URL_BASE}/v1/chat/completions"
LLAMA_SERVER_ARGS = [
    "-m", LLAMA_SERVER_MODEL_PATH,
    "-c", "4096", # <<< AJUSTE O TAMANHO DO CONTEXTO CONFORME NECESSÁRIO >>>
    "--port", str(LLAMA_SERVER_PORT),
    "--host", LLAMA_SERVER_HOST,
    # Adicione outros argumentos necessários aqui (ex: --ngl 35, --temp 0.7)
    # "--ngl", "35",
]
LLAMA_SERVER_STARTUP_TIMEOUT = 120 # seconds

# --- Stable Diffusion Server (a3x/servers/sd_api_server.py) Configuration ---
SD_SERVER_MODULE = "a3x.servers.sd_api_server"
# Correct path: Directly join PROJECT_ROOT with the relative path to the SD WebUI directory
SD_WEBUI_DEFAULT_PATH_CONFIG = os.getenv('SD_WEBUI_PATH', str(PROJECT_ROOT / "stable-diffusion-webui")) # Use PROJECT_ROOT
SD_API_HOST = "127.0.0.1"
SD_API_PORT = 7860
SD_API_URL_BASE = f"http://{SD_API_HOST}:{SD_API_PORT}"
SD_API_CHECK_ENDPOINT = f"{SD_API_URL_BASE}/docs" # Endpoint to check if API is ready
SD_SERVER_STARTUP_TIMEOUT = 180 # seconds, can take longer to start

# --- General Server Manager Config ---
SERVER_CHECK_INTERVAL = 5 # seconds
SERVER_LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "servers.log")

# Garantir que o diretório de logs exista
os.makedirs(os.path.dirname(SERVER_LOG_FILE), exist_ok=True)

# URL do servidor LLAMA (agora usa as constantes acima)
LLAMA_SERVER_URL = LLAMA_CHAT_ENDPOINT # Usado por llm_interface.py
