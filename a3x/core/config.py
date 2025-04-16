import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis do .env se existir
dotenv_path = Path(__file__).parent.parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# Diretórios principais
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
MEMORY_DIR = PROJECT_ROOT / "a3x" / "memory"
LEARNING_LOGS_DIR = MEMORY_DIR / "learning_logs"
LLM_LOGS_DIR = MEMORY_DIR / "llm_logs"
LEARNING_HISTORY_DIR = MEMORY_DIR / "learning_history"
SELF_EVAL_LOG = MEMORY_DIR / "self_evaluation_log.jsonl"
LOGS_DIR = PROJECT_ROOT / "logs"
PROMPT_DIR = PROJECT_ROOT / "a3x" / "prompts"

# Arquivos de log/heurísticas
SERVER_LOG_FILE = LOGS_DIR / "servers.log"
HEURISTIC_LOG_FILE = LEARNING_LOGS_DIR / "learned_heuristics.jsonl"
HEURISTIC_LOG_CONSOLIDATED_FILE = LEARNING_LOGS_DIR / "learned_heuristics_consolidated.jsonl"
HEURISTICS_VALIDATION_LOG = LEARNING_LOGS_DIR / "heuristics_validation.jsonl"
AUTO_EVAL_LOG = LEARNING_LOGS_DIR / "auto_evaluation.jsonl"
BATERIA_AUTOAVALIACAO_LOG = LEARNING_LOGS_DIR / "bateria_autoavaliacao.log"
LLM_DECISION_REFLECTIONS_LOG = LLM_LOGS_DIR / "decision_reflections.jsonl"
AUTO_PROMPT_EVOLUTION_LOG = LEARNING_HISTORY_DIR / "auto_prompt_evolution.jsonl"

# <<< ADDED Logging Format >>>
LOG_FORMAT = os.getenv("LOG_FORMAT", "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
LOG_DATE_FORMAT = os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")

# Limites e parâmetros
MAX_REACT_ITERATIONS = int(os.getenv("MAX_REACT_ITERATIONS", 10))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 5))
MAX_TOKENS_FALLBACK = int(os.getenv("MAX_TOKENS_FALLBACK", 4096))
SEMANTIC_SEARCH_TOP_K = int(os.getenv("SEMANTIC_SEARCH_TOP_K", 5))
EPISODIC_RETRIEVAL_LIMIT = int(os.getenv("EPISODIC_RETRIEVAL_LIMIT", 10))

# Modelos
LLAMA_SERVER_MODEL_PATH = os.getenv("LLAMA_SERVER_MODEL_PATH", "models/google_gemma-3-4b-it-Q4_K_S.gguf")

# <<< ADDED Base model name for training >>>
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", LLAMA_SERVER_MODEL_PATH) # Use the server model as default

# Configuração da API Gumroad
GUMROAD_API_BASE_URL = os.getenv("GUMROAD_API_BASE_URL", "https://api.gumroad.com/v2/")
GUMROAD_API_KEY = os.getenv("GUMROAD_API_KEY", "")

# Caminho do índice FAISS para memória semântica
SEMANTIC_INDEX_PATH = str(MEMORY_DIR / "indexes" / "semantic_memory")

# Caminho do banco de dados principal
DATABASE_PATH = str(MEMORY_DIR / "memory.db")

# Diretório para armazenamento multimodal (screenshots, etc.)
MULTIMODAL_STORAGE_DIR = MEMORY_DIR / "multimodal"

# <<< ADDED Server Binaries/Modules >>>
# <<< CORRECTED Default Path >>>
LLAMA_SERVER_BINARY = os.getenv("LLAMA_SERVER_BINARY", str(PROJECT_ROOT / "llama.cpp" / "build" / "bin" / "llama-server"))
SD_SERVER_MODULE = os.getenv("SD_SERVER_MODULE", "a3x.servers.sd_api_server")
# TODO: Adicionar mais configs de servidor (paths, args) aqui
LLAMA_CPP_DIR = os.getenv("LLAMA_CPP_DIR", str(PROJECT_ROOT / "llama.cpp"))
# <<< CORRECTED: Use absolute path for default model in args >>>
default_model_path = str(PROJECT_ROOT / "models" / "google_gemma-3-4b-it-Q4_K_S.gguf")
LLAMA_SERVER_ARGS = os.getenv(
    "LLAMA_SERVER_ARGS",
    f"-m {default_model_path} -c 4096 -ngl 24 --host 0.0.0.0 --port 8080 --log-disable" # Use f-string for absolute path, increased default context
).split()
LLAMA_HEALTH_ENDPOINT = os.getenv("LLAMA_HEALTH_ENDPOINT", "http://127.0.0.1:8080/health")
# <<< ADDED: Explicit server URL for API calls >>>
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
LLAMA_SERVER_STARTUP_TIMEOUT = int(os.getenv("LLAMA_SERVER_STARTUP_TIMEOUT", 240))
SD_API_CHECK_ENDPOINT = os.getenv("SD_API_CHECK_ENDPOINT", "http://127.0.0.1:7861/sdapi/v1/progress")
SD_SERVER_STARTUP_TIMEOUT = int(os.getenv("SD_SERVER_STARTUP_TIMEOUT", 180))
SD_WEBUI_DEFAULT_PATH_CONFIG = os.getenv("SD_WEBUI_DEFAULT_PATH_CONFIG", str(PROJECT_ROOT / "stable-diffusion-webui"))
SERVER_CHECK_INTERVAL = int(os.getenv("SERVER_CHECK_INTERVAL", 2))

# <<< ADDED Training Configs >>>
# QLoRA parameters
QLORA_R = int(os.getenv("QLORA_R", 8))
QLORA_ALPHA = int(os.getenv("QLORA_ALPHA", 16))
QLORA_DROPOUT = float(os.getenv("QLORA_DROPOUT", 0.05))

# Training hyperparameters
TRAINING_OUTPUT_DIR = os.getenv("TRAINING_OUTPUT_DIR", str(PROJECT_ROOT / "a3x_training_output" / "qlora_adapters"))
TRAINING_BATCH_SIZE = int(os.getenv("TRAINING_BATCH_SIZE", 1))
TRAINING_GRAD_ACCUMULATION = int(os.getenv("TRAINING_GRAD_ACCUMULATION", 4))
TRAINING_EPOCHS = int(os.getenv("TRAINING_EPOCHS", 1)) # Default to 1 epoch for quick test
TRAINING_LEARNING_RATE = float(os.getenv("TRAINING_LEARNING_RATE", 2e-5))

# Outras configurações podem ser adicionadas aqui conforme necessário

# <<< ADDED: Define where skill packages are located >>>
# List of Python package paths where skills are defined (e.g., 'a3x.skills.core', 'a3x.skills.web')
# SKILL_PACKAGES = os.getenv("SKILL_PACKAGES", '["a3x.skills"]').split(',') # Original line with potential parsing issue

# <<< CORRIGIDO: Parsing correto para SKILL_PACKAGES >>>
skill_packages_env = os.getenv("SKILL_PACKAGES")
if skill_packages_env:
    try:
        # Tenta carregar como JSON se for uma lista no .env (ex: '["pkg1", "pkg2"]')
        import json
        SKILL_PACKAGES = json.loads(skill_packages_env)
        if not isinstance(SKILL_PACKAGES, list):
            raise ValueError("SKILL_PACKAGES env var must be a JSON list of strings.")
    except (json.JSONDecodeError, ValueError):
        # Se não for JSON válido ou não for lista, trata como string separada por vírgula
        SKILL_PACKAGES = [pkg.strip() for pkg in skill_packages_env.split(',') if pkg.strip()]
else:
    # Default é uma lista Python
    SKILL_PACKAGES = ["a3x.skills"]
