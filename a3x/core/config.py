import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis do .env se existir
dotenv_path = Path(__file__).parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# Diretórios principais
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MEMORY_DIR = PROJECT_ROOT / "memory"
LOGS_DIR = PROJECT_ROOT / "logs"
LEARNING_LOG_DIR = MEMORY_DIR / "learning_logs"

# Arquivos de log/heurísticas
HEURISTIC_LOG_FILE = "learned_heuristics.jsonl"

# Limites e parâmetros
MAX_REACT_ITERATIONS = int(os.getenv("MAX_REACT_ITERATIONS", 10))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 5))
MAX_TOKENS_FALLBACK = int(os.getenv("MAX_TOKENS_FALLBACK", 4096))
SEMANTIC_SEARCH_TOP_K = int(os.getenv("SEMANTIC_SEARCH_TOP_K", 5))
EPISODIC_RETRIEVAL_LIMIT = int(os.getenv("EPISODIC_RETRIEVAL_LIMIT", 10))

# Modelos
LLAMA_SERVER_MODEL_PATH = os.getenv("LLAMA_SERVER_MODEL_PATH", "models/google_gemma-3-4b-it-Q4_K_S.gguf")

# Outras configurações podem ser adicionadas aqui conforme necessário
