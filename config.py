"""
Configurações do sistema A³X.
"""

# Dimensão dos embeddings gerados pelo modelo all-MiniLM-L6-v2
EMBEDDING_DIM = 384

# Configurações de logging
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# Configurações de memória
MEMORY_FILE = 'data/memory.json'
MAX_MEMORY_ITEMS = 1000

# Configurações de execução
MAX_CODE_LENGTH = 1000
TIMEOUT = 5  # segundos 