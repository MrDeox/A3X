import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# URL do servidor LLAMA (Ollama, LM Studio, etc.)
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1/chat/completions") # Default if not set

# Chave da API Tavily (se usar busca web com Tavily)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)

# Configurações de Execução de Código
PYTHON_EXEC_TIMEOUT = 10 # Segundos máximos para execução de código
# FIREJAIL_PROFILE = os.path.join(os.path.dirname(__file__), 'python.profile') # Path relative to config.py - REMOVED/COMMENTED

# Configurações do Agente ReAct
MAX_REACT_ITERATIONS = 7 # Aumentado um pouco

# Configurações de Histórico (se aplicável fora do agente)
MAX_HISTORY_TURNS = 5 # Keep one definition

# Outras configurações globais podem ser adicionadas aqui
# Ex: Nível de log, caminhos de diretório padrão, etc.

# Configuração do histórico
# MAX_HISTORY_TURNS = 5  # Quantos pares (usuário + assistente) lembrar - REMOVED DUPLICATE 