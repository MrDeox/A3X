# core/log_config.py
import logging
import sys
from .config import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT

# Configuração básica do logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    stream=sys.stdout # Envia logs para stdout por padrão
)

# Obtém o logger específico para o agente
agent_logger = logging.getLogger("ReactAgent")

# Você pode adicionar mais configurações aqui se necessário,
# como handlers para arquivos, etc.
