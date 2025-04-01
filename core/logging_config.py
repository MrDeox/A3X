import logging
from core.config import LOG_LEVEL  # Import central config

LOG_FORMAT = "[%(levelname)s %(name)s] %(message)s"


def setup_logging():
    """Configura o logging básico para o projeto."""
    # Tentativa de obter o nível de log do enum logging se for uma string
    numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)  # Default INFO
    if not isinstance(numeric_level, int):
        print(
            f"[Logging Setup Warning] Invalid LOG_LEVEL '{LOG_LEVEL}'. Defaulting to INFO."
        )
        numeric_level = logging.INFO

    logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
    logging.getLogger().setLevel(numeric_level)  # Ensure root logger level is set
    # Optional: Silence logs from noisy libraries
    # logging.getLogger("requests").setLevel(logging.WARNING)
    # logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.info(f"Logging configured with level: {LOG_LEVEL} ({numeric_level})")
