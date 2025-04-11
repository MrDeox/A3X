import logging
import os # Add os import

# from core.config import LOG_LEVEL  # Import central config
from a3x.core.config import LOG_LEVEL, SERVER_LOG_FILE, LOG_FORMAT, LOG_DATE_FORMAT # Import SERVER_LOG_FILE

# LOG_FORMAT = "[%(levelname)s %(name)s] %(message)s" # Use format from config

# <<< REMOVE: redundant setup_logging definition >>>
# def setup_logging():
#     ...

# <<< NEW: Function to setup logging, including server log file >>>
def setup_logging():
    """Configura o logging para o projeto, incluindo um arquivo para logs de servidor."""
    numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    if not isinstance(numeric_level, int):
        print(
            f"[Logging Setup Warning] Invalid LOG_LEVEL '{LOG_LEVEL}'. Defaulting to INFO."
        )
        numeric_level = logging.INFO

    # Configure root logger for console output
    logging.basicConfig(level=numeric_level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    logging.getLogger().setLevel(numeric_level)

    # Configure file handler specifically for server logs
    server_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    server_handler = logging.FileHandler(SERVER_LOG_FILE, mode='a') # Append mode
    server_handler.setFormatter(server_formatter)
    server_handler.setLevel(logging.INFO) # Log INFO level and above for servers

    # Get specific loggers or a general server logger
    # Option 1: Log specific server components
    # logging.getLogger("a3x.core.server_manager").addHandler(server_handler)
    # logging.getLogger("a3x.servers.sd_api_server").addHandler(server_handler)
    # Option 2: Create a dedicated server logger and add handler to it
    # server_logger = logging.getLogger("A3XServerManager")
    # server_logger.addHandler(server_handler)
    # server_logger.setLevel(logging.INFO)
    # server_logger.propagate = False # Prevent duplication to root logger if desired
    
    # Option 3: Add handler to the root logger to catch all server logs (simplest)
    # Be mindful this might duplicate logs if sub-loggers also log to console.
    # For now, let's add to root to capture logs from subprocesses managed elsewhere.
    # logging.getLogger().addHandler(server_handler) # Let's rely on specific loggers for now

    logging.info(f"Logging configured with level: {LOG_LEVEL} ({numeric_level})")
    logging.info(f"Server logs will be written to: {SERVER_LOG_FILE}")

# <<< Call setup_logging() directly upon import? Or rely on CLI calling it? >>>
# Let's assume CLI calls it early on.
# setup_logging() # Remove automatic call here
