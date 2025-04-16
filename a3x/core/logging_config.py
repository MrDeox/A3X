import logging
import os
from typing import Optional
from pathlib import Path

# from core.config import LOG_LEVEL  # Import central config
from a3x.core.config import SERVER_LOG_FILE, LOG_FORMAT, LOG_DATE_FORMAT # Import SERVER_LOG_FILE

# LOG_FORMAT = "[%(levelname)s %(name)s] %(message)s" # Use format from config

# <<< REMOVE: redundant setup_logging definition >>>
# def setup_logging():
#     ...

# <<< UPDATED: Function signature to accept arguments >>>
def setup_logging(log_level_str: str = "INFO", log_file_path: Optional[str] = None):
    """Configura o logging para o projeto, usando os níveis e arquivos fornecidos."""

    # <<< Use provided log_level_str >>>
    log_level_upper = log_level_str.upper()
    numeric_level = getattr(logging, log_level_upper, logging.INFO)
    if not isinstance(numeric_level, int):
        print(
            f"[Logging Setup Warning] Invalid log level '{log_level_str}'. Defaulting to INFO."
        )
        numeric_level = logging.INFO

    # --- Configure Root Logger (Console and General File) ---
    root_logger = logging.getLogger()
    # Try removing existing handlers FIRST
    for handler in root_logger.handlers[:]:
        try:
            handler.close() # Close before removing
            root_logger.removeHandler(handler)
        except Exception as e:
             # Log potential error during handler removal (might not be visible)
             print(f"[Logging Setup Error] Failed to remove/close handler: {e}")

    # Basic config sets up console handler by default
    # <<< Wrap basicConfig >>>
    try:
        logging.basicConfig(level=numeric_level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        root_logger.setLevel(numeric_level)
    except Exception as e:
        print(f"[Logging Setup CRITICAL] logging.basicConfig failed: {e}")
        import sys
        sys.exit(f"CRITICAL Error during basicConfig: {e}")

    # --- General File Handler ---
    if log_file_path is None:
        log_file_path = Path(os.getcwd()) / "logs" / "a3x_cli.log"
    else:
        log_file_path = Path(log_file_path)

    # <<< Wrap mkdir >>>
    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[Logging Setup CRITICAL] Failed to create log directory {log_file_path.parent}: {e}")
        import sys
        sys.exit(f"CRITICAL Error creating log directory: {e}")

    # <<< Wrap FileHandler >>>
    try:
        general_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        general_file_handler = logging.FileHandler(log_file_path, mode='a')
        general_file_handler.setFormatter(general_formatter)
        general_file_handler.setLevel(numeric_level)
        root_logger.addHandler(general_file_handler)
    except Exception as e:
        print(f"[Logging Setup CRITICAL] Failed to create/add general file handler for {log_file_path}: {e}")
        import sys
        sys.exit(f"CRITICAL Error creating file handler: {e}")

    # --- Specific Server Log File Handler ---
    server_log_path = Path(SERVER_LOG_FILE)
    # <<< Wrap mkdir >>>
    try:
        server_log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[Logging Setup CRITICAL] Failed to create server log directory {server_log_path.parent}: {e}")
        import sys
        sys.exit(f"CRITICAL Error creating server log directory: {e}")

    # <<< Wrap FileHandler >>>
    try:
        server_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        server_handler = logging.FileHandler(SERVER_LOG_FILE, mode='a')
        server_handler.setFormatter(server_formatter)
        server_handler.setLevel(logging.INFO) # Or numeric_level
        root_logger.addHandler(server_handler)
    except Exception as e:
        print(f"[Logging Setup CRITICAL] Failed to create/add server file handler for {SERVER_LOG_FILE}: {e}")
        import sys
        sys.exit(f"CRITICAL Error creating server file handler: {e}")

    # If we reach here, configuration *should* have worked.
    logging.info(f"Logging configured. Level: {log_level_upper}. General Log: {log_file_path}. Server Log: {SERVER_LOG_FILE}")

# <<< Call setup_logging() directly upon import? Or rely on CLI calling it? >>>
# Let's assume CLI calls it early on.
# setup_logging() # Remove automatic call here
