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
    # Remove existing handlers to avoid duplication if setup is called again
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Basic config sets up console handler by default
    logging.basicConfig(level=numeric_level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    root_logger.setLevel(numeric_level) # Ensure root logger level is set

    # --- General File Handler (Using provided path or default) ---
    if log_file_path is None:
        log_file_path = Path(os.getcwd()) / "logs" / "a3x_cli.log"
    else:
        log_file_path = Path(log_file_path)

    # Create log directory if it doesn't exist
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create and add the general file handler
    general_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    general_file_handler = logging.FileHandler(log_file_path, mode='a') # Append mode
    general_file_handler.setFormatter(general_formatter)
    general_file_handler.setLevel(numeric_level) # Use the same level as console
    root_logger.addHandler(general_file_handler)

    # --- Specific Server Log File Handler (Keep as is) ---
    # Configure file handler specifically for server logs
    # Ensure server log directory exists too
    server_log_path = Path(SERVER_LOG_FILE)
    server_log_path.parent.mkdir(parents=True, exist_ok=True)

    server_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    server_handler = logging.FileHandler(SERVER_LOG_FILE, mode='a') # Append mode
    server_handler.setFormatter(server_formatter)
    # Server log level can be independent if needed, keeping INFO for now
    server_handler.setLevel(logging.INFO)

    # Assign this handler to a specific logger (or root if necessary)
    # Let's keep it separate by assigning to 'A3XServerManager' or similar if used,
    # or just add to root for now if server components log directly.
    # For simplicity/capture-all, adding to root. Can refine later.
    root_logger.addHandler(server_handler)

    logging.info(f"Logging configured. Level: {log_level_upper}. General Log: {log_file_path}. Server Log: {SERVER_LOG_FILE}")

# <<< Call setup_logging() directly upon import? Or rely on CLI calling it? >>>
# Let's assume CLI calls it early on.
# setup_logging() # Remove automatic call here
