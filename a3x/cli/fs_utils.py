# a3x/cli/fs_utils.py
import os
import sys
import logging
from pathlib import Path

from rich.console import Console

# Import config needed for PROJECT_ROOT
try:
    from a3x.core.config import PROJECT_ROOT
except ImportError as e:
    print(f"[CLI FS Utils Error] Failed to import PROJECT_ROOT from config: {e}")
    # Fallback: Assume script is run from within a3x/cli
    PROJECT_ROOT = Path(__file__).resolve().parents[2] # Sobe dois níveis de a3x/cli/fs_utils.py

logger = logging.getLogger(__name__)
console = Console()

def change_to_project_root():
    """Changes the current working directory to the project root."""
    if not PROJECT_ROOT:
        logger.error("PROJECT_ROOT not configured. Cannot change directory.")
        return

    try:
        absolute_project_root = os.path.abspath(PROJECT_ROOT)
        if os.getcwd() != absolute_project_root:
            os.chdir(absolute_project_root)
            logger.info(f"Changed working directory to: {absolute_project_root}")
            # Add project root to sys.path AFTER changing directory
            if absolute_project_root not in sys.path:
                sys.path.insert(0, absolute_project_root)
                logger.debug(f"Added {absolute_project_root} to sys.path")
        else:
            logger.debug(f"Already in project root: {absolute_project_root}")
    except FileNotFoundError:
        logger.error(f"Project root directory not found: {PROJECT_ROOT}")
        # Consider raising an exception here instead of just logging
    except Exception as e:
        logger.exception(f"Error changing directory to project root {PROJECT_ROOT}:")
        # Consider raising an exception

def load_system_prompt(file_path_relative: str = "data/prompts/react_system_prompt.md") -> str:
    """Carrega o conteúdo de um arquivo de prompt do sistema."""
    full_path = PROJECT_ROOT / file_path_relative
    logger.debug(f"Attempting to load system prompt from: {full_path}")
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"System prompt file not found: {full_path}")
        console.print(
            f"[bold red][Error][/] System prompt file not found at '{full_path}'. Using a minimal fallback."
        )
        return "You are a helpful assistant."
    except Exception as e:
        logger.exception(f"Error reading system prompt file {full_path}:")
        console.print(
            f"[bold red][Error][/] Could not read system prompt file: {e}. Using minimal fallback."
        )
        return "You are a helpful assistant." 