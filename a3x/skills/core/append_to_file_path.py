"""
Skill para anexar texto a um arquivo especificado por path.
"""

import logging
import os
from typing import Dict, Any

# Core imports
from a3x.core.tools import skill

# Assume SkillContext provides logger

logger = logging.getLogger(__name__)

@skill(
    name="append_to_file_path",
    description="Anexa uma linha de texto a um arquivo especificado, criando diretórios se necessário.",
    parameters={
        "path": (str, ...), # Caminho completo ou relativo para o arquivo.
        "text": (str, ...)  # Texto a ser anexado (uma nova linha será adicionada automaticamente).
    }
)
async def append_to_file_path(ctx, path: str, text: str) -> Dict[str, str]:
    """
    Appends a line of text to a specified file, creating parent directories if needed.

    Args:
        ctx: The skill execution context (provides logger).
        path: The absolute or relative path to the target file.
        text: The text content to append. A newline will be added.

    Returns:
        A dictionary with {"status": "ok"} on success or {"error": ...} on failure.
    """
    logger.info(f"Attempting to append to file: {path}")
    try:
        # Ensure parent directory exists
        dir_path = os.path.dirname(path)
        if dir_path: # Only create if path includes a directory
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")

        # Append text to the file
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

        logger.info(f"Successfully appended text to {path}")
        return {"status": "ok"}

    except OSError as e:
        logger.exception(f"OS error appending to file {path}: {e}")
        return {"error": f"OS error writing to file {path}: {e}"}
    except Exception as e:
        logger.exception(f"Unexpected error appending to file {path}: {e}")
        return {"error": f"Unexpected error writing to file {path}: {e}"} 