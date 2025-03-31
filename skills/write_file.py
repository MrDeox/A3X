# skills/write_file.py
import logging
from pathlib import Path
import traceback
import os
from typing import Dict, Any
# from core.validators import validate_path # REMOVED - assuming validation logic is inline
from core.tools import skill

# Corrected absolute import using alias
from core.config import PROJECT_ROOT as WORKSPACE_ROOT

# Initialize logger
logger = logging.getLogger(__name__)

# Define potentially sensitive or large file extensions to warn about (optional)
WARN_EXTENSIONS = {".json", ".csv", ".log", ".env"}
MAX_SIZE_WARN = 1 * 1024 * 1024  # 1MB warning threshold

@skill(
    name="write_file",
    description="Creates a new text file or overwrites an existing one at a specified path within the workspace. Alias for create_file.",
    parameters={
        "filepath": (str, ...),
        "content": (str, ...),
        "overwrite": (bool, False),
        "agent_history": (list | None, None)
    }
)
def write_file(filepath: str, content: str, overwrite: bool = False, agent_memory: dict | None = None, agent_history: list | None = None) -> dict:
    """
    Creates a new file or overwrites an existing file with the provided content.

    Args:
        filepath (str): The relative path within the workspace where the file should be written.
        content (str): The string content to write to the file.
        overwrite (bool, optional): Whether to overwrite the file if it exists. Defaults to False.
        agent_memory (dict | None, optional): Agent's memory (not used). Defaults to None.
        agent_history (list | None, optional): Conversation history (not used). Defaults to None.

    Returns:
        dict: Standardized dictionary with the result.
    """
    logger.debug(f"Skill 'write_file' requested for: '{filepath}' with content length: {len(content)}")

    # --- Path Validation ---
    if not isinstance(filepath, str) or not filepath:
        return {"status": "error", "action": "write_file_failed", "data": {"message": "Filepath parameter cannot be empty or invalid."}}
    if not isinstance(content, str):
         # Allow empty content, but log it
         logger.warning(f"Content for '{filepath}' is not a string (type: {type(content)}), writing empty string.")
         content = "" # Coerce to empty string if not string, or handle as error? Let's allow empty for now.

    try:
        workspace_resolved_path = Path(WORKSPACE_ROOT).resolve()
        # Prevent absolute paths or paths trying to escape the workspace
        if os.path.isabs(filepath) or ".." in filepath or ".." in Path(filepath).parts:
            raise ValueError("Invalid path: Absolute paths or paths with '..' are not allowed. Use relative paths within the workspace.")

        abs_path = workspace_resolved_path / filepath
        resolved_path = abs_path.resolve()

        # Final check: ensure the resolved path is strictly within the workspace
        if not str(resolved_path).startswith(str(workspace_resolved_path)):
            raise ValueError("Invalid path: Path resolves outside the workspace boundaries.")

        # Check if the target is a directory (we cannot overwrite a directory with a file)
        if resolved_path.is_dir():
            raise ValueError("Invalid path: Provided path points to an existing directory, cannot overwrite with a file.")

    except ValueError as e:
        logger.warning(f"Path validation failed for '{filepath}': {e}")
        return {"status": "error", "action": "write_file_failed_validation", "data": {"message": f"Path validation failed: {e}"}}
    except Exception as e:
        logger.error(f"Unexpected error resolving path '{filepath}': {e}", exc_info=True)
        return {"status": "error", "action": "write_file_failed_internal", "data": {"message": f"Internal error processing path: {e}"}}
    # --- End Path Validation ---

    logger.debug(f"Path validated for writing: '{filepath}' -> '{resolved_path}'")

    # --- Optional Checks ---
    file_ext = resolved_path.suffix.lower()
    if file_ext in WARN_EXTENSIONS:
        logger.warning(f"Writing to potentially sensitive file type: '{file_ext}' for path '{filepath}'")

    if len(content.encode('utf-8')) > MAX_SIZE_WARN:
        logger.warning(f"Writing large content ({len(content.encode('utf-8')) / (1024*1024):.2f} MB) to '{filepath}'")
    # --- End Optional Checks ---

    try:
        # Create parent directories if they don't exist
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured parent directory exists: {resolved_path.parent}")

        # Write the file (overwrite if exists)
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Successfully wrote content to: {resolved_path} (requested path: '{filepath}')")

        return {
            "status": "success",
            "action": "file_written",
            "data": {
                "message": f"Content successfully written to file '{filepath}'.",
                "filepath_written": filepath,
                "absolute_path": str(resolved_path), # Might be useful for debugging
                "content_length": len(content)
            }
        }

    except PermissionError:
        logger.error(f"Permission error trying to write to '{resolved_path}'", exc_info=True)
        return {"status": "error", "action": "write_file_failed_permission", "data": {"message": f"Permission denied to write file: '{filepath}'"}}
    except IsADirectoryError: # Should be caught by validation, but as a safeguard
         logger.error(f"Attempted to write to a path that is a directory: '{resolved_path}'", exc_info=True)
         return {"status": "error", "action": "write_file_failed_isDirectory", "data": {"message": f"Cannot write file to a path that is an existing directory: '{filepath}'"}}
    except Exception as e:
        logger.exception(f"Unexpected error trying to write to '{resolved_path}':")
        return {"status": "error", "action": "write_file_failed_unexpected", "data": {"message": f"Unexpected error writing file '{filepath}': {e}"}}

