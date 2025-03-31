import os
import logging
from pathlib import Path

# Corrected absolute import using alias
from core.config import PROJECT_ROOT as WORKSPACE_ROOT

# Importar usando paths relativos
from core.tools import skill

# Initialize logger
logger = logging.getLogger(__name__)

def _is_path_within_workspace(path: str | Path) -> bool:
    """Verifica se o caminho fornecido está dentro do WORKSPACE_ROOT."""
    try:
        absolute_path = Path(path).resolve()
        return absolute_path.is_relative_to(WORKSPACE_ROOT)
    except ValueError: # Caso o path seja inválido
        return False
    except Exception as e: # Outros erros inesperados (ex: permissão)
        logger.error(f"Erro ao verificar caminho '{path}': {e}")
        return False

def _resolve_path(filepath: str) -> Path | None:
    """Resolve o filepath para um caminho absoluto seguro dentro do workspace."""
    path = Path(filepath)
    if path.is_absolute():
        # Se absoluto, já verifica se está no workspace
        if _is_path_within_workspace(path):
            return path.resolve()
        else:
            logger.warning(f"Acesso negado: Caminho absoluto fora do workspace: {path}")
            return None
    else:
        # Se relativo, junta com o WORKSPACE_ROOT
        resolved_path = (WORKSPACE_ROOT / path).resolve()
        # Verifica novamente se o caminho resolvido ainda está no workspace
        if _is_path_within_workspace(resolved_path):
            return resolved_path
        else:
            # Isso pode acontecer com caminhos relativos como "../../../etc/passwd"
            logger.warning(f"Acesso negado: Caminho relativo resolvido fora do workspace: {filepath} -> {resolved_path}")
            return None

def _create_file_logic(resolved_path: Path, filepath_original: str, content: str, overwrite: bool) -> dict:
    """Internal function to create/overwrite a file at a validated path."""
    try:
        # Check existence and type *again* just before writing (safety check)
        if resolved_path.exists():
            if resolved_path.is_dir():
                 return {"status": "error", "action": "create_file_failed", "data": {"message": f"Cannot create file, a directory already exists at '{filepath_original}'"}}
            if not overwrite:
                 # This condition should ideally be caught by the main skill's pre-check,
                 # but kept here as a final safeguard.
                return {"status": "error", "action": "create_file_failed", "data": {"message": f"File '{filepath_original}' already exists. Use overwrite=True to replace it."}}
            # If overwrite is True and it's a file, proceed.

        # Ensure parent directory exists
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(content)

        action_name = "file_overwritten" if overwrite and resolved_path.exists() else "file_created"
        message = f"File '{filepath_original}' was successfully {'overwritten' if action_name == 'file_overwritten' else 'created'}."
        logger.info(message)
        return {"status": "success", "action": action_name, "data": {"message": message, "filepath": filepath_original}}

    except PermissionError:
        logger.error(f"Permission error creating/overwriting file: {resolved_path}", exc_info=True)
        return {"status": "error", "action": "create_file_failed", "data": {"message": f"Permission denied to create/overwrite file: '{filepath_original}'"}}
    except IsADirectoryError:
         # Should be caught above, but handle defensively
         return {"status": "error", "action": "create_file_failed", "data": {"message": f"Cannot create file, path is a directory: '{filepath_original}'"}}
    except Exception as e:
        logger.exception(f"Unexpected error creating/overwriting file '{resolved_path}':")
        return {"status": "error", "action": "create_file_failed", "data": {"message": f"Unexpected error managing file '{filepath_original}': {e}"}}


def _append_to_file_logic(resolved_path: Path, filepath_original: str, content: str) -> dict:
    """Internal function to append content to a validated file path."""
    # Existence and type checks are done by the calling skill function
    try:
        # Ensure parent directory exists (safety check, though file existence implies parent exists)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        with open(resolved_path, "a", encoding="utf-8") as f:
            # Ensure content ends with a newline for better formatting when appending multiple times
            if not content.endswith('\n'):
                content += '\n'
            f.write(content)

        message = f"Content successfully appended to file '{filepath_original}'."
        logger.info(message)
        return {"status": "success", "action": "file_appended", "data": {"message": message, "filepath": filepath_original}}

    except PermissionError:
        logger.error(f"Permission error appending to file: {resolved_path}", exc_info=True)
        return {"status": "error", "action": "append_failed", "data": {"message": f"Permission denied to append to file: '{filepath_original}'"}}
    except IsADirectoryError:
         # Should be caught by caller, but handle defensively
         return {"status": "error", "action": "append_failed", "data": {"message": f"Cannot append to a directory: '{filepath_original}'"}}
    except Exception as e:
        logger.exception(f"Unexpected error appending to file '{resolved_path}':")
        return {"status": "error", "action": "append_failed", "data": {"message": f"Unexpected error appending to file '{filepath_original}': {e}"}}


# --- Skill: Create or Overwrite File ---
@skill(
    name="create_file",
    description="Creates a new text file or completely overwrites an existing one at a specified path within the workspace.",
    parameters={
        "filepath": (str, ...),
        "content": (str, ...),
        "overwrite": (bool, False),
        "agent_history": (list | None, None)
    }
)
def create_file(filepath: str, content: str, overwrite: bool = False, agent_memory: dict | None = None, agent_history: list | None = None) -> dict:
    """
    Creates a new file or overwrites an existing file with the provided content.

    Args:
        filepath (str): The relative path within the workspace where the file should be created/overwritten.
        content (str): The text content to write to the file.
        overwrite (bool, optional): If True, overwrites the file if it already exists.
                                    If False (default), returns an error if the file exists.
        agent_memory (dict, optional): Agent's memory (not used). Defaults to None.
        agent_history (list | None, optional): Conversation history (not used). Defaults to None.

    Returns:
        dict: Standardized dictionary with the result.
    """
    logger.debug(f"Skill 'create_file' requested for: '{filepath}', overwrite={overwrite}")

    # --- Path Validation ---
    if not isinstance(filepath, str) or not filepath:
        return {"status": "error", "action": "create_file_failed", "data": {"message": "Filepath parameter cannot be empty."}}
    if not isinstance(content, str): # Content can be empty string, but must be string
         return {"status": "error", "action": "create_file_failed", "data": {"message": "Content parameter must be a string."}}
    if not isinstance(overwrite, bool):
        return {"status": "error", "action": "create_file_failed", "data": {"message": "Overwrite parameter must be a boolean (true/false)."}}


    try:
        workspace_resolved_path = Path(WORKSPACE_ROOT).resolve()
        # Check both string and Path object decomposition for '..'
        if os.path.isabs(filepath) or ".." in filepath or ".." in Path(filepath).parts:
             raise ValueError("Invalid path: Absolute paths or paths with '..' are not allowed. Use relative paths within the workspace.")

        abs_path = workspace_resolved_path / filepath
        resolved_path = abs_path.resolve()

        if not str(resolved_path).startswith(str(workspace_resolved_path)):
             raise ValueError("Invalid path: Path resolves outside the workspace boundaries.")

        # Pre-flight check for existence and type (before calling logic)
        if resolved_path.exists():
            if resolved_path.is_dir():
                 return {"status": "error", "action": "create_file_failed", "data": {"message": f"Cannot create file, a directory already exists at '{filepath}'"}}
            if not overwrite:
                return {"status": "error", "action": "create_file_failed", "data": {"message": f"File '{filepath}' already exists. Use overwrite=True to replace it."}}
            # If it exists, is a file, and overwrite is True, proceed.

    except ValueError as e:
        logger.warning(f"Path validation failed for '{filepath}': {e}")
        return {"status": "error", "action": "create_file_failed", "data": {"message": f"Path validation failed: {e}"}}
    except Exception as e:
        logger.error(f"Unexpected error resolving path '{filepath}': {e}", exc_info=True)
        return {"status": "error", "action": "create_file_failed", "data": {"message": f"Internal error processing path: {e}"}}
    # --- End Path Validation ---

    logger.debug(f"Path validated for creation: '{filepath}' -> '{resolved_path}'")
    # Call the core logic function with the validated path
    return _create_file_logic(resolved_path, filepath, content, overwrite)


# --- Skill: Append to File ---
@skill(
    name="append_to_file",
    description="Appends text content to the end of an existing file within the workspace.",
    parameters={
        "filepath": (str, ...),
        "content": (str, ...),
        "agent_history": (list | None, None)
    }
)
def append_to_file(filepath: str, content: str, agent_memory: dict | None = None, agent_history: list | None = None) -> dict:
    """
    Appends content to the end of an existing file.

    Args:
        filepath (str): The relative path within the workspace to the file to append to.
        content (str): The text content to append.
        agent_memory (dict, optional): Agent's memory (not used). Defaults to None.
        agent_history (list | None, optional): Conversation history (not used). Defaults to None.

    Returns:
        dict: Standardized dictionary with the result.
    """
    logger.debug(f"Skill 'append_to_file' requested for: '{filepath}'")

    # --- Path Validation ---
    if not isinstance(filepath, str) or not filepath:
        return {"status": "error", "action": "append_failed", "data": {"message": "Filepath parameter cannot be empty."}}
    if not isinstance(content, str):
         return {"status": "error", "action": "append_failed", "data": {"message": "Content parameter must be a string."}}

    try:
        workspace_resolved_path = Path(WORKSPACE_ROOT).resolve()
        # Check both string and Path object decomposition for '..'
        if os.path.isabs(filepath) or ".." in filepath or ".." in Path(filepath).parts:
             raise ValueError("Invalid path: Absolute paths or paths with '..' are not allowed. Use relative paths within the workspace.")

        abs_path = workspace_resolved_path / filepath
        resolved_path = abs_path.resolve()

        if not str(resolved_path).startswith(str(workspace_resolved_path)):
             raise ValueError("Invalid path: Path resolves outside the workspace boundaries.")

        # Pre-flight check for existence and type (MUST exist and be a file for append)
        if not resolved_path.exists():
             return {"status": "error", "action": "append_failed", "data": {"message": f"File not found: Cannot append to non-existent file '{filepath}'."}}
        if not resolved_path.is_file():
            return {"status": "error", "action": "append_failed", "data": {"message": f"Path is not a file: Cannot append to '{filepath}'."}}

    except ValueError as e:
        logger.warning(f"Path validation failed for '{filepath}': {e}")
        return {"status": "error", "action": "append_failed", "data": {"message": f"Path validation failed: {e}"}}
    except Exception as e:
        logger.error(f"Unexpected error resolving path '{filepath}': {e}", exc_info=True)
        return {"status": "error", "action": "append_failed", "data": {"message": f"Internal error processing path: {e}"}}
    # --- End Path Validation ---

    logger.debug(f"Path validated for append: '{filepath}' -> '{resolved_path}'")
    # Call the core logic function with the validated path
    return _append_to_file_logic(resolved_path, filepath, content)

# Remove the old combined function if it exists
# try:
#     del skill_manage_files
# except NameError:
#     pass 