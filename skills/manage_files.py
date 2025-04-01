import os
import logging
from pathlib import Path

# Corrected absolute import using alias
from core.config import PROJECT_ROOT as WORKSPACE_ROOT

# Importar usando paths relativos
from core.tools import skill
from core.validators import validate_workspace_path

# Initialize logger
logger = logging.getLogger(__name__)

def _create_file_logic(resolved_path: Path, filepath_original: str, content: str, overwrite: bool) -> dict:
    """Internal function to create/overwrite a file at a validated path."""
    try:
        # Check existence and type *again* just before writing (safety check)
        # The decorator handles initial existence check based on overwrite logic needs.
        if resolved_path.exists():
            if resolved_path.is_dir():
                 return {"status": "error", "action": "create_file_failed", "data": {"message": f"Cannot create file, a directory already exists at '{filepath_original}'"}}
            # Decorator won't check existence for create_file, so we check overwrite here.
            if not overwrite:
                return {"status": "error", "action": "create_file_failed", "data": {"message": f"File '{filepath_original}' already exists. Use overwrite=True to replace it."}}
            # If overwrite is True and it's a file, proceed.

        # Ensure parent directory exists
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Determine action name based on whether the file existed *before* this write operation
        # This is tricky without knowing pre-decorator state. Assume overwrite means it might have existed.
        action_name = "file_overwritten" if overwrite else "file_created"
        message = f"File '{filepath_original}' was successfully {'overwritten' if overwrite else 'created'}."
        logger.info(message)
        return {"status": "success", "action": action_name, "data": {"message": message, "filepath": filepath_original}}

    except PermissionError:
        logger.error(f"Permission error creating/overwriting file: {resolved_path}", exc_info=True)
        return {"status": "error", "action": "create_file_failed", "data": {"message": f"Permission denied to create/overwrite file: '{filepath_original}'"}}
    except IsADirectoryError:
         return {"status": "error", "action": "create_file_failed", "data": {"message": f"Cannot create file, path is a directory: '{filepath_original}'"}}
    except Exception as e:
        logger.exception(f"Unexpected error creating/overwriting file '{resolved_path}':")
        return {"status": "error", "action": "create_file_failed", "data": {"message": f"Unexpected error managing file '{filepath_original}': {e}"}}

def _append_to_file_logic(resolved_path: Path, filepath_original: str, content: str) -> dict:
    """Internal function to append content to a validated file path."""
    # Existence and type checks are done by the decorator for the calling skill
    try:
        # Ensure parent directory exists (safety check)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        with open(resolved_path, "a", encoding="utf-8") as f:
            # Ensure content ends with a newline for better formatting
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
         return {"status": "error", "action": "append_failed", "data": {"message": f"Cannot append to a directory: '{filepath_original}'"}}
    except Exception as e:
        logger.exception(f"Unexpected error appending to file '{resolved_path}':")
        return {"status": "error", "action": "append_failed", "data": {"message": f"Unexpected error appending to file '{filepath_original}': {e}"}}


# --- Skill: Create or Overwrite File --- (Refactored)
@skill(
    name="create_file",
    description="Creates a new text file or completely overwrites an existing one at a specified path within the workspace.",
    parameters={
        "content": (str, ...),
        "overwrite": (bool, False) # Logic handled internally based on decorator outcome
    }
)
@validate_workspace_path(arg_name='filepath', check_existence=False) # Decorator handles path validation
def create_file(content: str, overwrite: bool, resolved_path: Path, original_path_str: str, **kwargs) -> dict:
    """
    Creates/overwrites file. Path validation via @validate_workspace_path.
    Injects resolved_path: Path, original_path_str: str.
    """
    logger.debug(f"Skill 'create_file' requested. Path: '{original_path_str}', Resolved: '{resolved_path}', Overwrite: {overwrite}")
    
    # Basic type check for other args (decorator handles filepath)
    if not isinstance(content, str):
         return {"status": "error", "action": "create_file_failed", "data": {"message": "Content parameter must be a string."}}
    if not isinstance(overwrite, bool):
        return {"status": "error", "action": "create_file_failed", "data": {"message": "Overwrite parameter must be a boolean (true/false)."}}

    # Call the core logic function with the validated path
    return _create_file_logic(resolved_path, original_path_str, content, overwrite)


# --- Skill: Append to File --- (Refactored)
@skill(
    name="append_to_file",
    description="Appends text content to the end of an existing file within the workspace.",
    parameters={
        "content": (str, ...)
    }
)
@validate_workspace_path(arg_name='filepath', check_existence=True, target_type='file') # Decorator handles validation
def append_to_file(content: str, resolved_path: Path, original_path_str: str, **kwargs) -> dict:
    """
    Appends content to an existing file. Path validation via @validate_workspace_path.
    Injects resolved_path: Path, original_path_str: str.
    """
    logger.debug(f"Skill 'append_to_file' requested. Path: '{original_path_str}', Resolved: '{resolved_path}'")
    
    # Basic type check for content (decorator handles filepath)
    if not isinstance(content, str):
         return {"status": "error", "action": "append_failed", "data": {"message": "Content parameter must be a string."}}

    # Call the core logic function with the validated path
    return _append_to_file_logic(resolved_path, original_path_str, content)

# Remove the old combined function if it exists
# try:
#     del skill_manage_files
# except NameError:
#     pass 