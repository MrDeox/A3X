# skills/delete_file.py
import os
import logging
from pathlib import Path
import traceback
from typing import Dict, Any
import re  # Adicionado para remover cores ANSI
import json
# from core.validators import validate_path # REMOVED - assuming validation logic is inline
from core.tools import skill

# Corrected absolute import using alias
from core.config import PROJECT_ROOT as WORKSPACE_ROOT
try:
    from ..core.backup import create_backup
    BACKUP_ENABLED = True
    logger = logging.getLogger(__name__)
    logger.debug("Backup module loaded successfully.")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Backup module (core.backup) not found or failed to import. File deletion will proceed without backup.")
    create_backup = None # type: ignore
    BACKUP_ENABLED = False

# Define protected extensions (using a set for efficiency)
PROTECTED_EXTENSIONS = {".db", ".sqlite", ".sqlite3", ".env"}

@skill(
    name="delete_file",
    description="Deletes a specified file or directory within the workspace.",
    parameters={
        "filepath": (str, ...),
        "backup": (bool, False), # Added parameter for backup control
        "agent_history": (list | None, None) # Added missing parameter
    }
)
def delete_file(filepath: str, backup: bool = False, agent_memory: dict | None = None, agent_history: list | None = None) -> dict:
    """
    Deletes the specified file or directory within the workspace.

    Args:
        filepath (str): Relative path within the workspace to the file or directory to delete.
        backup (bool, optional): Flag to create a backup before deleting. Defaults to False.
        agent_memory (dict | None, optional): Agent's memory (not used). Defaults to None.
        agent_history (list | None, optional): Conversation history (not used). Defaults to None.

    Returns:
        dict: Standardized dictionary with the result.
    """
    logger.debug(f"Skill 'delete_file' requested for: '{filepath}', backup={backup}")

    # 1. Validate Confirmation Parameter
    # Pydantic handles bool type check via the decorator schema, but we need to check the value.
    if not backup:
        # This is not an error per se, but a failure to meet pre-conditions.
        logger.warning(f"Deletion confirmation missing or false for '{filepath}'. Aborting.")
        return {"status": "error", "action": "delete_confirmation_missing", "data": {"message": f"Deletion not confirmed for '{filepath}'. Set 'backup': true to proceed."}}

    # --- Path Validation --- (Type check for filepath done by Pydantic)
    if not filepath:
        return {"status": "error", "action": "delete_file_failed", "data": {"message": "Filepath parameter cannot be empty."}}

    try:
        workspace_resolved_path = Path(WORKSPACE_ROOT).resolve()
        # Check for invalid path patterns
        if os.path.isabs(filepath) or ".." in filepath or ".." in Path(filepath).parts:
             raise ValueError("Invalid path: Absolute paths or paths with '..' are not allowed. Use relative paths within the workspace.")

        abs_path = workspace_resolved_path / filepath
        resolved_path = abs_path.resolve()

        # Check if resolved path is within workspace
        if not str(resolved_path).startswith(str(workspace_resolved_path)):
             raise ValueError("Invalid path: Path resolves outside the workspace boundaries.")

        # Check existence and type (must exist and be a file or directory)
        if not resolved_path.exists():
            # Return success if trying to delete non-existent file? Or error? Error is safer.
            logger.warning(f"Attempted deletion of non-existent file or directory: {resolved_path}")
            return {"status": "error", "action": "delete_file_failed_not_found", "data": {"message": f"File or directory not found: '{filepath}' does not exist."}}
        if not resolved_path.is_file() and not resolved_path.is_dir():
             logger.warning(f"Attempted deletion of non-file or non-directory path: {resolved_path}")
             return {"status": "error", "action": "delete_file_failed_not_file_or_directory", "data": {"message": f"Path is not a file or directory: Cannot delete '{filepath}'."}}

        # Check for hidden files (optional, uncomment if needed)
        # if resolved_path.name.startswith('.'):
        #     logger.warning(f"Attempted deletion of hidden file: {resolved_path}")
        #     return {"status": "error", "action": "delete_file_failed_hidden", "data": {"message": f"Deleting hidden files is not allowed: '{filepath}'"}}


    except ValueError as e:
        logger.warning(f"Path validation failed for '{filepath}': {e}")
        return {"status": "error", "action": "delete_file_failed_validation", "data": {"message": f"Path validation failed: {e}"}}
    except Exception as e:
        logger.error(f"Unexpected error resolving path '{filepath}': {e}", exc_info=True)
        return {"status": "error", "action": "delete_file_failed_internal", "data": {"message": f"Internal error processing path: {e}"}}
    # --- End Path Validation ---

    logger.debug(f"Path validated for deletion: '{filepath}' -> '{resolved_path}'")

    try:
        # 2. Security Check - Protected Extension?
        if resolved_path.suffix.lower() in PROTECTED_EXTENSIONS:
            logger.warning(f"Attempted deletion of protected file type: {resolved_path}")
            return {"status": "error", "action": "delete_file_failed_protected", "data": {"message": f"Operation not permitted: Deleting files with the extension '{resolved_path.suffix}' is not allowed for '{filepath}'."}}

        # 3. Backup (if enabled)
        backup_path_str = None
        if BACKUP_ENABLED and create_backup:
            logger.debug(f"Attempting to create backup for {resolved_path}")
            try:
                 backup_path_obj = create_backup(str(resolved_path)) # Pass absolute path string
                 if not backup_path_obj:
                     # Error should be logged within create_backup
                     return {"status": "error", "action": "delete_file_failed_backup", "data": {"message": f"Failed to create backup before deleting '{filepath}'. Deletion aborted."}}
                 # Calculate relative path for reporting, handle potential errors
                 try:
                     backup_path_str = str(backup_path_obj.relative_to(workspace_resolved_path))
                 except ValueError:
                     logger.warning(f"Could not determine relative path for backup {backup_path_obj}. Reporting absolute path.")
                     backup_path_str = str(backup_path_obj)

                 logger.info(f"Backup created successfully at workspace relative path: {backup_path_str}")
            except Exception as backup_err:
                 logger.error(f"Error during backup creation for {resolved_path}: {backup_err}", exc_info=True)
                 return {"status": "error", "action": "delete_file_failed_backup", "data": {"message": f"Error during backup creation for '{filepath}': {backup_err}. Deletion aborted."}}
        elif not BACKUP_ENABLED:
             logger.warning(f"Proceeding with deletion of '{filepath}' without backup as backup module is disabled.")


        # 4. Deletion
        logger.info(f"Proceeding with confirmed deletion of validated path: {resolved_path}")
        resolved_path.unlink() # Deletes the file or directory
        logger.info(f"File or directory '{resolved_path.name}' (from requested path '{filepath}') deleted successfully.")

        return {
            "status": "success",
            "action": "file_deleted",
            "data": {
                "message": f"File or directory '{filepath}' deleted successfully.",
                "filepath_deleted": filepath,
                "backup_path": backup_path_str if backup_path_str else ("Backup disabled" if not BACKUP_ENABLED else "Backup failed")
            }
        }

    except PermissionError:
        logger.error(f"Permission error trying to delete '{resolved_path}'", exc_info=True)
        return {"status": "error", "action": "delete_file_failed_permission", "data": {"message": f"Permission denied to delete file or directory: '{filepath}'"}}
    except Exception as e:
        logger.exception(f"Unexpected error trying to delete '{resolved_path}':")
        return {"status": "error", "action": "delete_file_failed_unexpected", "data": {"message": f"Unexpected error deleting file or directory '{filepath}': {e}"}}

# Remove old function if needed
# try:
#     del skill_delete_file
# except NameError:
#     pass
