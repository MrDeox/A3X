# skills/delete_file.py
import os
import logging
from pathlib import Path
import traceback

# Ajuste para importar do diretório pai
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

# Import dependencies
try:
    from core.backup import create_backup
    from core.validators import validate_workspace_path
    # Use the same name consistently (PROJECT_ROOT as WORKSPACE_PATH)
    from core.config import PROJECT_ROOT as WORKSPACE_PATH_STR
    WORKSPACE_PATH = Path(WORKSPACE_PATH_STR).resolve()
    logger.debug(f"delete_file using WORKSPACE_PATH from core.config: {WORKSPACE_PATH}")
except ImportError as e:
    logger.critical(f"CRITICAL: Failed to import dependencies for delete_file: {e}", exc_info=True)
    create_backup = None
    validate_workspace_path = None # Need to handle this case
    WORKSPACE_PATH = None

# Define extensões perigosas/protegidas que não podem ser deletadas
PROTECTED_EXTENSIONS = [".db", ".sqlite", ".sqlite3", ".env"]

# Apply the decorator
@validate_workspace_path(
    arg_name='file_path', # The keyword argument holding the path in action_input
    check_existence=True, # The file must exist to be deleted
    target_type='file', # It must be a file
    allow_hidden=False, # Generally safer not to delete hidden files
    action_name_on_error="delete_file_failed" # Action name for error reporting
)
def skill_delete_file(action_input: dict, resolved_path: Path = None, original_path_str: str = None, agent_memory: dict = None, agent_history: list | None = None) -> dict:
    """
    Deletes a specified file AFTER confirmation, with automatic backup.
    Relies on @validate_workspace_path for path validation and checks.

    Args:
        action_input (dict): Dictionary containing:
            - file_path (str): Relative or absolute path to the file to delete. (Processed by decorator)
            - confirm (bool): Confirmation flag (MUST be True).
        resolved_path (Path, injected): The validated, absolute Path object for the file.
        original_path_str (str, injected): The original path string requested.
        agent_memory (dict, optional): Agent's memory (not used).
        agent_history (list | None, optional): Conversation history (not used).

    Returns:
        dict: Standardized dictionary with the result.
    """
    logger.debug(f"Executing skill_delete_file for validated path: {resolved_path} (original input: '{original_path_str}')")

    # Basic check if decorator failed or dependencies are missing
    if not resolved_path or not WORKSPACE_PATH or not create_backup:
        # Log detailed error if possible
        if not resolved_path:
             logger.error("Decorator failed to inject resolved_path into skill_delete_file.")
        if not WORKSPACE_PATH:
             logger.error("WORKSPACE_PATH is not available in skill_delete_file.")
        if not create_backup:
             logger.error("create_backup function is not available in skill_delete_file.")
        return {"status": "error", "action": "delete_file_failed", "data": {"message": "Internal configuration error: Skill dependencies missing."}}

    # Use original_path_str for user-facing messages if available
    display_path = original_path_str if original_path_str else str(resolved_path.relative_to(WORKSPACE_PATH))

    # 1. Validate Confirmation Input (Still need this specific logic)
    confirm = action_input.get("confirm", False) # Default False if not provided
    if not isinstance(confirm, bool):
         return {"status": "error", "action": "delete_file_failed", "data": {"message": f"Parameter 'confirm' must be a boolean (true/false) for '{display_path}'."}}
    if not confirm:
        return {"status": "error", "action": "delete_file_failed", "data": {"message": f"Deletion confirmation missing for '{display_path}'. Set 'confirm': true to proceed."}}

    # Path validation (existence, type=file, workspace containment) is now done by the decorator.

    try:
        # 2. Validação de Segurança - Extensão Protegida? (Still need this business logic)
        if resolved_path.suffix.lower() in PROTECTED_EXTENSIONS:
            return {"status": "error", "action": "delete_file_failed", "data": {"message": f"Operation not permitted: Deleting files with the extension '{resolved_path.suffix}' is not allowed for '{display_path}'."}}

        # 3. Backup
        logger.debug(f"Attempting to create backup for {resolved_path}")
        backup_path = create_backup(str(resolved_path)) # Pass the absolute path
        if not backup_path:
            # Error already logged within create_backup
            return {"status": "error", "action": "delete_file_failed", "data": {"message": f"Failed to create backup before deleting '{display_path}'."}}
        logger.info(f"Backup created successfully at {backup_path}")

        # 4. Exclusão
        logger.debug(f"Attempting to delete {resolved_path}")
        resolved_path.unlink() # Deletes the original file
        logger.info(f"File '{resolved_path.name}' (from path '{display_path}') deleted successfully.")

        backup_display_path = str(backup_path.relative_to(WORKSPACE_PATH)) if WORKSPACE_PATH else str(backup_path)

        return {
            "status": "success",
            "action": "file_deleted",
            "data": {
                "message": f"File '{display_path}' deleted successfully.",
                "file_path": display_path,
                "backup_path": backup_display_path # Report relative backup path
            }
        }

    except PermissionError:
        logger.error(f"Permission error trying to delete '{resolved_path}'", exc_info=True)
        return {"status": "error", "action": "delete_file_failed", "data": {"message": f"Permission denied to delete file: '{display_path}'"}}
    except Exception as e:
        logger.exception(f"Unexpected error trying to delete '{resolved_path}':")
        return {"status": "error", "action": "delete_file_failed", "data": {"message": f"Unexpected error deleting file '{display_path}': {e}"}}
