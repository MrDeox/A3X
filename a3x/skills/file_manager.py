from __future__ import annotations
import logging
from pathlib import Path
import shutil
from typing import Dict, Optional
import os

# Corrected absolute import using alias
from a3x.core.config import PROJECT_ROOT as WORKSPACE_ROOT

# Importar usando paths relativos
from a3x.core.skills import skill
from a3x.core.validators import validate_workspace_path
from a3x.core.context import SharedTaskContext
from a3x.core.context_accessor import ContextAccessor
from a3x.core.context import _ToolExecutionContext

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize a ContextAccessor instance for use in skills
_context_accessor = ContextAccessor()

# Define constants
MAX_ITEMS = 1000  # Limit for number of items listed
MAX_READ_SIZE = 1 * 1024 * 1024  # 1MB limit for reading files
PROTECTED_EXTENSIONS = {
    ".db",
    ".sqlite",
    ".sqlite3",
    ".env",
}  # Files not allowed to be deleted
TEXT_EXTENSIONS = {
    ".txt",
    ".py",
    ".md",
    ".json",
    ".env",
    ".csv",
    ".log",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".bash",
    ".zsh",
}  # Allowed extensions for reading

# Define WORKSPACE_ROOT here or ensure it's properly imported/configured
# Fallback to PROJECT_ROOT if a specific workspace isn't set elsewhere.
# This might be overridden by instance variable later.
# WORKSPACE_ROOT = WORKSPACE_ROOT # This line was redundant and potentially problematic


class FileManagerSkill:
    """Consolidated skill for managing files and directories within the workspace."""

    def __init__(self, workspace_root: Path | str | None = None):
        """Initializes the FileManagerSkill.

        Args:
            workspace_root: The root directory for file operations.
                          Defaults to PROJECT_ROOT from core.config.
        """
        if workspace_root is None:
            self.workspace_root = Path(WORKSPACE_ROOT)
            logger.info(
                f"FileManagerSkill initialized with default workspace: {self.workspace_root}"
            )
        else:
            self.workspace_root = Path(workspace_root).resolve()
            logger.info(
                f"FileManagerSkill initialized with provided workspace: {self.workspace_root}"
            )

        # Check for backup module on initialization
        try:
            from a3x.core.backup import create_backup

            self._create_backup = create_backup
            self._backup_enabled = True
            logger.debug("Backup module loaded successfully for FileManager.")
        except ImportError:
            logger.warning(
                "Backup module (core.backup) not found or failed to import for FileManager. Deletion backup will be unavailable."
            )
            self._create_backup = None
            self._backup_enabled = False

    @skill(
        name="list_directory",
        description="Lists files and directories within a specified sub-directory of the workspace.",
        parameters={
            "directory": {"type": str, "default": ".", "description": "Relative path to the directory (default: '.')"},
            "extension": {"type": Optional[str], "default": None, "description": "Filter by file extension (e.g., '.txt', optional)"}
        },
    )
    @validate_workspace_path(
        arg_name="directory", target_type="dir", check_existence=True
    )
    async def list_directory(
        self,
        ctx: _ToolExecutionContext,
        resolved_path: Path,
        original_path_str: str,
        directory: str = ".",
        extension: str | None = None,
    ) -> dict:
        """Lists directory contents. Path validation via decorator."""
        logger.debug(
            f"Skill 'list_directory' requested for: '{original_path_str}', Resolved: '{resolved_path}', Filter: '{extension}'"
        )

        # --- Extension Filter Validation ---
        filter_active = False
        normalized_extension = None
        if extension:
            if not isinstance(extension, str):
                logger.warning(
                    f"Invalid extension type: {type(extension)}. Filter ignored."
                )
            elif not extension.startswith("."):
                if "*." in extension:
                    normalized_extension = extension.replace("*.", ".").lower()
                elif extension:
                    normalized_extension = "." + extension.lower()
            elif len(extension) > 1:
                normalized_extension = extension.lower()

            if normalized_extension:
                filter_active = True
                logger.debug(f"Extension filter active: '{normalized_extension}'")
            elif extension:
                logger.debug(
                    f"Invalid or empty extension filter ('{extension}'), ignoring filter."
                )
                extension = None
        # --- End Filter Validation ---

        try:
            items = []
            count = 0
            workspace_resolved_path = self.workspace_root.resolve()

            for item in resolved_path.iterdir():
                if count >= MAX_ITEMS:
                    logger.warning(
                        f"Item limit ({MAX_ITEMS}) reached while listing '{resolved_path}'. List truncated."
                    )
                    break
                if item.name.startswith("."):
                    continue
                if (
                    filter_active
                    and item.is_file()
                    and item.suffix.lower() != normalized_extension
                ):
                    continue

                try:
                    relative_path = str(item.relative_to(workspace_resolved_path))
                except ValueError:
                    relative_path = item.name
                    logger.warning(
                        f"Could not make path relative: {item}. Using name only."
                    )

                items.append(relative_path + ("/" if item.is_dir() else ""))
                count += 1

            filter_message = (
                f" matching extension '{normalized_extension}'" if filter_active else ""
            )
            num_items = len(items)
            message = f"{num_items} non-hidden item(s){filter_message} found in '{original_path_str}'."
            if count >= MAX_ITEMS:
                message += f" (Result truncated at {MAX_ITEMS} items)"

            return {
                "status": "success",
                "action": "directory_listed",
                "data": {
                    "directory_requested": original_path_str,
                    "directory_resolved": str(
                        resolved_path.relative_to(workspace_resolved_path)
                    ),
                    "items": sorted(items),
                    "item_count": num_items,
                    "message": message,
                },
            }
        except PermissionError:
            logger.error(
                f"Permission error listing directory: {resolved_path}", exc_info=True
            )
            return {
                "status": "error",
                "action": "list_files_failed",
                "data": {
                    "message": f"Permission denied to list directory: '{original_path_str}'"
                },
            }
        except Exception as e:
            logger.exception(f"Unexpected error listing directory '{resolved_path}':")
            return {
                "status": "error",
                "action": "list_files_failed",
                "data": {
                    "message": f"Unexpected error listing directory '{original_path_str}': {e}"
                },
            }

    @skill(
        name="read_file",
        description="Reads the contents of a file at the specified path.",
        parameters={
            "path": {"type": str, "description": "The path to the file to read."},
        }
    )
    @validate_workspace_path(arg_name="path", check_existence=True, target_type="file")
    async def read_file(
        self,
        ctx: _ToolExecutionContext,
        path: str,
        resolved_path: Path = None,
        original_path_str: str = None,
    ) -> dict:
        """
        Reads the contents of a file at the specified path.
        
        Args:
            path (str): The path to the file to read.
        """
        log_prefix = "[read_file]"
        # Use original_path_str for user-facing logs/messages if needed
        logger.info(f"{log_prefix} Attempting to read file: '{original_path_str}' (Resolved: {resolved_path})")
        
        try:
            # Check if file exists (using resolved_path)
            if not resolved_path.exists(): # <<< CORRECTED
                error_msg = f"File not found: '{original_path_str}' (Resolved path did not exist: {resolved_path})" # Use original for msg
                logger.error(f"{log_prefix} {error_msg}")
                return {
                    "status": "error",
                    "action": "read_file_failed",
                    "data": {"message": f"Error: {error_msg}"},
                }
            
            # Check if it's actually a file (using resolved_path)
            if not resolved_path.is_file(): # <<< CORRECTED
                error_msg = f"Path is not a file: '{original_path_str}' (Resolved: {resolved_path})" # Use original for msg
                logger.error(f"{log_prefix} {error_msg}")
                return {
                    "status": "error",
                    "action": "read_file_failed",
                    "data": {"message": f"Error: {error_msg}"},
                }
            
            # Check if file is readable (using Path.stat() for permission check)
            # os.access might not be reliable, Path object methods are better
            try:
                _ = resolved_path.stat() # Check accessibility
            except PermissionError:
                error_msg = f"File is not readable (permission denied): '{original_path_str}' (Resolved: {resolved_path})" # Use original for msg
                logger.error(f"{log_prefix} {error_msg}")
                return {
                    "status": "error",
                    "action": "read_file_failed",
                    "data": {"message": f"Error: {error_msg}"},
                }
            except FileNotFoundError: # Should be caught above, but handle defensively
                 error_msg = f"File disappeared after check: '{original_path_str}'"
                 logger.error(f"{log_prefix} {error_msg}")
                 return {"status": "error", "action": "read_file_failed", "data": {"message": f"Error: {error_msg}"}}

            # Read the file contents (using resolved_path)
            with resolved_path.open('r', encoding='utf-8') as file: # <<< CORRECTED
                content = file.read()
            
            # Log using original path for user clarity
            logger.info(f"{log_prefix} Successfully read file: '{original_path_str}' ({len(content)} characters)")
            
            # Update context accessor using original path string (consistent key)
            await _context_accessor.set_last_read_file(original_path_str) # <<< Use original path
            logger.info(f"{log_prefix} Updated context with last read file: {original_path_str}")
            
            max_len_preview = 500
            content_preview = content[:max_len_preview] + (
                "..." if len(content) > max_len_preview else ""
            )

            line_count = len(content.splitlines())
            file_content = content

            result_payload = {
                "status": "success",
                "action": "file_read",
                "data": {
                    "filepath": original_path_str, # Keep original path here
                    "content": file_content,
                    "message": f"File '{original_path_str}' read successfully (Preview: {content_preview})",
                    "lines_read": line_count,
                },
            }

            return result_payload
        except Exception as e:
            error_msg = f"Failed to read file '{original_path_str}': {str(e)}" # Use original for msg
            logger.error(f"{log_prefix} {error_msg}", exc_info=True)
            return {
                "status": "error",
                "action": "read_file_failed",
                "data": {"message": f"Error: {error_msg}"},
            }

    @skill(
        name="write_file",
        description="Writes content to a specified file within the workspace.",
        parameters={
            "filename": {"type": str, "description": "The relative path (filename) of the file."},
            "content": {"type": str, "description": "The content to write to the file."},
            "overwrite": {"type": bool, "default": False, "description": "Whether to overwrite the file if it exists (default: False)."},
            "create_backup_flag": {"type": bool, "default": True, "description": "Whether to create a backup before writing (default: True)."},
        },
    )
    @validate_workspace_path(arg_name="filename", check_existence=False, target_type="file")
    async def write_file(
        self,
        ctx: _ToolExecutionContext,
        filename: str,
        content: str,
        overwrite: bool = False,
        create_backup_flag: bool = True,
        resolved_path: Path = None,
        original_path_str: str = None,
    ) -> dict:
        logger.debug(f"[DEBUG write_file] Entrou no método com argumentos: self={self}, ctx={ctx}, filename={filename}, content={content}, overwrite={overwrite}, create_backup_flag={create_backup_flag}, resolved_path={resolved_path}, original_path_str={original_path_str}")
        print(f"[DEBUG write_file] self={self}, ctx={ctx}, filename={filename}, content={content}, overwrite={overwrite}, create_backup_flag={create_backup_flag}, resolved_path={resolved_path}, original_path_str={original_path_str}")
        if resolved_path is None:
            logger.error(f"[DEBUG write_file] resolved_path está None! original_path_str={original_path_str}")
            print(f"[DEBUG write_file] resolved_path está None! original_path_str={original_path_str}")
        else:
            logger.debug(f"[DEBUG write_file] resolved_path={resolved_path}")
            print(f"[DEBUG write_file] resolved_path={resolved_path}")
        """
        Writes content to a specified file within the workspace.

        Args:
            filename (str): The relative path (filename) of the file.
            content (str): The content to write to the file.
            overwrite (bool): Whether to overwrite the file if it exists.
            create_backup_flag (bool): Whether to create a backup before writing.
            resolved_path (Path): The resolved absolute path (injected by decorator).
            original_path_str (str): The original path string passed (injected by decorator).

        Returns:
            dict: A dictionary indicating the result of the operation.
        """
        log_prefix = f"[write_file: '{original_path_str}']"
        logger.info(f"[DEBUG TEST] write_file called with filename={filename}, resolved_path={resolved_path}, overwrite={overwrite}, create_backup_flag={create_backup_flag}")
        # --- Log context if available ---
        # Access task_id via the shared context
        try:
            # task_id_info = f"Task ID from context: {ctx.task_id}" # OLD: Incorrect access
            task_id_info = f"Task ID from context: {ctx.shared_task_context.task_id}" # CORRECT: Access via shared_task_context
            ctx.logger.info(f"Write operation for {filename}. {task_id_info}")
        except AttributeError:
            ctx.logger.warning(f"Could not retrieve task_id from context in write_file.")
            task_id_info = "Task ID: Unknown"

        # Use the resolved path injected by the decorator
        target_path = resolved_path

        # Check if file exists and handle overwrite logic
        if target_path.exists() and not overwrite:
            msg = f"File '{original_path_str}' already exists and overwrite is False."
            logger.warning(f"{log_prefix} {msg}")
            return {
                "status": "error",
                "action": "write_file_failed",
                "data": {"message": msg},
            }

        # Attempt backup before writing if required and possible
        backup_made = False
        if (
            create_backup_flag
            and self._backup_enabled
            and target_path.exists()
        ):
            try:
                self._create_backup(target_path)
                backup_made = True
                logger.info(f"{log_prefix} Backup created for '{original_path_str}'")
            except Exception as backup_err:
                logger.error(
                    f"{log_prefix} Backup failed for '{original_path_str}'. Write operation cancelled. Error: {backup_err}",
                    exc_info=True,
                )
                return {
                    "status": "error",
                    "action": "write_file_failed_backup",
                    "data": {
                        "message": f"Backup failed for '{original_path_str}', write cancelled: {backup_err}"
                    },
                }

        try:
            # Ensure parent directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"{log_prefix} Ensured directory exists: {target_path.parent}")

            # Write the content
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"{log_prefix} Content successfully written.")

            return {
                "status": "success",
                "action": "file_written",
                "data": {
                    "filename": original_path_str, # Report original name
                    "message": f"File '{original_path_str}' written successfully.",
                    "backup_created": backup_made,
                },
            }
        except PermissionError:
            logger.error(f"{log_prefix} Permission error writing to file.", exc_info=True)
            return {
                "status": "error",
                "action": "write_file_failed",
                "data": {
                    "message": f"Permission denied writing to file: '{original_path_str}'"
                },
            }
        except Exception as e:
            logger.exception(f"{log_prefix} Unexpected error writing file:")
            return {
                "status": "error",
                "action": "write_file_failed",
                "data": {
                    "message": f"Unexpected error writing file '{original_path_str}': {e}"
                },
            }

    @skill(
        name="append_to_file",
        description="Appends content to an existing file within the workspace. Creates the file if it does not exist.",
        parameters={
            "path": {"type": str, "description": "The relative path to the file within the workspace."},
            "content": {"type": str, "description": "The content to append to the file."}
        },
    )
    @validate_workspace_path(arg_name="path", target_type="file", check_existence=False)
    async def append_to_file(
        self,
        ctx: _ToolExecutionContext,
        content: str,
        path: str,
        *,
        resolved_path: Path,
        original_path_str: str,
        **kwargs,
    ) -> dict:
        """Appends content to a file. Path validation/resolution via decorator."""
        logger.debug(
            f"Skill 'append_to_file' requested for: '{original_path_str}', Resolved: '{resolved_path}'"
        )

        if not isinstance(content, str):
            return {
                "status": "error",
                "action": "append_failed",
                "data": {"message": "Content parameter must be a string."},
            }

        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure the content to append ends with a newline IF content is provided
            content_to_write = content
            if content and not content.endswith("\n"):
                content_to_write += "\n"

            # Open in append mode and write the newline-terminated content
            with resolved_path.open("a", encoding="utf-8") as f:
                f.write(content_to_write)

            message = f"Content successfully appended to file '{original_path_str}'."
            logger.info(message)
            return {
                "status": "success",
                "action": "file_appended",
                "data": {"message": message, "filepath": original_path_str},
            }

        except PermissionError:
            logger.error(
                f"Permission error appending to file: {resolved_path}", exc_info=True
            )
            return {
                "status": "error",
                "action": "append_failed",
                "data": {
                    "message": f"Permission denied to append to file: '{original_path_str}'"
                },
            }
        except IsADirectoryError:
            return {
                "status": "error",
                "action": "append_failed",
                "data": {
                    "message": f"Cannot append to a directory: '{original_path_str}'"
                },
            }
        except Exception as e:
            logger.exception(f"Unexpected error appending to file '{resolved_path}':")
            return {
                "status": "error",
                "action": "append_failed",
                "data": {
                    "message": f"Unexpected error appending to file '{original_path_str}': {e}"
                },
            }

    @skill(
        name="delete_path",
        description="Deletes a specified file or directory within the workspace. Backup is mandatory.",
        parameters={
            "path": {"type": str, "description": "The relative path to the file or directory to delete."},
        },
    )
    @validate_workspace_path(arg_name="path", target_type="any", check_existence=True)
    async def delete_path(
        self,
        ctx: _ToolExecutionContext,
        resolved_path: Path,
        original_path_str: str,
        path: str,
        **kwargs,
    ) -> dict:
        """Deletes file/directory. Path validation via decorator. Backup mandatory."""
        logger.debug(
            f"Skill 'delete_path' requested for: '{original_path_str}', backup={True}"
        )

        # --- Security Check - Protected Extension ---
        if (
            resolved_path.is_file()
            and resolved_path.suffix.lower() in PROTECTED_EXTENSIONS
        ):
            logger.warning(
                f"Attempted deletion of protected file type: {resolved_path}"
            )
            return {
                "status": "error",
                "action": "delete_failed_protected",
                "data": {
                    "message": f"Operation not permitted: Deleting files with extension '{resolved_path.suffix}' is not allowed."
                },
            }

        # --- Backup ---
        backup_path_str = None
        if self._backup_enabled and self._create_backup:
            logger.debug(
                f"Attempting to create backup for {resolved_path} relative to {self.workspace_root}"
            )
            try:
                # Re-add passing the instance workspace root to the backup function
                backup_path_obj = self._create_backup(
                    str(resolved_path), workspace_root=self.workspace_root
                )
                if not backup_path_obj:
                    # Backup function should log the specific error
                    return {
                        "status": "error",
                        "action": "delete_failed_backup",
                        "data": {
                            "message": f"Failed to create backup for '{original_path_str}'. Deletion aborted."
                        },
                    }
                try:
                    # Try to make backup path relative to instance workspace for reporting
                    backup_path_str = str(
                        backup_path_obj.relative_to(self.workspace_root)
                    )
                except ValueError:
                    backup_path_str = str(
                        backup_path_obj
                    )  # Report absolute path if outside
                logger.info(
                    f"Backup created at workspace relative path: {backup_path_str}"
                )
            except Exception as backup_err:
                logger.error(
                    f"Error during backup creation for {resolved_path}: {backup_err}",
                    exc_info=True,
                )
                return {
                    "status": "error",
                    "action": "delete_failed_backup",
                    "data": {
                        "message": f"Error during backup creation for '{original_path_str}': {backup_err}. Deletion aborted."
                    },
                }
        elif not self._backup_enabled:
            logger.warning(
                f"Backup module disabled. Mandatory backup for '{original_path_str}' cannot be performed. Aborting deletion."
            )
            # Abort if backup is mandatory but disabled
            return {
                "status": "error",
                "action": "delete_failed_backup_disabled",
                "data": {
                    "message": f"Backup module is disabled, cannot proceed with mandatory backup for '{original_path_str}'."
                },
            }
        # If backup is False but required, the check at the start already handled it.

        # --- Deletion ---
        try:
            logger.info(
                f"Proceeding with confirmed deletion of validated path: {resolved_path}"
            )
            if resolved_path.is_dir():
                shutil.rmtree(resolved_path)
                action_name = "directory_deleted"
                message = f"Directory '{original_path_str}' deleted successfully."
            else:  # It's a file (decorator ensured it exists and is file or dir)
                resolved_path.unlink()
                action_name = "file_deleted"
                message = f"File '{original_path_str}' successfully deleted."

            logger.info(message)
            return {
                "status": "success",
                "action": action_name,
                "data": {
                    "message": message,
                    "filepath_deleted": original_path_str,
                    "backup_path": backup_path_str,
                },
            }
        except PermissionError:
            logger.error(f"Permission error deleting: {resolved_path}", exc_info=True)
            return {
                "status": "error",
                "action": "delete_failed_permission",
                "data": {
                    "message": f"Permission denied to delete: '{original_path_str}'"
                },
            }
        except OSError as e:  # Catch errors like deleting non-empty dir if rmtree fails
            logger.error(f"OS error deleting: {resolved_path}: {e}", exc_info=True)
            return {
                "status": "error",
                "action": "delete_failed_oserror",
                "data": {"message": f"OS error deleting '{original_path_str}': {e}"},
            }
        except Exception as e:
            logger.exception(f"Unexpected error deleting '{resolved_path}':")
            return {
                "status": "error",
                "action": "delete_failed_unexpected",
                "data": {
                    "message": f"Unexpected error deleting '{original_path_str}': {e}"
                },
            }

    @skill(
        name="create_directory",
        description="Creates a directory (including any necessary parent directories) at the specified path within the workspace.",
        parameters={
            "directory": {"type": str, "description": "The relative path to the directory to create."}
        }
    )
    @validate_workspace_path(arg_name="directory", target_type="dir", check_existence=False)
    async def create_directory(
        self,
        ctx: _ToolExecutionContext,
        directory: str,
        resolved_path: Path,
        original_path_str: str,
    ) -> dict:
        """Creates a directory, including parent directories if needed."""
        logger.info(f"[create_directory] Attempting to create directory: '{original_path_str}' (Resolved: {resolved_path})")
        
        try:
            resolved_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"[create_directory] Successfully created directory: {resolved_path}")
            return {
                "status": "success",
                "action": "directory_created",
                "data": {"message": f"Directory '{original_path_str}' created successfully."}
            }
        except PermissionError:
            logger.error(f"[create_directory] Permission denied to create directory: {resolved_path}", exc_info=True)
            return {
                "status": "error",
                "action": "create_directory_failed",
                "data": {"message": f"Permission denied to create directory '{original_path_str}'."}
            }
        except Exception as e:
            logger.exception(f"[create_directory] Unexpected error creating directory '{resolved_path}':")
            return {
                "status": "error",
                "action": "create_directory_failed",
                "data": {"message": f"Unexpected error creating directory '{original_path_str}': {e}"}
            }


# Instantiate the skill class so the methods are registered by the decorator
file_manager = FileManagerSkill()
