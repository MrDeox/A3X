import logging
from pathlib import Path
import shutil

# Corrected absolute import using alias
from core.config import PROJECT_ROOT as WORKSPACE_ROOT

# Importar usando paths relativos
from core.tools import skill
from core.validators import validate_workspace_path

# Initialize logger
logger = logging.getLogger(__name__)

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
WORKSPACE_ROOT = WORKSPACE_ROOT


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
            from core.backup import create_backup

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
            "directory": (str, "Relative path to the directory (default: '.')"),
            "extension": (str | None, "Filter by file extension (e.g., '.txt')"),
        },
    )
    @validate_workspace_path(
        arg_name="directory", target_type="dir", check_existence=True
    )
    async def list_directory(
        self,
        resolved_path: Path,
        original_path_str: str,
        directory: str = ".",
        extension: str | None = None,
        **kwargs,
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
        description=f"Reads the entire content of a specified text file within the workspace (up to {MAX_READ_SIZE // (1024 * 1024)}MB). Supported extensions: {', '.join(TEXT_EXTENSIONS)}.",
        parameters={"filepath": (str, ...)},
    )
    @validate_workspace_path(
        arg_name="filepath", target_type="file", check_existence=True
    )
    async def read_file(
        self, resolved_path: Path, original_path_str: str, filepath: str, **kwargs
    ) -> dict:
        """Reads file content. Path validation via decorator."""
        logger.debug(
            f"Skill 'read_file' requested for: '{original_path_str}', Resolved: '{resolved_path}'"
        )

        file_ext = resolved_path.suffix.lower()
        if file_ext not in TEXT_EXTENSIONS:
            logger.warning(
                f"Attempted read of unsupported extension: {file_ext} in '{original_path_str}'"
            )
            return {
                "status": "error",
                "action": "read_file_failed_unsupported_ext",
                "data": {
                    "message": f"Extension '{file_ext}' not supported for reading. Allowed: {', '.join(TEXT_EXTENSIONS)}"
                },
            }

        try:
            file_size = resolved_path.stat().st_size
            if file_size > MAX_READ_SIZE:
                logger.warning(
                    f"Attempted read of large file: {file_size} bytes in '{original_path_str}'"
                )
                return {
                    "status": "error",
                    "action": "read_file_failed_too_large",
                    "data": {
                        "message": f"File too large ({file_size / (1024 * 1024):.2f} MB). Limit: {MAX_READ_SIZE // (1024 * 1024)} MB."
                    },
                }
        except OSError as e:
            logger.error(
                f"OSError checking size of '{original_path_str}': {e}", exc_info=True
            )
            return {
                "status": "error",
                "action": "read_file_failed_stat_error",
                "data": {"message": f"Error checking file size: {e}"},
            }

        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                content = f.read()

            max_len_preview = 500
            content_preview = content[:max_len_preview] + (
                "..." if len(content) > max_len_preview else ""
            )

            return {
                "status": "success",
                "action": "file_read",
                "data": {
                    "filepath": original_path_str,
                    "content": content,
                    "message": f"File '{original_path_str}' read successfully (Preview: {content_preview})",
                },
            }
        except PermissionError:
            logger.error(
                f"Permission error reading file: {original_path_str}", exc_info=True
            )
            return {
                "status": "error",
                "action": "read_file_failed",
                "data": {
                    "message": f"Permission denied to read file: '{original_path_str}'"
                },
            }
        except IsADirectoryError:
            return {
                "status": "error",
                "action": "read_file_failed",
                "data": {
                    "message": f"Path is a directory, not a file: '{original_path_str}'"
                },
            }
        except Exception as e:
            logger.exception(f"Unexpected error reading file '{original_path_str}':")
            return {
                "status": "error",
                "action": "read_file_failed",
                "data": {
                    "message": f"Unexpected error reading file '{original_path_str}': {e}"
                },
            }

    @skill(
        name="write_file",
        description="Creates a new text file or completely overwrites an existing one at a specified path within the workspace.",
        parameters={
            "filepath": (str, ...),
            "content": (str, ...),
            "overwrite": (bool, False),
        },
    )
    @validate_workspace_path(
        arg_name="filepath", target_type="any", check_existence=False
    )
    async def write_file(
        self,
        content: str,
        overwrite: bool,
        resolved_path: Path,
        original_path_str: str,
        filepath: str,
        **kwargs,
    ) -> dict:
        """Creates/overwrites file. Path validation via decorator."""
        logger.debug(
            f"Skill 'write_file' requested. Path: '{original_path_str}', Overwrite: {overwrite}"
        )

        if not isinstance(content, str):
            return {
                "status": "error",
                "action": "write_file_failed",
                "data": {"message": "Content parameter must be a string."},
            }
        if not isinstance(overwrite, bool):
            return {
                "status": "error",
                "action": "write_file_failed",
                "data": {
                    "message": "Overwrite parameter must be a boolean (true/false)."
                },
            }

        try:
            # Check existence and type *again* based on overwrite flag
            if resolved_path.exists():
                if resolved_path.is_dir():
                    return {
                        "status": "error",
                        "action": "write_file_failed",
                        "data": {
                            "message": f"Cannot create file, a directory already exists at '{original_path_str}'"
                        },
                    }
                if not overwrite:
                    return {
                        "status": "error",
                        "action": "write_file_failed",
                        "data": {
                            "message": f"File '{original_path_str}' already exists. Use overwrite=True to replace it."
                        },
                    }
                # If overwrite=True and it's a file, we proceed.

            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(content)

                action_name = (
                    "file_overwritten"
                    if overwrite and resolved_path.exists()
                    else "file_created"
                )  # Adjust action based on prior existence for accuracy
                message = f"File '{original_path_str}' was successfully {action_name.replace('_', ' ')}."
            logger.info(message)
            return {
                "status": "success",
                "action": action_name,
                "data": {"message": message, "filepath": original_path_str},
            }

        except PermissionError:
            logger.error(
                f"Permission error writing file: {resolved_path}", exc_info=True
            )
            return {
                "status": "error",
                "action": "write_file_failed",
                "data": {
                    "message": f"Permission denied to write file: '{original_path_str}'"
                },
            }
        except IsADirectoryError:  # Should be caught above, but safeguard
            return {
                "status": "error",
                "action": "write_file_failed",
                "data": {
                    "message": f"Cannot write to a directory: '{original_path_str}'"
                },
            }
        except Exception as e:
            logger.exception(f"Unexpected error writing file '{resolved_path}':")
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
            "filepath": (str, "The relative path to the file within the workspace."),
            "content": (str, "The content to append to the file."),
        },
    )
    @validate_workspace_path(
        arg_name="filepath", target_type="file", check_existence=False
    )
    async def append_to_file(
        self,
        content: str,
        resolved_path: Path,
        original_path_str: str,
        filepath: str,
        **kwargs,
    ) -> dict:
        """Appends content to file. Path validation via decorator."""
        logger.debug(f"Skill 'append_to_file' requested. Path: '{original_path_str}'")

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
            "filepath": (str, ...),
            "backup": (bool, True),
        },  # Backup defaults to True and is mandatory
    )
    @validate_workspace_path(
        arg_name="filepath", target_type="any", check_existence=True
    )
    async def delete_path(
        self,
        backup: bool,
        resolved_path: Path,
        original_path_str: str,
        filepath: str,
        **kwargs,
    ) -> dict:
        """Deletes file/directory. Path validation via decorator. Backup mandatory."""
        logger.debug(
            f"Skill 'delete_path' requested for: '{original_path_str}', backup={backup}"
        )

        # Enforce backup=True
        if not backup:
            logger.warning(
                f"Deletion requires 'backup=True' for '{original_path_str}'. Aborting."
            )
            return {
                "status": "error",
                "action": "delete_confirmation_missing",
                "data": {
                    "message": f"Deletion requires 'backup=True' for '{original_path_str}'. Operation aborted."
                },
            }

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


# Instantiate the skill class so the methods are registered by the decorator
file_manager = FileManagerSkill()
