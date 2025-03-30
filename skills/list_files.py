# skills/list_files.py
import logging
from pathlib import Path

# Import the new validator
from core.validators import validate_workspace_path
# Import WORKSPACE_PATH only for constructing relative paths in the output, if needed
# Or rely on the fact that items are already relative if iterdir() is used on a path within WORKSPACE_PATH
# Try importing WORKSPACE_PATH directly (adjust name if needed, e.g., PROJECT_ROOT)
try:
    from core.config import PROJECT_ROOT as WORKSPACE_PATH
    if isinstance(WORKSPACE_PATH, str):
         WORKSPACE_PATH = Path(WORKSPACE_PATH).resolve()
    logging.debug(f"list_files using WORKSPACE_PATH from core.config: {WORKSPACE_PATH}")
except ImportError:
    logging.error("CRITICAL: Could not import WORKSPACE_PATH from core.config in list_files.py.", exc_info=True)
    WORKSPACE_PATH = None # Indicate failure


# Initialize logger
logger = logging.getLogger(__name__)

# Define constants
MAX_ITEMS = 1000 # Limit for number of items listed

# --- Skill Function ---

# Apply the decorator to validate the 'directory' argument
@validate_workspace_path(
    arg_name='directory', # The keyword argument holding the path
    check_existence=True, # The directory must exist
    target_type='dir', # It must be a directory
    allow_hidden=False, # Do not allow listing hidden paths explicitly
    action_name_on_error="list_files_failed", # Action name for error reporting
    default_value="." # Default to current directory
)
def skill_list_files(action_input: dict, resolved_path: Path = None, original_path_str: str = None, agent_memory: dict = None, agent_history: list | None = None) -> dict:
    """
    Lists non-hidden files and directories within a specified workspace directory.
    Optionally filters by a specific file extension.

    Relies on the @validate_workspace_path decorator to handle path validation,
    workspace checks, existence checks, and type checks for the directory.

    Args:
        action_input (dict): The original dictionary passed to the skill, potentially
                             containing 'directory' or an alias.
        resolved_path (Path, injected): The validated, absolute Path object for the
                                       directory, injected by the decorator.
        extension (str, optional): If provided (e.g., via action_input['extension']),
                                   filters the results to include only files with this extension.
                                   Must include the leading dot (e.g., '.py').
        original_path_str (str, injected): The original path string requested,
                                           injected by the decorator.
        agent_memory (dict, optional): Agent's memory (not used).
        agent_history (list | None, optional): Conversation history (not used).


    Returns:
        dict: Standardized dictionary:
              {"status": "success/error", "action": "directory_listed/list_files_failed",
               "data": {"message": "...", "directory": "...", "items": [...]}}
    """
    # The decorator handles 'directory'. We need to manually get 'extension'.
    # <<< MODIFIED: Check for 'file_filter' and 'file_name_pattern' as well >>>
    possible_keys = ['extension', 'file_filter', 'file_name_pattern', 'filter']
    extension = None
    if isinstance(action_input, dict):
        for key in possible_keys:
            if key in action_input:
                extension = action_input[key]
                break # Use the first one found

    # extension = action_input.get('extension') or action_input.get('file_filter') if isinstance(action_input, dict) else None
    if extension and not extension.startswith('.'):
        # Handle glob patterns like '*.py'
        if '*.' in extension:
            extension = extension.replace('*.', '.') # Normalize to just the extension
        elif not extension.startswith('.'):
            extension = "." + extension # Ensure leading dot for simple extensions
    if extension == '.': # Ignore if just a dot
        extension = None

    # The decorator has already validated 'directory' from action_input, resolved it to 'resolved_path',
    # checked it exists, is a directory, and is within the workspace.
    # It also provides the original string via 'original_path_str'.
    # We can directly use 'resolved_path'.

    if not resolved_path:
         # Should technically be caught by decorator, but as a safeguard:
         logger.error("Decorator failed to inject resolved_path into skill_list_files.")
         return {"status": "error", "action": "list_files_failed", "data": {"message": "Internal error: Path validation failed unexpectedly."}}

    if WORKSPACE_PATH is None:
         logger.error("WORKSPACE_PATH is not available in skill_list_files.")
         return {"status": "error", "action": "list_files_failed", "data": {"message": "Internal configuration error: WORKSPACE_PATH not loaded."}}

    # Use original_path_str for logging/reporting if available, otherwise fallback
    path_repr = original_path_str if original_path_str else str(resolved_path)

    logger.debug(f"Executing skill_list_files for validated path: {resolved_path} (original input: '{path_repr}')")

    try:
        items = []
        count = 0
        # Use iterdir on the validated, resolved path
        filter_active = bool(extension)
        logger.debug(f"Listing directory: {resolved_path}. Extension filter: '{extension}' (Active: {filter_active})")

        for item in resolved_path.iterdir():
            if count >= MAX_ITEMS:
                 logger.warning(f"Item limit ({MAX_ITEMS}) reached while listing '{resolved_path}'. List truncated.")
                 break
            # Filter hidden files/directories (starting with '.')
            # The decorator already prevents targeting hidden dirs directly,
            # but we still need to filter contents *within* the target dir.
            if not item.name.startswith('.'):
                try:
                    # Apply extension filter if active AND item is a file
                    if filter_active and item.is_file() and item.suffix.lower() != extension.lower():
                        continue # Skip this item if extension doesn't match

                    # Make path relative to WORKSPACE_PATH for consistent output
                    relative_path = str(item.relative_to(WORKSPACE_PATH))
                except ValueError:
                     # Should not happen if resolved_path is within WORKSPACE_PATH,
                     # but handle as fallback just in case (e.g., strange mounts/links)
                     relative_path = item.name
                     logger.warning(f"Could not make path relative to workspace: {item}. Using name only.")

                # Append '/' to directories for clarity in the output list
                if item.is_dir():
                    items.append(relative_path + "/")
                else:
                    items.append(relative_path)
                count += 1

        filter_message = f" matching extension '{extension}'" if filter_active else ""
        num_items = len(items)
        message = f"{num_items} non-hidden item(s){filter_message} found in '{path_repr}'."
        if count >= MAX_ITEMS:
            message += f" (Result truncated at {MAX_ITEMS} items)"

        return {
            "status": "success",
            "action": "directory_listed",
            "data": {
                "directory": path_repr, # Report the original requested path
                "resolved_directory": str(resolved_path.relative_to(WORKSPACE_PATH)), # Relative resolved path
                "items": sorted(items), # Return the full (but potentially truncated) sorted list
                "item_count": num_items,
                "message": message
            }
        }

    except PermissionError:
        logger.error(f"Permission error listing directory: {resolved_path}", exc_info=True)
        return {"status": "error", "action": "list_files_failed", "data": {"message": f"Permission denied to list directory: '{path_repr}'"}}
    except Exception as e:
        logger.exception(f"Unexpected error listing directory '{resolved_path}':")
        return {"status": "error", "action": "list_files_failed", "data": {"message": f"Unexpected error listing directory '{path_repr}': {e}"}}
