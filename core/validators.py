# core/validators.py
import functools
import logging
from pathlib import Path
from typing import Callable, Any, Literal

# Assuming WORKSPACE_PATH is defined in config and setup correctly
try:
    # Try importing the specific variable we need
    from core.config import PROJECT_ROOT as WORKSPACE_PATH_STR
    WORKSPACE_PATH = Path(WORKSPACE_PATH_STR).resolve()
    logging.debug(f"validators.py using WORKSPACE_PATH from core.config: {WORKSPACE_PATH}")
except ImportError:
    logging.critical("CRITICAL: Could not import PROJECT_ROOT from core.config in validators.py. Path validation will fail.")
    # Define a fallback or raise an error if config is essential at import time
    # For now, let it fail during usage if WORKSPACE_PATH is None
    WORKSPACE_PATH = None

logger = logging.getLogger(__name__)

def validate_workspace_path(
    arg_name: str, # The name of the argument in the decorated function holding the path string
    check_existence: bool = True, # Should the validator check if the path exists?
    target_type: Literal['file', 'dir', 'any'] = 'any', # Should it be a file, a directory, or either?
    allow_hidden: bool = False, # Allow paths starting with '.'?
    action_name_on_error: str = "path_validation_failed", # Action name in error dict
    default_value: str | None = None # Default value if arg_name is missing
) -> Callable:
    """
    Decorator to validate a path argument for a skill function.

    Ensures the path is:
    1. A string.
    2. Resolvable within the defined WORKSPACE_PATH.
    3. Not pointing to hidden files/directories (unless allow_hidden=True).
    4. Optionally checks for existence.
    5. Optionally checks if it's a file or directory.

    Injects the resolved Path object as 'resolved_path' into the decorated function's kwargs.
    Returns a standard error dictionary on failure.

    Args:
        arg_name: The name of the keyword argument containing the path string.
        check_existence: If True, checks if the resolved path exists.
        target_type: 'file', 'dir', or 'any'. Checks the type if path exists.
        allow_hidden: If False, rejects paths containing components starting with '.'.
        action_name_on_error: The 'action' value for the error dict.
        default_value: Default value if arg_name is missing

    Returns:
        The decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if WORKSPACE_PATH is None:
                logger.error("WORKSPACE_PATH is not configured. Cannot perform path validation.")
                return {"status": "error", "action": action_name_on_error, "data": {"message": "Internal configuration error: WORKSPACE_PATH not set."}}

            path_str = kwargs.get(arg_name)

            # Enhanced logic to find path_str, assuming first arg might be action_input dict
            if not path_str:
                if args and isinstance(args[0], dict):
                    action_input = args[0]
                    path_str = action_input.get(arg_name)
                    if path_str:
                        logger.debug(f"Path argument '{arg_name}' found in action_input dictionary.")
                    else:
                        # Check for common aliases like 'file_path', 'directory', 'dir_path'
                        # This part might become complex, consider enforcing specific arg names or passing action_input
                        aliases = ['file_path', 'directory', 'dir_path']
                        if arg_name in aliases:
                             for alias in aliases:
                                 if alias != arg_name:
                                     path_str = action_input.get(alias)
                                     if path_str:
                                         logger.warning(f"Path argument '{arg_name}' not found, but alias '{alias}' was used.")
                                         # Decide if we should proceed with the alias value or enforce the primary arg_name
                                         # For now, let's proceed if an alias is found and primary is missing
                                         break # Use the first alias found

                # If still not found, log error
                if not path_str:
                    if default_value is not None:
                        path_str = default_value
                        logger.debug(f"Path argument '{arg_name}' not found, using default value: '{default_value}'")
                    else:
                        logger.error(f"Path argument '{arg_name}' not found in kwargs or action_input and no default value provided for {func.__name__}.")
                        return {"status": "error", "action": action_name_on_error, "data": {"message": f"Required path parameter '{arg_name}' is missing."}}
            
            # Validate type after confirming path_str is found
            if not isinstance(path_str, str) or not path_str:
                message = f"Invalid path parameter '{arg_name}'. Must be a non-empty string, but got type {type(path_str).__name__}."
                logger.error(message)
                return {"status": "error", "action": action_name_on_error, "data": {"message": message}}

            logger.debug(f"Validating path '{path_str}' for arg '{arg_name}' in function '{func.__name__}'")

            try:
                # --- Resolve Path ---
                path_obj = Path(path_str)

                # Prevent access to hidden files/dirs unless explicitly allowed
                # Check individual parts, ignore '.' and '..'
                if not allow_hidden and any(part.startswith('.') and part not in ['.', '..'] for part in path_obj.parts):
                    message = f"Access denied: Path '{path_str}' contains hidden components (starting with '.')."
                    logger.warning(message)
                    return {"status": "error", "action": action_name_on_error, "data": {"message": message}}

                # Resolve the path relative to workspace
                if path_obj.is_absolute():
                    # Even if absolute, ensure it resolves before checking workspace containment
                    try:
                       resolved_path = path_obj.resolve(strict=True) # Strict check if absolute path exists
                    except FileNotFoundError:
                        # If absolute path doesn't exist, but check_existence is False, we might allow it if it's within workspace bounds
                        if check_existence:
                             message = f"Path not found: Absolute path '{path_str}' does not exist."
                             logger.warning(message)
                             return {"status": "error", "action": action_name_on_error, "data": {"message": message}}
                        else:
                             # Resolve without strict to check containment only
                             resolved_path = path_obj.resolve()
                             logger.debug(f"Absolute path '{path_str}' does not exist, but proceeding as check_existence=False.")
                    except (OSError) as e:
                        logger.error(f"Error resolving absolute path '{path_str}': {e}", exc_info=True)
                        message = f"Error resolving absolute path '{path_str}'. It might be invalid or inaccessible. Error: {e}"
                        return {"status": "error", "action": action_name_on_error, "data": {"message": message}}
                else:
                    # Resolve relative paths from WORKSPACE_PATH
                    # Use strict=check_existence to control whether non-existent relative paths fail early
                    try:
                        resolved_path = (WORKSPACE_PATH / path_obj).resolve(strict=check_existence)
                    except FileNotFoundError:
                         # This happens if strict=True and path doesn't exist
                         message = f"Path not found: Relative path '{path_str}' does not exist within the workspace."
                         logger.warning(message)
                         return {"status": "error", "action": action_name_on_error, "data": {"message": message}}
                    except (OSError) as e:
                        logger.error(f"Error resolving relative path '{path_str}': {e}", exc_info=True)
                        message = f"Error resolving relative path '{path_str}'. It might be invalid or inaccessible. Error: {e}"
                        return {"status": "error", "action": action_name_on_error, "data": {"message": message}}

                # --- Workspace Check --- 
                # Need to re-resolve without strict if strict=False and path didn't exist initially
                if not check_existence and not resolved_path.exists():
                     # Re-resolve non-strictly ONLY to check containment boundary
                     containment_check_path = (WORKSPACE_PATH / path_obj).resolve() if not path_obj.is_absolute() else path_obj.resolve()
                else:
                     containment_check_path = resolved_path
                
                if not containment_check_path.is_relative_to(WORKSPACE_PATH):
                    message = f"Access denied: Path '{path_str}' resolves outside the designated workspace ('{WORKSPACE_PATH}'). Resolved to '{containment_check_path}'"
                    logger.warning(message)
                    return {"status": "error", "action": action_name_on_error, "data": {"message": message}}

                # --- Existence Check (Redundant if strict=True was used, but safe check) ---
                path_exists = resolved_path.exists()
                if check_existence and not path_exists:
                    # This case should ideally be caught by strict=True, but included for robustness
                    message = f"Path not found: '{path_str}' (resolved to '{resolved_path}') does not exist." 
                    logger.warning(message)
                    return {"status": "error", "action": action_name_on_error, "data": {"message": message}}

                # --- Type Check --- 
                if path_exists: # Only check type if it exists
                    is_file = resolved_path.is_file()
                    is_dir = resolved_path.is_dir()

                    if target_type == 'file' and not is_file:
                        message = f"Path is not a file: '{path_str}' (resolved to '{resolved_path}') is not a valid file."
                        logger.warning(message)
                        return {"status": "error", "action": action_name_on_error, "data": {"message": message}}
                    elif target_type == 'dir' and not is_dir:
                         message = f"Path is not a directory: '{path_str}' (resolved to '{resolved_path}') is not a valid directory."
                         logger.warning(message)
                         return {"status": "error", "action": action_name_on_error, "data": {"message": message}}
                # If path doesn't exist but check_existence is False, type check is skipped
                elif not check_existence:
                     logger.debug(f"Skipping type check for non-existent path '{path_str}' as check_existence=False.")


                # --- Validation Success --- 
                logger.debug(f"Path validation successful for '{path_str}'. Resolved to: {resolved_path}")
                # Inject the resolved path into the function call
                # Ensure we don't overwrite if already present (though functools.wraps helps)
                kwargs['resolved_path'] = resolved_path 
                # Pass the original path string as well, might be useful for messages
                kwargs['original_path_str'] = path_str 

                return func(*args, **kwargs)

            except (ValueError, OSError) as e: # Catch potential final resolution/filesystem errors
                 logger.error(f"Error validating path '{path_str}': {e}", exc_info=True)
                 message = f"Error processing path '{path_str}'. It might be invalid or inaccessible. Error: {e}"
                 return {"status": "error", "action": action_name_on_error, "data": {"message": message}}
            except Exception as e:
                 logger.exception(f"Unexpected error during path validation for '{path_str}':")
                 message = f"An unexpected error occurred while validating path '{path_str}' ({type(e).__name__})."
                 return {"status": "error", "action": action_name_on_error, "data": {"message": message}}

        return wrapper
    return decorator
