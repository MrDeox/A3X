# core/validators.py
import functools
import logging
from pathlib import Path
from typing import Callable, Any, Literal
import inspect

# Assuming WORKSPACE_PATH is defined in config and setup correctly
try:
    # Try importing the specific variable we need
    # from core.config import PROJECT_ROOT as WORKSPACE_PATH_STR
    from a3x.core.config import PROJECT_ROOT as WORKSPACE_PATH_STR

    WORKSPACE_PATH = Path(WORKSPACE_PATH_STR).resolve()
    logging.debug(
        f"validators.py using WORKSPACE_PATH from core.config: {WORKSPACE_PATH}"
    )
except ImportError:
    logging.critical(
        "CRITICAL: Could not import PROJECT_ROOT from core.config in validators.py. Path validation will fail."
    )
    # Define a fallback or raise an error if config is essential at import time
    # For now, let it fail during usage if WORKSPACE_PATH is None
    WORKSPACE_PATH = None

logger = logging.getLogger(__name__)


def validate_workspace_path(
    arg_name: str,  # The name of the argument in the decorated function holding the path string
    check_existence: bool = True,  # Should the validator check if the path exists?
    target_type: Literal[
        "file", "dir", "any"
    ] = "any",  # Should it be a file, a directory, or either?
    allow_hidden: bool = False,  # Allow paths starting with '.'?
    action_name_on_error: str = "path_validation_failed",  # Action name in error dict
    default_value: str | None = None,  # Default value if arg_name is missing
) -> Callable:
    """
    Decorator to validate a path argument for a skill function.

    Ensures the path is:
    1. A string.
    2. Resolvable within the skill instance's defined workspace_root.
    3. Not pointing to hidden files/directories (unless allow_hidden=True).
    4. Optionally checks for existence.
    5. Optionally checks if it's a file or directory.

    Injects the resolved Path object as 'resolved_path' into the decorated function's kwargs.
    Returns a standard error dictionary on failure.
    Correctly handles decorating async functions.

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
        async def wrapper(*args, **kwargs) -> Any:  # Make wrapper async
            # Extract the instance (self) from the first argument
            if not args or not hasattr(args[0], "workspace_root"):
                logger.error(
                    f"Validator applied to non-skill instance or instance without 'workspace_root' for {func.__name__}"
                )
                return {
                    "status": "error",
                    "action": "validator_error",
                    "data": {
                        "message": "Internal validator configuration error (instance or workspace missing)."
                    },
                }
            instance = args[0]
            instance_workspace_root = instance.workspace_root.resolve()
            logger.debug(
                f"Validator using instance workspace: {instance_workspace_root}"
            )

            # Combine positional (excluding self) and keyword args for inspection
            passed_args_tuple = args[1:]
            passed_kwargs = kwargs.copy()

            # Locate the path argument (could be positional or keyword)
            sig = inspect.signature(func)
            path_str = None

            # Try kwargs first
            if arg_name in passed_kwargs:
                path_str = passed_kwargs[arg_name]
            else:
                # Try positional
                try:
                    bound_args = sig.bind_partial(*passed_args_tuple).arguments
                    if arg_name in bound_args:
                        path_str = bound_args[arg_name]
                except TypeError:
                    pass  # bind_partial fails if required args are missing

            # Use default if not found and default is provided
            if path_str is None and default_value is not None:
                path_str = default_value
                logger.debug(
                    f"Using default value '{default_value}' for path argument '{arg_name}'"
                )
            elif path_str is None:
                logger.error(
                    f"Path argument '{arg_name}' not found and no default value provided for {func.__name__}."
                )
                return {
                    "status": "error",
                    "action": "validator_error",
                    "data": {"message": f"Path argument '{arg_name}' missing."},
                }

            # Validate type after confirming path_str is found/defaulted
            if not isinstance(path_str, str) or not path_str:
                message = f"Path must be a non-empty string, but received: {path_str} ({type(path_str).__name__}) for argument '{arg_name}'"
                logger.warning(message)
                return {
                    "status": "error",
                    "action": action_name_on_error,
                    "data": {"message": message},
                }

            logger.debug(
                f"Validating path '{path_str}' for arg '{arg_name}' in function '{func.__name__}' relative to {instance_workspace_root}"
            )

            try:
                # --- Resolve Path ---
                path_obj = Path(path_str)

                # Prevent access to hidden files/dirs unless explicitly allowed
                if not allow_hidden and any(
                    part.startswith(".") and part not in [".", ".."]
                    for part in path_obj.parts
                ):
                    message = f"Access denied: Path '{path_str}' contains hidden components (starting with '.')."
                    logger.warning(message)
                    return {
                        "status": "error",
                        "action": action_name_on_error,
                        "data": {"message": message},
                    }

                # Resolve the path relative to instance workspace
                if path_obj.is_absolute():
                    try:
                        resolved_path = path_obj.resolve(strict=False)
                    except OSError as e:
                        logger.error(
                            f"Error resolving absolute path '{path_str}': {e}",
                            exc_info=True,
                        )
                        message = f"Error resolving absolute path '{path_str}'. It might be invalid or inaccessible. Error: {e}"
                        return {
                            "status": "error",
                            "action": action_name_on_error,
                            "data": {"message": message},
                        }
                else:
                    # Resolve relative paths from instance_workspace_root
                    try:
                        resolved_path = (instance_workspace_root / path_obj).resolve(
                            strict=False
                        )
                    except OSError as e:
                        logger.error(
                            f"Error resolving relative path '{path_str}' relative to {instance_workspace_root}: {e}",
                            exc_info=True,
                        )
                        message = f"Error resolving relative path '{path_str}'. It might be invalid or inaccessible. Error: {e}"
                        return {
                            "status": "error",
                            "action": action_name_on_error,
                            "data": {"message": message},
                        }

                # --- Workspace Check ---
                # Use the resolved instance workspace root for the check
                if not resolved_path.is_relative_to(instance_workspace_root):
                    message = f"Access denied: Path '{path_str}' resolves outside the designated workspace ('{instance_workspace_root}'). Resolved to '{resolved_path}'"
                    logger.warning(message)
                    return {
                        "status": "error",
                        "action": action_name_on_error,
                        "data": {"message": message},
                    }

                # --- Existence Check ---
                path_exists = resolved_path.exists()
                if check_existence and not path_exists:
                    message = f"Path not found: '{path_str}' (resolved to '{resolved_path}') does not exist."
                    logger.warning(message)
                    return {
                        "status": "error",
                        "action": action_name_on_error,
                        "data": {"message": message},
                    }

                # --- Type Check ---
                if path_exists:  # Only check type if it exists
                    is_file = resolved_path.is_file()
                    is_dir = resolved_path.is_dir()

                    if target_type == "file" and not is_file:
                        message = f"Path is not a file: '{path_str}' (resolved to '{resolved_path}') is not a valid file."
                        logger.warning(message)
                        return {
                            "status": "error",
                            "action": action_name_on_error,
                            "data": {"message": message},
                        }
                    elif target_type == "dir" and not is_dir:
                        message = f"Path is not a directory: '{path_str}' (resolved to '{resolved_path}') is not a valid directory."
                        logger.warning(message)
                        return {
                            "status": "error",
                            "action": action_name_on_error,
                            "data": {"message": message},
                        }
                elif not check_existence:
                    logger.debug(
                        f"Skipping type check for non-existent path '{path_str}' as check_existence=False."
                    )

                # --- Validation Success ---
                logger.debug(
                    f"Path validation successful for '{path_str}'. Resolved to: {resolved_path}"
                )

                # Prepare kwargs for the wrapped function
                func_kwargs = kwargs.copy()
                func_kwargs["resolved_path"] = resolved_path
                func_kwargs["original_path_str"] = path_str

                # Await the original async function call
                return await func(instance, **func_kwargs)  # Use await here

            except (ValueError, OSError) as e:
                logger.error(f"Error validating path '{path_str}': {e}", exc_info=True)
                message = f"Error processing path '{path_str}'. It might be invalid or inaccessible. Error: {e}"
                return {
                    "status": "error",
                    "action": action_name_on_error,
                    "data": {"message": message},
                }
            except Exception as e:
                logger.exception(
                    f"Unexpected error during path validation for '{path_str}':"
                )
                message = f"An unexpected error occurred while validating path '{path_str}' ({type(e).__name__})."
                return {
                    "status": "error",
                    "action": action_name_on_error,
                    "data": {"message": message},
                }

        return wrapper

    return decorator
