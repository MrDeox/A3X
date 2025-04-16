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
            logger = logging.getLogger(__name__)
            logger.debug(f"[DEBUG validate_workspace_path] Entrou no wrapper do decorador para {func.__name__} com args={args}, kwargs={kwargs}")
            print(f"[DEBUG validate_workspace_path] Entrou no wrapper do decorador para {func.__name__} com args={args}, kwargs={kwargs}")

            instance = None
            ctx = None

            # --- Try to find instance and ctx ---
            # Prioritize positional arguments based on common pattern (instance, ctx, ...)
            if len(args) >= 1:
                instance = args[0] # Assume first arg is the instance (e.g., 'self')
                logger.debug(f"[DEBUG validate_workspace_path] Found potential instance in args[0]: {type(instance)}")
                print(f"[DEBUG validate_workspace_path] Found potential instance in args[0]: {type(instance)}")
            if len(args) >= 2:
                 # Check if second arg could be the context object
                 # A simple check could be duck typing for a known attribute like 'shared_context'
                 potential_ctx = args[1]
                 if hasattr(potential_ctx, 'shared_context'): # Simple check for ToolExecutionContext
                    ctx = potential_ctx
                    logger.debug(f"[DEBUG validate_workspace_path] Found potential ctx in args[1]: {type(ctx)}")
                    print(f"[DEBUG validate_workspace_path] Found potential ctx in args[1]: {type(ctx)}")

            # Fallback to keyword arguments if not found positionally
            if instance is None and 'self' in kwargs: # Common kwarg name for instance
                instance = kwargs['self']
                logger.debug(f"[DEBUG validate_workspace_path] Found potential instance in kwargs['self']: {type(instance)}")
                print(f"[DEBUG validate_workspace_path] Found potential instance in kwargs['self']: {type(instance)}")

            if ctx is None and 'ctx' in kwargs:
                ctx = kwargs['ctx']
                logger.debug(f"[DEBUG validate_workspace_path] Found potential ctx in kwargs['ctx']: {type(ctx)}")
                print(f"[DEBUG validate_workspace_path] Found potential ctx in kwargs['ctx']: {type(ctx)}")

            # --- Get workspace_root from instance ---
            instance_workspace_root = None
            if instance and hasattr(instance, 'workspace_root') and instance.workspace_root:
                instance_workspace_root = Path(instance.workspace_root).resolve()
                logger.debug(f"[DEBUG validate_workspace_path] Validator using workspace from instance: {instance_workspace_root}")
                print(f"[DEBUG validate_workspace_path] Validator using workspace from instance: {instance_workspace_root}")
            elif instance:
                 logger.error(f"[DEBUG validate_workspace_path] Validator found instance {type(instance)} but it lacks 'workspace_root'.")
                 print(f"[DEBUG validate_workspace_path] Validator found instance {type(instance)} but it lacks 'workspace_root'.")
                 # Decide if this is an error or if workspace comes only from ctx
                 # For now, let's rely on ctx if instance lacks it.

            # --- Get workspace_root from ctx if not found on instance or if instance is None ---
            # This part might need adjustment based on where workspace_root SHOULD live.
            # Currently, the logic relies *only* on the instance having it.
            # Let's stick to the original logic: workspace MUST be on the instance.
            if instance_workspace_root is None:
                 logger.error(f"[DEBUG validate_workspace_path] Could not determine workspace_root from instance for {func.__name__}.")
                 print(f"[DEBUG validate_workspace_path] Could not determine workspace_root from instance for {func.__name__}.")
                 # Log details about instance and ctx for debugging
                 if instance:
                     logger.error(f"[DEBUG validate_workspace_path] Instance ({type(instance)}) found but missing 'workspace_root'. Attributes: {dir(instance)}")
                     print(f"[DEBUG validate_workspace_path] Instance ({type(instance)}) found but missing 'workspace_root'. Attributes: {dir(instance)}")
                 else:
                     logger.error(f"[DEBUG validate_workspace_path] Instance not found.")
                     print(f"[DEBUG validate_workspace_path] Instance not found.")
                 if ctx:
                     logger.error(f"[DEBUG validate_workspace_path] Context ({type(ctx)}) found. Attributes: {dir(ctx)}")
                     print(f"[DEBUG validate_workspace_path] Context ({type(ctx)}) found. Attributes: {dir(ctx)}")
                 else:
                      logger.error(f"[DEBUG validate_workspace_path] Context not found.")
                      print(f"[DEBUG validate_workspace_path] Context not found.")

                 return {
                    "status": "error",
                    "action": "validator_error",
                    "data": {"message": "Internal validator error: Could not determine workspace_root from skill instance."}
                }

            # --- Proceed with path validation using instance_workspace_root ---
            # (Rest of the original logic, adjusted to use instance_workspace_root)

            if instance is None: # Should be caught by workspace root check now, but belt-and-suspenders
                 logger.critical(f"[DEBUG validate_workspace_path] Validator chegou na resolução de path mas instance está None para {func.__name__}. Abortando.")
                 print(f"[DEBUG validate_workspace_path] Validator chegou na resolução de path mas instance está None para {func.__name__}. Abortando.")
                 return {"status": "error", "action": "validator_error", "data": {"message": "Internal validator error: instance not found before path resolution."} }

            passed_args_tuple = args
            passed_kwargs = kwargs.copy()
            sig = inspect.signature(func)
            path_str = None
            path_arg_found_in = None # Track where we found it

            # Try to find the path argument (arg_name)
            # 1. Check kwargs directly
            if arg_name in passed_kwargs:
                path_str = passed_kwargs[arg_name]
                path_arg_found_in = "kwargs"
            else:
                # 2. Check bound positional/keyword arguments
                try:
                    # Bind *excluding* instance and ctx if they were positional,
                    # as the original function signature won't have them listed explicitly usually.
                    # Or better: bind all, then extract.
                    bound_args = sig.bind_partial(*passed_args_tuple, **passed_kwargs).arguments
                    if arg_name in bound_args:
                        path_str = bound_args[arg_name]
                        path_arg_found_in = "bound_args"
                    else:
                         # Check if arg_name is VAR_POSITIONAL or VAR_KEYWORD - less common for paths
                         pass # Simplified for now

                except TypeError as e: # Mismatched arguments
                    logger.error(f"[DEBUG validate_workspace_path] TypeError binding args for {func.__name__} ({arg_name}): {e}. Args: {args}, Kwargs: {kwargs}")
                    print(f"[DEBUG validate_workspace_path] TypeError binding args for {func.__name__} ({arg_name}): {e}. Args: {args}, Kwargs: {kwargs}")
                    # This might indicate instance/ctx were incorrectly identified or passed
                    return {"status": "error", "action": action_name_on_error, "data": {"message": f"Argument binding error for {func.__name__}: {e}"}}
                except Exception as e:
                    logger.error(f"[DEBUG validate_workspace_path] Erro ao tentar localizar argumento de caminho \'{arg_name}\' via bind_partial: {e}")
                    print(f"[DEBUG validate_workspace_path] Erro ao tentar localizar argumento de caminho \'{arg_name}\' via bind_partial: {e}")
                    # Fall through to check default value

            if path_str is None:
                if default_value is not None:
                    path_str = default_value
                    path_arg_found_in = "default"
                    logger.warning(f"[DEBUG validate_workspace_path] Usando valor default para \'{arg_name}\': {default_value}")
                    print(f"[DEBUG validate_workspace_path] Usando valor default para \'{arg_name}\': {default_value}")
                else:
                    logger.error(f"[DEBUG validate_workspace_path] Não foi possível encontrar argumento de caminho \'{arg_name}\' em {func.__name__}. args={args}, kwargs={kwargs}")
                    print(f"[DEBUG validate_workspace_path] Não foi possível encontrar argumento de caminho \'{arg_name}\' em {func.__name__}. args={args}, kwargs={kwargs}")
                    return {"status": "error", "action": action_name_on_error, "data": {"message": f"Missing required path argument \'{arg_name}\'."}}

            # Ensure path_str is a string
            if not isinstance(path_str, str):
                 logger.error(f"[DEBUG validate_workspace_path] Argumento de caminho \'{arg_name}\' não é uma string: {type(path_str)} ({path_str}). Found in: {path_arg_found_in}")
                 print(f"[DEBUG validate_workspace_path] Argumento de caminho \'{arg_name}\' não é uma string: {type(path_str)} ({path_str}). Found in: {path_arg_found_in}")
                 return {"status": "error", "action": action_name_on_error, "data": {"message": f"Path argument \'{arg_name}\' must be a string, got {type(path_str).__name__}."}}


            # Resolve path using the confirmed instance_workspace_root
            try:
                # Ensure path_str is treated as relative to the workspace root
                # Avoid joining absolute paths
                path_to_join = Path(path_str)
                if path_to_join.is_absolute():
                    logger.error(f"[DEBUG validate_workspace_path] Path must be relative, but got absolute path: {path_str}")
                    print(f"[DEBUG validate_workspace_path] Path must be relative, but got absolute path: {path_str}")
                    return {"status": "error", "action": action_name_on_error, "data": {"message": f"Path must be relative to the workspace, got absolute path: '{path_str}'"}}

                resolved_path = (instance_workspace_root / path_to_join).resolve()
                logger.debug(f"[DEBUG validate_workspace_path] resolved_path calculado: {resolved_path} (from workspace {instance_workspace_root} and input {path_str})")
                print(f"[DEBUG validate_workspace_path] resolved_path calculado: {resolved_path} (from workspace {instance_workspace_root} and input {path_str})")

                # Security check: Ensure resolved path is still within the workspace
                if instance_workspace_root not in resolved_path.parents and resolved_path != instance_workspace_root:
                     logger.error(f"[DEBUG validate_workspace_path] Path traversal detected! Resolved path '{resolved_path}' is outside workspace '{instance_workspace_root}'. Input: '{path_str}'")
                     print(f"[DEBUG validate_workspace_path] Path traversal detected! Resolved path '{resolved_path}' is outside workspace '{instance_workspace_root}'. Input: '{path_str}'")
                     return {"status": "error", "action": action_name_on_error, "data": {"message": f"Path resolution resulted in access outside the allowed workspace: '{path_str}'"}}

                # Check for hidden files/dirs
                relative_path_parts = resolved_path.relative_to(instance_workspace_root).parts
                if not allow_hidden and any(part.startswith('.') for part in relative_path_parts):
                    logger.error(f"[DEBUG validate_workspace_path] Caminho contém componentes ocultos: {resolved_path}")
                    print(f"[DEBUG validate_workspace_path] Caminho contém componentes ocultos: {resolved_path}")
                    return {"status": "error", "action": action_name_on_error, "data": {"message": f"Hidden files or directories are not allowed in path: '{path_str}'"}}

                # Existence/type checks
                path_exists = resolved_path.exists()
                if check_existence and not path_exists:
                    logger.warning(f"[DEBUG validate_workspace_path] Caminho não existe: {resolved_path} (checking existence for '{path_str}')")
                    print(f"[DEBUG validate_workspace_path] Caminho não existe: {resolved_path} (checking existence for '{path_str}')")
                    # Allow non-existence for write operations usually, maybe decorator needs config?
                    # For now, strictly follow check_existence flag
                    return {"status": "error", "action": action_name_on_error, "data": {"message": f"Path not found: '{path_str}' (resolved to '{resolved_path}') does not exist."}}

                if path_exists: # Only check type if it exists
                    if target_type == "file" and not resolved_path.is_file():
                        logger.error(f"[DEBUG validate_workspace_path] Esperado arquivo mas é diretório: {resolved_path}")
                        print(f"[DEBUG validate_workspace_path] Esperado arquivo mas é diretório: {resolved_path}")
                        return {"status": "error", "action": action_name_on_error, "data": {"message": f"Path '{path_str}' (resolved to '{resolved_path}') is not a file."}}
                    if target_type == "dir" and not resolved_path.is_dir():
                        logger.error(f"[DEBUG validate_workspace_path] Esperado diretório mas é arquivo: {resolved_path}")
                        print(f"[DEBUG validate_workspace_path] Esperado diretório mas é arquivo: {resolved_path}")
                        return {"status": "error", "action": action_name_on_error, "data": {"message": f"Path '{path_str}' (resolved to '{resolved_path}') is not a directory."}}

                # --- Inject paths into kwargs for the decorated function ---
                # Remove original path arg if it was in kwargs to avoid duplication if name clashes
                # with injected names, though unlikely.
                # if path_arg_found_in == "kwargs" and arg_name in passed_kwargs:
                #     del passed_kwargs[arg_name] # Be careful if func relies on it by name

                # Inject resolved Path object and original string path
                passed_kwargs['resolved_path'] = resolved_path
                passed_kwargs['original_path_str'] = path_str # Keep the original path string available if needed


                # Adjust args tuple to remove instance/ctx if they were positional
                # so they are not passed twice (once positionally, once via kwargs if bound)
                final_args = args
                if instance is args[0] and ctx is args[1] and len(args) >= 2:
                    final_args = args[2:] # Pass only remaining positional args
                elif instance is args[0] and len(args) >= 1:
                    final_args = args[1:]


                logger.debug(f"[DEBUG validate_workspace_path] Chamando {func.__name__} com args={final_args} e kwargs={passed_kwargs}")
                print(f"[DEBUG validate_workspace_path] Chamando {func.__name__} com args={final_args} e kwargs={passed_kwargs}")

                # Call the original async function
                result = await func(*final_args, **passed_kwargs)

                logger.debug(f"[DEBUG validate_workspace_path] Saída do método {func.__name__}: {result}")
                print(f"[DEBUG validate_workspace_path] Saída do método {func.__name__}: {result}")
                return result
            except ValueError as e: # Catches Path resolution errors like invalid chars
                logger.error(f"[DEBUG validate_workspace_path] Erro ao resolver path \'{path_str}\': {e}", exc_info=True)
                print(f"[DEBUG validate_workspace_path] Erro ao resolver path \'{path_str}\': {e}")
                return {"status": "error", "action": action_name_on_error, "data": {"message": f"Invalid path format '{path_str}': {e}"}}
            except Exception as e:
                logger.error(f"[DEBUG validate_workspace_path] Exceção inesperada em {func.__name__} durante validação/execução: {e}", exc_info=True)
                print(f"[DEBUG validate_workspace_path] Exceção inesperada em {func.__name__} durante validação/execução: {e}")
                return {"status": "error", "action": action_name_on_error, "data": {"message": f"An unexpected error occurred while validating/executing with path '{path_str}' ({type(e).__name__})."}}
        return wrapper
    return decorator
