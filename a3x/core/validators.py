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
    Decorator for skill methods that validates a path argument.

    Ensures the path is within the agent's workspace, optionally checks
    existence and type (file/dir), and prevents accessing hidden files/dirs
    unless explicitly allowed. Resolves the path relative to the workspace root
    obtained from the execution context (`ctx`).

    Injects 'resolved_path' (Path object) and 'original_path_str' into the
    decorated function's kwargs.

    Assumes the decorated function is an async method of a class where the
    first argument is the instance (`self`) and there's a `ctx` argument
    of type `_ToolExecutionContext`.
    """
    if not isinstance(arg_name, str) or not arg_name:
        raise TypeError("`arg_name` must be a non-empty string")

    def decorator(func: Callable) -> Callable:
        if not inspect.iscoroutinefunction(func):
             raise TypeError(f"Function {func.__name__} must be an async function to be decorated by validate_workspace_path.")

        # Get the signature of the decorated function once
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        # Heuristic: Assume 'self' or the first param is the instance
        instance_param_name = param_names[0] if param_names else None
        # Assume 'ctx' is the name for the context parameter
        ctx_param_name = 'ctx'
        if ctx_param_name not in sig.parameters:
             logger.warning(f"Function {func.__name__} decorated by validate_workspace_path does not seem to have a '{ctx_param_name}' parameter. Path validation might fail.")
             # Attempt to proceed, but it's likely to fail later if ctx is needed

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:  # Make wrapper async
            bound_args = None
            try:
                # Bind arguments to handle both positional and keyword args correctly
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults() # Apply defaults if any args were missing

                arguments = bound_args.arguments
                passed_args = bound_args.args
                passed_kwargs = bound_args.kwargs

                logger.debug(f"[DEBUG validate_workspace_path] Args recebidos: args={args}, kwargs={kwargs}")
                logger.debug(f"[DEBUG validate_workspace_path] Args vinculados: {arguments}")
                print(f"[DEBUG validate_workspace_path] Args recebidos: args={args}, kwargs={kwargs}")
                print(f"[DEBUG validate_workspace_path] Args vinculados: {arguments}")


                # --- Find instance and context reliably ---
                instance = arguments.get(instance_param_name) if instance_param_name else None
                ctx: _ToolExecutionContext | None = arguments.get(ctx_param_name)

                if not ctx or not hasattr(ctx, 'workspace_root') or not ctx.workspace_root:
                    logger.error(f"[DEBUG validate_workspace_path] Contexto (`{ctx_param_name}`) inválido ou sem `workspace_root` para {func.__name__}.")
                    print(f"[DEBUG validate_workspace_path] Contexto (`{ctx_param_name}`) inválido ou sem `workspace_root` para {func.__name__}.")

                    return {"status": "error", "action": action_name_on_error, "data": {"message": f"Execution context with workspace_root is missing for path validation in {func.__name__}."}}

                # --- Extract path string ---
                path_str = arguments.get(arg_name)
                path_arg_found_in = "arguments" # Since we use bound_args.arguments

                if path_str is None:
                    if default_value is not None:
                        path_str = default_value
                        logger.debug(f"[DEBUG validate_workspace_path] Usando valor default para '{arg_name}': '{default_value}'")
                        print(f"[DEBUG validate_workspace_path] Usando valor default para '{arg_name}': '{default_value}'")

                        # Add default value back into arguments if it wasn't present
                        arguments[arg_name] = path_str
                        # Also update passed_kwargs in case the function expects it there
                        # This depends on how bind reorganizes args/kwargs, safer to update arguments dict
                        # which will be used for the final call.
                        passed_kwargs[arg_name] = path_str # Add/overwrite in kwargs for the call

                    else:
                        logger.error(f"[DEBUG validate_workspace_path] Argumento de path obrigatório '{arg_name}' não encontrado em {func.__name__}.")
                        print(f"[DEBUG validate_workspace_path] Argumento de path obrigatório '{arg_name}' não encontrado em {func.__name__}.")
                        return {"status": "error", "action": action_name_on_error, "data": {"message": f"Required path argument '{arg_name}' is missing."}}

                if not isinstance(path_str, str) or not path_str:
                    logger.error(f"[DEBUG validate_workspace_path] Argumento de path '{arg_name}' deve ser uma string não vazia, recebido: {path_str} ({type(path_str).__name__})")
                    print(f"[DEBUG validate_workspace_path] Argumento de path '{arg_name}' deve ser uma string não vazia, recebido: {path_str} ({type(path_str).__name__})")
                    return {"status": "error", "action": action_name_on_error, "data": {"message": f"Path argument '{arg_name}' must be a non-empty string."}}

                # --- Validate and Resolve Path ---
                workspace_root = Path(ctx.workspace_root).resolve()
                target_path = Path(path_str)

                # Prevent absolute paths or paths trying to escape the workspace
                if target_path.is_absolute():
                    logger.error(f"[DEBUG validate_workspace_path] Caminho absoluto não permitido: {target_path}")
                    print(f"[DEBUG validate_workspace_path] Caminho absoluto não permitido: {target_path}")

                    return {"status": "error", "action": action_name_on_error, "data": {"message": f"Absolute paths are not allowed: '{path_str}'."}}

                # Resolve the path relative to the workspace root
                # Use resolve() AFTER joining to correctly handle '..' sequences within the workspace context
                resolved_path = (workspace_root / target_path).resolve()

                logger.debug(f"[DEBUG validate_workspace_path] Validando: '{path_str}' -> '{resolved_path}' (Workspace: '{workspace_root}')")
                print(f"[DEBUG validate_workspace_path] Validando: '{path_str}' -> '{resolved_path}' (Workspace: '{workspace_root}')")


                # Security Check: Ensure the resolved path is still within the workspace
                # This check prevents escaping via sequences like 'subdir/../../outside'
                # Note: Path.is_relative_to is available in Python 3.9+
                try:
                    if not resolved_path.relative_to(workspace_root):
                         # This case should theoretically not be reached if resolve() works correctly
                         # after joining with an absolute workspace_root, but added for safety.
                         raise ValueError("Path resolution resulted outside workspace.")
                except ValueError: # Handles cases where relative_to fails (e.g., different drive on Windows)
                    logger.error(f"[DEBUG validate_workspace_path] Path escapou do workspace: '{resolved_path}' (Workspace: '{workspace_root}')")
                    print(f"[DEBUG validate_workspace_path] Path escapou do workspace: '{resolved_path}' (Workspace: '{workspace_root}')")
                    return {"status": "error", "action": action_name_on_error, "data": {"message": f"Path '{path_str}' attempts to access outside the allowed workspace."}}


                # Check for hidden files/directories (starting with '.')
                # Check the original *relative* path components for hidden parts
                if not allow_hidden and any(part.startswith('.') for part in target_path.parts):
                    logger.error(f"[DEBUG validate_workspace_path] Acesso a path oculto não permitido: {target_path}")
                    print(f"[DEBUG validate_workspace_path] Acesso a path oculto não permitido: {target_path}")

                    return {"status": "error", "action": action_name_on_error, "data": {"message": f"Access to hidden path '{path_str}' is not allowed."}}

                # --- Existence and Type Check ---
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
                # Use the bound arguments dictionary to pass modified/added args correctly
                final_call_kwargs = arguments.copy() # Start with all bound arguments
                # Inject the resolved path and original string path as separate arguments
                final_call_kwargs['resolved_path'] = resolved_path
                final_call_kwargs['original_path_str'] = path_str # path_str holds the original value

                # Remove original instance and ctx from final kwargs if they were passed positionally
                # This is now implicitly handled by calling func(**final_call_kwargs)

                # Call the original async function with potentially modified args/kwargs
                # Passing **final_call_kwargs should work if all params can be passed by keyword
                logger.debug(f"[DEBUG validate_workspace_path] Chamando {func.__name__} com argumentos injetados: resolved_path='{resolved_path}', original_path_str='{path_str}'")
                print(f"[DEBUG validate_workspace_path] Chamando {func.__name__} com argumentos injetados: resolved_path='{resolved_path}', original_path_str='{path_str}'")
                result = await func(**final_call_kwargs)


                # Alternative if positional args are strictly needed:
                # final_args_tuple = tuple(final_call_kwargs[p] for p in sig.parameters if sig.parameters[p].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or sig.parameters[p].kind == inspect.Parameter.POSITIONAL_ONLY)
                # final_kwargs_dict = {k: v for k, v in final_call_kwargs.items() if sig.parameters[k].kind == inspect.Parameter.KEYWORD_ONLY or sig.parameters[k].kind == inspect.Parameter.VAR_KEYWORD}
                # logger.debug(f"[DEBUG validate_workspace_path] Chamando {func.__name__} com args={final_args_tuple} e kwargs={final_kwargs_dict}")
                # print(f"[DEBUG validate_workspace_path] Chamando {func.__name__} com args={final_args_tuple} e kwargs={final_kwargs_dict}")
                # result = await func(*final_args_tuple, **final_kwargs_dict)


                logger.debug(f"[DEBUG validate_workspace_path] Saída do método {func.__name__}: {result}")
                print(f"[DEBUG validate_workspace_path] Saída do método {func.__name__}: {result}")
                return result
            except TypeError as e: # Catch binding errors
                 logger.error(f"[DEBUG validate_workspace_path] Erro ao vincular argumentos para {func.__name__}: {e}. Args: {args}, Kwargs: {kwargs}", exc_info=True)
                 print(f"[DEBUG validate_workspace_path] Erro ao vincular argumentos para {func.__name__}: {e}. Args: {args}, Kwargs: {kwargs}")
                 return {"status": "error", "action": action_name_on_error, "data": {"message": f"Type error during argument binding for {func.__name__}: {e}"}}
            except ValueError as e: # Catches Path resolution errors like invalid chars OR binding errors
                # Check if it's a binding error or path error
                if bound_args is None: # Error happened during bind()
                     logger.error(f"[DEBUG validate_workspace_path] Erro ao vincular argumentos para {func.__name__}: {e}. Args: {args}, Kwargs: {kwargs}", exc_info=True)
                     print(f"[DEBUG validate_workspace_path] Erro ao vincular argumentos para {func.__name__}: {e}. Args: {args}, Kwargs: {kwargs}")
                     return {"status": "error", "action": action_name_on_error, "data": {"message": f"Argument binding error for {func.__name__}: {e}"}}
                else: # Error likely happened during path resolution/validation
                    # path_str might not be defined if error happened before extraction
                    current_path_str = arguments.get(arg_name, "<path not extracted>") if arguments else "<arguments not bound>"
                    logger.error(f"[DEBUG validate_workspace_path] Erro de valor (possivelmente path inválido) para '{current_path_str}': {e}", exc_info=True)
                    print(f"[DEBUG validate_workspace_path] Erro de valor (possivelmente path inválido) para '{current_path_str}': {e}")
                    return {"status": "error", "action": action_name_on_error, "data": {"message": f"Invalid path format or value error for path '{current_path_str}': {e}"}}
            except Exception as e:
                # path_str might not be defined if error happened before extraction
                current_path_str = arguments.get(arg_name, "<path not extracted>") if arguments else "<arguments not bound>"
                logger.error(f"[DEBUG validate_workspace_path] Exceção inesperada em {func.__name__} durante validação/execução com path '{current_path_str}': {e}", exc_info=True)
                print(f"[DEBUG validate_workspace_path] Exceção inesperada em {func.__name__} durante validação/execução com path '{current_path_str}': {e}")
                return {"status": "error", "action": action_name_on_error, "data": {"message": f"An unexpected error occurred while validating/executing with path '{current_path_str}' ({type(e).__name__})."}}
        return wrapper
    return decorator
