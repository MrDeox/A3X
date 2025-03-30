# skills/read_file.py
import logging
from pathlib import Path
import traceback # Keep for unexpected errors

# <<< IMPORT VALIDATOR >>>
from core.validators import validate_workspace_path

# Initialize logger
logger = logging.getLogger(__name__)

# <<< REMOVE WORKSPACE_ROOT definition and helper functions >>>
# WORKSPACE_ROOT = ...
# def _is_path_within_workspace(...)
# def _resolve_path(...)

# --- Core Reading Function (Adapted from manage_files.py) ---
# <<< MODIFIED: Now takes resolved_path directly >>>
def _read_file_content(resolved_path: Path, filepath_original: str, warning_msg: str | None = None) -> dict:
    """Internal function to read content from a resolved path."""
    # <<< REMOVED is_file check (done by decorator) >>>
    # if not resolved_path.is_file(): ...

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Limita o tamanho do conteúdo retornado na mensagem para não poluir logs/prompt
        max_len_preview = 500
        content_preview = content[:max_len_preview] + ("..." if len(content) > max_len_preview else "")

        return_data = {
            "filepath": filepath_original, # Retorna o path original solicitado
            "content": content, # Retorna o conteúdo completo
            "message": f"Conteúdo do arquivo '{filepath_original}' lido com sucesso (Prévia: {content_preview})"
        }
        if warning_msg:
            return_data["warning"] = warning_msg # Add warning if present

        return {
            "status": "success",
            "action": "file_read", # Consistent action name
            "data": return_data
        }
    except FileNotFoundError: # Should be caught by decorator, but for safety
        return {"status": "error", "action": "read_file_failed", "data": {"message": f"Arquivo não encontrado: '{filepath_original}'"}}
    except PermissionError:
        logger.error(f"Erro de permissão ao ler arquivo: {filepath_original}", exc_info=True)
        return {"status": "error", "action": "read_file_failed", "data": {"message": f"Permissão negada para ler o arquivo: '{filepath_original}'"}}
    except IsADirectoryError:
         return {"status": "error", "action": "read_file_failed", "data": {"message": f"O caminho fornecido é um diretório, não um arquivo: '{filepath_original}'"}}
    except Exception as e:
        logger.exception(f"Erro inesperado ao ler arquivo '{filepath_original}':")
        return {"status": "error", "action": "read_file_failed", "data": {"message": f"Erro inesperado ao ler arquivo '{filepath_original}': {e}"}}

# --- Skill Function ---
# <<< ADDED Decorator >>>
@validate_workspace_path(
    # Try resolving common names first, then specific
    arg_name='file_name', # Primary argument name expected by the skill
    # Fallback aliases if arg_name not found directly in action_input:
    # aliases=['file_path'], # Decorator now handles aliases internally
    check_existence=True, # File must exist
    target_type='file', # Must be a file
    allow_hidden=False, # Do not allow reading hidden files
    action_name_on_error="path_validation_failed"
)
def skill_read_file(action_input: dict, resolved_path: Path = None, original_path_str: str = None, agent_memory: dict = None, agent_history: list | None = None) -> dict:
    """
    Lê e retorna TODO o conteúdo de um arquivo de texto especificado (limite 1MB).
    Formatos suportados: .txt, .py, .md, .json, .env, .csv, .log
    Relies on @validate_workspace_path decorator for path handling.

    Args:
        action_input (dict): Original action input dictionary.
        resolved_path (Path, injected): Validated absolute Path object.
        original_path_str (str, injected): Original path string requested.
        agent_memory (dict, optional): Agent's memory (not used). Defaults to None.
        agent_history (list | None, optional): Histórico da conversa (não usado). Defaults to None.

    Returns:
        dict: Dicionário padronizado com o resultado da operação:
              {"status": "success/error", "action": "file_read/read_file_failed",
               "data": {"message": "...", "filepath": "...", "content": "...", "warning": "..."(optional)}}
    """
    logger.debug(f"Skill 'read_file' executada para: '{original_path_str}' (resolved: {resolved_path})")

    # <<< REMOVED manual path resolution and checks (done by decorator) >>>
    # filepath = action_input.get("file_name") or action_input.get("file_path")
    # if not filepath: ...
    # resolved_path = _resolve_path(filepath)
    # if not resolved_path: ...

    if not resolved_path: # Safeguard check
         logger.error("Decorator failed to inject resolved_path into skill_read_file.")
         return {"status": "error", "action": "read_file_failed", "data": {"message": "Internal error: Path validation failed unexpectedly."}}

    # --- Checkers Adicionais (aplicados ao resolved_path validado) ---
    # 1. Verifica extensão suportada
    TEXT_EXTENSIONS = {".txt", ".py", ".md", ".json", ".env", ".csv", ".log"}
    file_ext = resolved_path.suffix.lower()

    if file_ext not in TEXT_EXTENSIONS:
        logger.warning(f"Tentativa de leitura de extensão não suportada: {file_ext} em '{original_path_str}'")
        return {
            "status": "error",
            "action": "read_file_failed_unsupported_ext",
            "data": {"message": f"Extensão '{file_ext}' não suportada. Use arquivos de texto: {', '.join(TEXT_EXTENSIONS)}"}
        }

    # 2. Limita tamanho do arquivo (1MB)
    MAX_SIZE = 1 * 1024 * 1024  # 1MB
    try:
        file_size = resolved_path.stat().st_size
        if file_size > MAX_SIZE:
            logger.warning(f"Tentativa de leitura de arquivo muito grande: {file_size} bytes em '{original_path_str}'")
            return {
                "status": "error",
                "action": "read_file_failed_too_large",
                "data": {"message": f"Arquivo muito grande ({file_size / (1024*1024):.2f} MB). Limite: {MAX_SIZE / (1024*1024):.0f} MB."}
            }
    except FileNotFoundError: # Should be caught by decorator, but safety first
         return {"status": "error", "action": "read_file_failed", "data": {"message": f"Erro ao verificar tamanho: Arquivo não encontrado em '{original_path_str}' (inesperado)."}}
    except OSError as e:
        logger.error(f"Erro OSError ao verificar tamanho de '{original_path_str}': {e}", exc_info=True)
        return {"status": "error", "action": "read_file_failed_stat_error", "data": {"message": f"Erro ao verificar tamanho do arquivo: {e}"}}

    # 3. Warning para extensões "arriscadas" (opcional)
    WARNING_EXTS = {".json", ".env", ".csv"}
    warning_msg = f"Atenção: '{file_ext}' pode conter dados estruturados ou sensíveis. Leia com cuidado." if file_ext in WARNING_EXTS else None
    # --- Fim Checkers ---

    # Chama a função interna passando o warning e o path original para mensagens
    return _read_file_content(resolved_path, original_path_str, warning_msg)

