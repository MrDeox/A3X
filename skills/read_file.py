# skills/read_file.py
import os
import logging
from pathlib import Path
import traceback # Keep for unexpected errors

# Initialize logger
logger = logging.getLogger(__name__)

# Define o diretório raiz do projeto (DEVE SER CONSISTENTE COM OUTROS MÓDULOS)
# Idealmente, viria de core.config, mas para desacoplar, definimos aqui temporariamente.
# Cuidado: Se mudar em config, tem que mudar aqui também!
try:
    # Tenta importar do config primeiro
    from core.config import PROJECT_ROOT as WORKSPACE_ROOT_STR
    WORKSPACE_ROOT = Path(WORKSPACE_ROOT_STR).resolve()
    logger.debug(f"read_file using WORKSPACE_ROOT from core.config: {WORKSPACE_ROOT}")
except ImportError:
    logger.warning("Could not import PROJECT_ROOT from core.config. Using hardcoded path relative to skill.")
    # Fallback to assuming this file is in skills/ and root is one level up
    skills_dir = Path(__file__).parent
    WORKSPACE_ROOT = skills_dir.parent.resolve()
    logger.debug(f"read_file using calculated WORKSPACE_ROOT: {WORKSPACE_ROOT}")


# --- Helper Functions (Copied from manage_files.py) ---

def _is_path_within_workspace(path: str | Path) -> bool:
    """Verifica se o caminho fornecido está dentro do WORKSPACE_ROOT."""
    try:
        # Resolve symlinks etc., and get the absolute path
        absolute_path = Path(path).resolve(strict=True) # strict=True to raise FileNotFoundError if path doesn't exist
        # Check if the resolved absolute path is relative to the WORKSPACE_ROOT
        return absolute_path.is_relative_to(WORKSPACE_ROOT)
    except FileNotFoundError:
         logger.warning(f"Path not found during workspace check: {path}")
         return False
    except ValueError: # Handle cases where the path is invalid or outside the root
        logger.warning(f"Path '{path}' is outside the defined WORKSPACE_ROOT.")
        return False
    except Exception as e: # Catch other potential errors like permission issues
        logger.error(f"Error checking path '{path}' against workspace: {e}")
        return False

def _resolve_path(filepath: str) -> Path | None:
    """Resolve o filepath para um caminho absoluto seguro dentro do workspace."""
    path = Path(filepath)
    # Prevent accessing hidden files/directories (basic check)
    if any(part.startswith('.') for part in path.parts):
         logger.warning(f"Acesso negado: Tentativa de acessar caminho oculto: {filepath}")
         return None
         
    if path.is_absolute():
        # Absolute paths are checked directly if they are within the workspace
        if _is_path_within_workspace(path):
            return path.resolve()
        else:
            logger.warning(f"Acesso negado: Caminho absoluto fora do workspace: {path}")
            return None
    else:
        # Resolve relative paths based on the WORKSPACE_ROOT
        resolved_path = (WORKSPACE_ROOT / path).resolve()
        # Double-check the resolved path is still within the workspace
        # This prevents tricks like ../../etc/passwd
        if _is_path_within_workspace(resolved_path):
            return resolved_path
        else:
            logger.warning(f"Acesso negado: Caminho relativo resolvido fora do workspace: {filepath} -> {resolved_path}")
            return None

# --- Core Reading Function (Adapted from manage_files.py) ---

def _read_file_content(resolved_path: Path, filepath_original: str, warning_msg: str | None = None) -> dict:
    """Internal function to read content from a resolved path."""
    try:
        if not resolved_path.is_file():
            return {"status": "error", "action": "read_file_failed", "data": {"message": f"Arquivo não encontrado ou não é um arquivo: '{filepath_original}'"}}

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
    except FileNotFoundError: # Should be caught by is_file(), but for safety
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

def skill_read_file(action_input: dict, agent_memory: dict = None, agent_history: list | None = None) -> dict:
    """
    Lê e retorna TODO o conteúdo de um arquivo de texto especificado (limite 1MB).
    Formatos suportados: .txt, .py, .md, .json, .env, .csv, .log

    Args:
        action_input (dict): Dicionário contendo:
            - file_name (str): O nome do arquivo a ser lido (preferencial).
            - file_path (str): Sinônimo alternativo para o nome do arquivo.
        agent_memory (dict, optional): Memória do agente (não usada). Defaults to None.
        agent_history (list | None, optional): Histórico da conversa (não usado). Defaults to None.

    Returns:
        dict: Dicionário padronizado com o resultado da operação:
              {"status": "success/error", "action": "file_read/read_file_failed",
               "data": {"message": "...", "filepath": "...", "content": "...", "warning": "..."(optional)}}
    """
    logger.debug(f"Skill 'read_file' executada com input: {action_input}")
    
    # Fallback tolerante: aceita "file_name" OU "file_path"
    filepath = action_input.get("file_name") or action_input.get("file_path")
    
    # Log warning if the non-preferred name was used
    if filepath and "file_path" in action_input and "file_name" not in action_input:
        logger.warning(f"LLM usou 'file_path' em vez do preferido 'file_name'. Input: {action_input}")

    if not filepath:
        return {"status": "error", "action": "read_file_failed", "data": {"message": "Parâmetro obrigatório ausente. Use 'file_name' ou 'file_path'."}} # Updated error message
    if not isinstance(filepath, str):
         # Error message remains relevant as it applies to the resolved value
         return {"status": "error", "action": "read_file_failed", "data": {"message": f"Parâmetro de caminho inválido (tipo {type(filepath)}). Esperado: string."}}

    resolved_path = _resolve_path(filepath)
    if not resolved_path:
        # _resolve_path already logged the warning
        return {"status": "error", "action": "read_file_failed", "data": {"message": f"Acesso negado ou caminho inválido: '{filepath}'"}}

    # --- Checkers Adicionais ---
    # 1. Verifica extensão suportada
    TEXT_EXTENSIONS = {".txt", ".py", ".md", ".json", ".env", ".csv", ".log"}
    file_ext = resolved_path.suffix.lower() # Get extension from resolved Path object

    if file_ext not in TEXT_EXTENSIONS:
        logger.warning(f"Tentativa de leitura de extensão não suportada: {file_ext} em '{filepath}'")
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
            logger.warning(f"Tentativa de leitura de arquivo muito grande: {file_size} bytes em '{filepath}'")
            return {
                "status": "error",
                "action": "read_file_failed_too_large",
                "data": {"message": f"Arquivo muito grande ({file_size / (1024*1024):.2f} MB). Limite: {MAX_SIZE / (1024*1024):.0f} MB."}
            }
    except FileNotFoundError: # Should not happen if _resolve_path worked, but safety first
         return {"status": "error", "action": "read_file_failed", "data": {"message": f"Erro ao verificar tamanho: Arquivo não encontrado em '{filepath}' (inesperado)."}}
    except OSError as e:
        logger.error(f"Erro OSError ao verificar tamanho de '{filepath}': {e}", exc_info=True)
        return {"status": "error", "action": "read_file_failed_stat_error", "data": {"message": f"Erro ao verificar tamanho do arquivo: {e}"}}

    # 3. Warning para extensões "arriscadas" (opcional)
    WARNING_EXTS = {".json", ".env", ".csv"}
    warning_msg = f"Atenção: '{file_ext}' pode conter dados estruturados ou sensíveis. Leia com cuidado." if file_ext in WARNING_EXTS else None
    # --- Fim Checkers ---

    # Chama a função interna passando o warning
    return _read_file_content(resolved_path, filepath, warning_msg)

