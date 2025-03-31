# skills/read_file.py
import logging
import traceback
import os # Para juntar paths
from typing import Dict, Any
from pathlib import Path
# from core.validators import validate_path # REMOVED - assuming validation logic is inline
from core.tools import skill

# Corrected absolute import using alias
from core.config import PROJECT_ROOT as WORKSPACE_ROOT

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
        return {"status": "error", "action": "read_file_failed", "data": {"message": f"Arquivo não encontrado: '{filepath_original}' (inesperado após validação)."}}
    except PermissionError:
        logger.error(f"Erro de permissão ao ler arquivo: {filepath_original}", exc_info=True)
        return {"status": "error", "action": "read_file_failed", "data": {"message": f"Permissão negada para ler o arquivo: '{filepath_original}'"}}
    except IsADirectoryError:
         return {"status": "error", "action": "read_file_failed", "data": {"message": f"O caminho fornecido é um diretório, não um arquivo: '{filepath_original}'"}}
    except Exception as e:
        logger.exception(f"Erro inesperado ao ler arquivo '{filepath_original}':")
        return {"status": "error", "action": "read_file_failed", "data": {"message": f"Erro inesperado ao ler arquivo '{filepath_original}': {e}"}}

# --- Skill Function com Validação Interna ---
@skill(
    name="read_file",
    description="Reads the entire content of a specified text file within the workspace (up to 1MB). Supported extensions: .txt, .py, .md, .json, .env, .csv, .log.",
    parameters={"file_path": (str, ...)} # Parâmetro obrigatório
)
def read_file(file_path: str, agent_memory: dict | None = None, agent_history: list | None = None) -> dict:
    """
    Lê e retorna TODO o conteúdo de um arquivo de texto especificado (limite 1MB).
    Formatos suportados: .txt, .py, .md, .json, .env, .csv, .log
    Realiza validação de path e tipo antes da leitura.

    Args:
        file_path (str): Caminho relativo do arquivo dentro do workspace.
        agent_memory (dict, optional): Agent's memory (not used). Defaults to None.
        agent_history (list | None, optional): Histórico da conversa (não usado). Defaults to None.

    Returns:
        dict: Dicionário padronizado com o resultado da operação.
    """
    logger.debug(f"Skill 'read_file' solicitada para: '{file_path}'")

    # --- Validação de Path (Reimplementada do antigo decorador) ---
    if not isinstance(file_path, str) or not file_path:
        return {"status": "error", "action": "path_validation_failed", "data": {"message": "Nome do arquivo inválido ou não fornecido."}}

    # Normalizar e resolver o path relativo ao WORKSPACE_ROOT
    try:
        # Evitar paths absolutos ou que tentem sair do workspace
        if os.path.isabs(file_path) or ".." in file_path:
             raise ValueError("Path inválido: não use paths absolutos ou '..'. Use paths relativos dentro do workspace.")

        abs_path = Path(WORKSPACE_ROOT) / file_path
        resolved_path = abs_path.resolve()

        # Verificar se o path resolvido ainda está dentro do WORKSPACE_ROOT
        if not str(resolved_path).startswith(str(Path(WORKSPACE_ROOT).resolve())):
            raise ValueError("Path inválido: tentativa de acesso fora do workspace.")

    except ValueError as e:
        logger.warning(f"Validação falhou para '{file_path}': {e}")
        return {"status": "error", "action": "path_validation_failed", "data": {"message": str(e)}}
    except Exception as e: # Pega outros erros de path inesperados
        logger.error(f"Erro inesperado ao resolver o path '{file_path}': {e}", exc_info=True)
        return {"status": "error", "action": "path_validation_failed", "data": {"message": f"Erro interno ao processar o path: {e}"}}

    # Verificar existência e tipo
    if not resolved_path.exists():
        return {"status": "error", "action": "path_validation_failed", "data": {"message": f"Arquivo não encontrado: '{file_path}'"}}

    if not resolved_path.is_file():
         return {"status": "error", "action": "path_validation_failed", "data": {"message": f"O caminho fornecido não é um arquivo: '{file_path}'"}}

    # Verificar arquivos ocultos (começam com .)
    # if resolved_path.name.startswith('.'):
    #     return {"status": "error", "action": "path_validation_failed", "data": {"message": f"Acesso a arquivos ocultos não permitido: '{file_path}'"}}
    # --- Fim da Validação de Path ---

    logger.debug(f"Path validado: '{file_path}' -> '{resolved_path}'")

    # --- Checkers Adicionais (aplicados ao resolved_path validado) ---
    TEXT_EXTENSIONS = {".txt", ".py", ".md", ".json", ".env", ".csv", ".log"}
    file_ext = resolved_path.suffix.lower()

    if file_ext not in TEXT_EXTENSIONS:
        logger.warning(f"Tentativa de leitura de extensão não suportada: {file_ext} em '{file_path}'")
        return {
            "status": "error",
            "action": "read_file_failed_unsupported_ext",
            "data": {"message": f"Extensão '{file_ext}' não suportada. Use arquivos de texto: {', '.join(TEXT_EXTENSIONS)}"}
        }

    MAX_SIZE = 1 * 1024 * 1024  # 1MB
    try:
        file_size = resolved_path.stat().st_size
        if file_size > MAX_SIZE:
            logger.warning(f"Tentativa de leitura de arquivo muito grande: {file_size} bytes em '{file_path}'")
            return {
                "status": "error",
                "action": "read_file_failed_too_large",
                "data": {"message": f"Arquivo muito grande ({file_size / (1024*1024):.2f} MB). Limite: {MAX_SIZE / (1024*1024):.0f} MB."}
            }
    except FileNotFoundError: # Segurança extra
         return {"status": "error", "action": "read_file_failed", "data": {"message": f"Erro ao verificar tamanho: Arquivo não encontrado em '{file_path}' (inesperado)."}}
    except OSError as e:
        logger.error(f"Erro OSError ao verificar tamanho de '{file_path}': {e}", exc_info=True)
        return {"status": "error", "action": "read_file_failed_stat_error", "data": {"message": f"Erro ao verificar tamanho do arquivo: {e}"}}

    WARNING_EXTS = {".json", ".env", ".csv"}
    warning_msg = f"Atenção: '{file_ext}' pode conter dados estruturados ou sensíveis. Leia com cuidado." if file_ext in WARNING_EXTS else None
    # --- Fim Checkers ---

    # Chama a função interna, passando o path original para logs/mensagens
    return _read_file_content(resolved_path, file_path, warning_msg)

# Remover função antiga se existir (a original era skill_read_file)
# del skill_read_file

