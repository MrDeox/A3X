# skills/list_files.py
import logging
from pathlib import Path
import os
from typing import List, Dict, Any
from core.tools import skill
# from core.validators import validate_path # REMOVED - validation logic is inline

# Corrected absolute import using alias
from core.config import PROJECT_ROOT as WORKSPACE_ROOT

# Initialize logger
logger = logging.getLogger(__name__)

# Define constants
MAX_ITEMS = 1000 # Limit for number of items listed

# --- Skill Function com Validação Interna ---
@skill(
    name="list_files",
    description="Lists non-hidden files and directories within a specified workspace directory (defaults to workspace root). Optionally filters by extension.",
    parameters={
        "directory": (str, "."), # Parâmetro opcional, default '.' (raiz do workspace)
        "extension": (str | None, None), # Parâmetro opcional para filtrar (e.g., '.py', '.txt')
        "agent_history": (list | None, None) # Added missing parameter
    }
)
def list_files(directory: str = ".", extension: str | None = None, agent_memory: dict | None = None, agent_history: list | None = None) -> dict:
    """
    Lists non-hidden files and directories within a specified workspace directory.
    Optionally filters by a specific file extension.

    Args:
        directory (str, optional): Path relative to the workspace root. Defaults to "." (root).
        extension (str | None, optional): If provided, filters results to files with this extension
                                          (e.g., '.py', '.txt'). Defaults to None.
        agent_memory (dict, optional): Agent's memory (not used). Defaults to None.
        agent_history (list | None, optional): Conversation history (not used). Defaults to None.

    Returns:
        dict: Standardized dictionary with the result:
              {"status": "success/error", "action": "directory_listed/list_files_failed",
               "data": {"message": "...", "directory_requested": "...",
                         "directory_resolved": "...", "items": [...], "item_count": ...}}
    """
    logger.debug(f"Skill 'list_files' requested for directory: '{directory}', filter: '{extension}'")

    # --- Validação de Path (Adaptada do antigo decorador) ---
    if not isinstance(directory, str) or not directory:
        logger.warning(f"Directory parameter received invalid type or empty string: {type(directory)}. Using default '.'")
        directory = "." # Usar default se inválido/vazio

    # Normalizar e resolver o path relativo ao WORKSPACE_ROOT
    try:
        workspace_resolved_path = Path(WORKSPACE_ROOT).resolve() # Resolve workspace path uma vez

        # Evitar paths absolutos ou que tentem sair do workspace
        # Check both string and Path object decomposition for '..'
        if os.path.isabs(directory) or ".." in directory or ".." in Path(directory).parts:
             raise ValueError("Path inválido: não use paths absolutos ou '..'. Use paths relativos dentro do workspace.")

        abs_path = workspace_resolved_path / directory
        resolved_path = abs_path.resolve()

        # Verificar se o path resolvido ainda está dentro do WORKSPACE_ROOT
        if not str(resolved_path).startswith(str(workspace_resolved_path)):
             # Check for symlinks pointing outside (if allowed, this check might need adjustment)
             # For now, strictly enforce containment.
             # is_symlink check might be needed depending on security policy for symlinks
            raise ValueError("Path inválido: tentativa de acesso fora do workspace.")


    except ValueError as e:
        logger.warning(f"Validação de path falhou para '{directory}': {e}")
        return {"status": "error", "action": "list_files_failed", "data": {"message": f"Path validation failed for '{directory}': {e}"}}
    except Exception as e:
        logger.error(f"Erro inesperado ao resolver o path '{directory}': {e}", exc_info=True)
        return {"status": "error", "action": "list_files_failed", "data": {"message": f"Erro interno ao processar o path '{directory}': {e}"}}

    # Verificar existência e tipo
    if not resolved_path.exists():
        return {"status": "error", "action": "list_files_failed", "data": {"message": f"Directory not found: '{directory}'"}}

    if not resolved_path.is_dir():
         return {"status": "error", "action": "list_files_failed", "data": {"message": f"The specified path is not a directory: '{directory}'"}}

    # --- Fim da Validação de Path ---

    logger.debug(f"Path validated: '{directory}' -> '{resolved_path}'")

    # --- Validação e Preparação do Filtro de Extensão ---
    filter_active = False
    normalized_extension = None
    if extension:
        if not isinstance(extension, str):
            logger.warning(f"Invalid extension type: {type(extension)}. Filter ignored.")
        elif not extension.startswith('.'):
            if '*.' in extension:
                normalized_extension = extension.replace('*.', '.').lower()
            elif extension: # Se não vazio e sem ponto/asterisco
                normalized_extension = '.' + extension.lower()
        elif len(extension) > 1: # Ignora se for apenas '.' mas permite '.py' etc
             normalized_extension = extension.lower()

        if normalized_extension:
            filter_active = True
            logger.debug(f"Extension filter active: '{normalized_extension}'")
        else:
             if extension: # Log only if extension was provided but deemed invalid
                logger.debug(f"Invalid or empty extension filter ('{extension}'), ignoring filter.")
             extension = None # Garante que extension seja None se inválido

    # --- Fim da Validação do Filtro ---

    try:
        items = []
        count = 0

        for item in resolved_path.iterdir():
            if count >= MAX_ITEMS:
                 logger.warning(f"Item limit ({MAX_ITEMS}) reached while listing '{resolved_path}'. List truncated.")
                 break

            # Filtra itens ocultos (começando com '.')
            if item.name.startswith('.'):
                continue

            # Aplica filtro de extensão se ativo E item for arquivo
            if filter_active and item.is_file() and item.suffix.lower() != normalized_extension:
                continue

            # Tenta tornar o path relativo ao WORKSPACE_ROOT para saída
            try:
                # Use the already resolved workspace path
                relative_path = str(item.relative_to(workspace_resolved_path))
            except ValueError:
                 relative_path = item.name # Fallback
                 logger.warning(f"Could not make path relative to workspace: {item}. Using name only.")

            # Adiciona '/' para diretórios
            if item.is_dir():
                items.append(relative_path + "/")
            else:
                items.append(relative_path)
            count += 1

        filter_message = f" matching extension '{normalized_extension}'" if filter_active else ""
        num_items = len(items)
        # Use 'directory' (o input original) na mensagem
        message = f"{num_items} non-hidden item(s){filter_message} found in '{directory}'."
        if count >= MAX_ITEMS:
            message += f" (Result truncated at {MAX_ITEMS} items)"

        # Calcula o path relativo resolvido para a saída
        try:
            resolved_relative_path = str(resolved_path.relative_to(workspace_resolved_path))
        except ValueError:
            resolved_relative_path = str(resolved_path) # Fallback se algo der errado
            logger.warning(f"Could not make resolved path relative: {resolved_path}")


        return {
            "status": "success",
            "action": "directory_listed",
            "data": {
                "directory_requested": directory, # O path como solicitado
                "directory_resolved": resolved_relative_path, # Path relativo resolvido
                "items": sorted(items),
                "item_count": num_items,
                "message": message
            }
        }

    except PermissionError:
        logger.error(f"Permission error listing directory: {resolved_path}", exc_info=True)
        return {"status": "error", "action": "list_files_failed", "data": {"message": f"Permission denied to list directory: '{directory}'"}}
    except Exception as e:
        logger.exception(f"Unexpected error listing directory '{resolved_path}':")
        return {"status": "error", "action": "list_files_failed", "data": {"message": f"Unexpected error listing directory '{directory}': {e}"}}

# Remover função antiga se existir (a original era skill_list_files)
# try:
#     del skill_list_files
# except NameError:
#     pass # Ignora se já foi removida ou renomeada
