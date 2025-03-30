import os
import logging
from pathlib import Path

# Initialize logger
logger = logging.getLogger(__name__)

# Define o diretório raiz do projeto (ajuste se necessário)
# Idealmente, isso seria configurado globalmente
WORKSPACE_ROOT = Path("/home/arthur/Projects/A3X").resolve()

def _is_path_within_workspace(path: str | Path) -> bool:
    """Verifica se o caminho fornecido está dentro do WORKSPACE_ROOT."""
    try:
        absolute_path = Path(path).resolve()
        return absolute_path.is_relative_to(WORKSPACE_ROOT)
    except ValueError: # Caso o path seja inválido
        return False
    except Exception as e: # Outros erros inesperados (ex: permissão)
        logger.error(f"Erro ao verificar caminho '{path}': {e}")
        return False

def _resolve_path(filepath: str) -> Path | None:
    """Resolve o filepath para um caminho absoluto seguro dentro do workspace."""
    path = Path(filepath)
    if path.is_absolute():
        # Se absoluto, já verifica se está no workspace
        if _is_path_within_workspace(path):
            return path.resolve()
        else:
            logger.warning(f"Acesso negado: Caminho absoluto fora do workspace: {path}")
            return None
    else:
        # Se relativo, junta com o WORKSPACE_ROOT
        resolved_path = (WORKSPACE_ROOT / path).resolve()
        # Verifica novamente se o caminho resolvido ainda está no workspace
        if _is_path_within_workspace(resolved_path):
            return resolved_path
        else:
            # Isso pode acontecer com caminhos relativos como "../../../etc/passwd"
            logger.warning(f"Acesso negado: Caminho relativo resolvido fora do workspace: {filepath} -> {resolved_path}")
            return None

def _create_file(filepath: str, content: str, overwrite: bool) -> dict:
    """Cria um novo arquivo."""
    resolved_path = _resolve_path(filepath)
    if not resolved_path:
        return {"status": "error", "action": "create_file_failed", "data": {"message": f"Acesso negado ou caminho inválido para criação: '{filepath}'"}}

    try:
        if resolved_path.exists():
            if resolved_path.is_dir():
                 return {"status": "error", "action": "create_file_failed", "data": {"message": f"Não é possível criar arquivo, já existe um diretório com este nome: '{filepath}'"}}
            if not overwrite:
                return {"status": "error", "action": "create_file_failed", "data": {"message": f"Arquivo '{filepath}' já existe. Use 'overwrite: True' para sobrescrever."}}
            # Se overwrite=True e é um arquivo, continua para sobrescrever

        # Garante que o diretório pai exista
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(content)
        action = "file_overwritten" if overwrite and resolved_path.exists() else "file_created"
        message = f"Arquivo '{filepath}' {'sobrescrito' if overwrite and resolved_path.exists() else 'criado'} com sucesso."

        return {"status": "success", "action": action, "data": {"message": message, "filepath": filepath}}

    except PermissionError:
        logger.error(f"Erro de permissão ao criar/sobrescrever arquivo: {filepath}", exc_info=True)
        return {"status": "error", "action": "create_file_failed", "data": {"message": f"Permissão negada para criar/sobrescrever o arquivo: '{filepath}'"}}
    except IsADirectoryError: # Caso tente criar onde já existe diretório (mkdir deve falhar antes?)
         return {"status": "error", "action": "create_file_failed", "data": {"message": f"Não é possível criar arquivo, já existe um diretório com este nome: '{filepath}'"}}
    except Exception as e:
        logger.exception(f"Erro inesperado ao criar/sobrescrever arquivo '{filepath}':")
        return {"status": "error", "action": "create_file_failed", "data": {"message": f"Erro inesperado ao criar/sobrescrever arquivo '{filepath}': {e}"}}

def _append_to_file(filepath: str, content: str) -> dict:
    """Adiciona conteúdo ao final de um arquivo."""
    resolved_path = _resolve_path(filepath)
    if not resolved_path:
        return {"status": "error", "action": "append_failed", "data": {"message": f"Acesso negado ou caminho inválido para append: '{filepath}'"}}

    try:
        if not resolved_path.exists():
             # Comportamento: criar o arquivo se não existir? Ou erro?
             # Vamos retornar erro por enquanto para ser explícito.
             # Poderia ser um parâmetro create_if_not_exists=True se necessário.
             return {"status": "error", "action": "append_failed", "data": {"message": f"Arquivo '{filepath}' não encontrado para adicionar conteúdo."}}
        if resolved_path.is_dir():
             return {"status": "error", "action": "append_failed", "data": {"message": f"Não é possível adicionar conteúdo a um diretório: '{filepath}'"}}

        # Garante que o diretório pai exista (embora se o arquivo existe, o pai deveria existir)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        with open(resolved_path, "a", encoding="utf-8") as f:
            # Adiciona uma nova linha se o conteúdo não terminar com uma?
            # Vamos adicionar sempre por consistência com o código anterior.
            f.write(content + "\n")

        return {"status": "success", "action": "file_appended", "data": {"message": f"Conteúdo adicionado a '{filepath}'.", "filepath": filepath}}

    except PermissionError:
        logger.error(f"Erro de permissão ao adicionar conteúdo ao arquivo: {filepath}", exc_info=True)
        return {"status": "error", "action": "append_failed", "data": {"message": f"Permissão negada para adicionar conteúdo ao arquivo: '{filepath}'"}}
    except IsADirectoryError:
         return {"status": "error", "action": "append_failed", "data": {"message": f"Não é possível adicionar conteúdo a um diretório: '{filepath}'"}}
    except Exception as e:
        logger.exception(f"Erro inesperado ao adicionar conteúdo ao arquivo '{filepath}':")
        return {"status": "error", "action": "append_failed", "data": {"message": f"Erro inesperado ao adicionar conteúdo ao arquivo '{filepath}': {e}"}}

def skill_manage_files(action_input: dict, agent_memory: dict = None, agent_history: list | None = None) -> dict:
    """
    Gerencia arquivos (criar, adicionar) dentro do workspace.
    NOTE: Leitura, listagem e deleção são feitas por skills separadas.

    Args:
        action_input (dict): Dicionário contendo a ação e parâmetros.
            Exemplos:
                {"action": "create", "filepath": "path/to/new.txt", "content": "Olá", "overwrite": False}
                {"action": "append", "filepath": "path/to/file.txt", "content": " Mundo"}
        agent_memory (dict, optional): Memória do agente (não usada). Defaults to None.
        agent_history (list | None, optional): Histórico da conversa (não usado). Defaults to None.

    Returns:
        dict: Dicionário padronizado com o resultado da operação:
              {"status": "success/error", "action": "...", "data": {"message": "...", ...}}
    """
    logger.debug(f"Recebido action_input para manage_files: {action_input}")

    action = action_input.get("action")

    if not action:
        return {"status": "error", "action": "manage_files_failed", "data": {"message": "Parâmetro 'action' ausente no Action Input."}}

    try:
        if action == "create":
            filepath = action_input.get("filepath")
            content = action_input.get("content")
            # overwrite default é False
            overwrite = action_input.get("overwrite", False)
            if not filepath or content is None: # content pode ser string vazia
                return {"status": "error", "action": "create_file_failed", "data": {"message": "Parâmetros 'filepath' e 'content' são obrigatórios para a ação 'create'."}}
            if not isinstance(overwrite, bool):
                 return {"status": "error", "action": "create_file_failed", "data": {"message": "Parâmetro 'overwrite' deve ser um booleano (true/false)."}}
            return _create_file(filepath, content, overwrite)

        elif action == "append":
            filepath = action_input.get("filepath")
            content = action_input.get("content")
            if not filepath or content is None: # content pode ser string vazia
                return {"status": "error", "action": "append_failed", "data": {"message": "Parâmetros 'filepath' e 'content' são obrigatórios para a ação 'append'."}}
            return _append_to_file(filepath, content)

        # --- Ação: Deletar Arquivo (Ainda Desabilitada) ---
        elif action == "delete":
             filepath = action_input.get("filepath")
             if not filepath:
                  return {"status": "error", "action": "delete_failed", "data": {"message": "Parâmetro 'filepath' obrigatório para 'delete'."}}

             logger.warning(f"Ação 'delete' chamada para '{filepath}', mas desabilitada.")
             # No futuro, a lógica de deleção segura iria aqui, talvez chamando _delete_file(filepath)
             # que faria as verificações de segurança (_resolve_path, etc.) e chamaria os.remove() ou shutil.rmtree()
             return {
                 "status": "error",
                 "action":"action_not_implemented",
                 "data": {
                     "message": f"A deleção de '{filepath}' ainda não está implementada/habilitada nesta skill."
                 }
             }

        else:
            return {"status": "error", "action": "manage_files_failed", "data": {"message": f"Ação '{action}' não é suportada pela skill 'manage_files'."}}

    except Exception as e:
        logger.exception(f"Erro inesperado na skill manage_files ao processar ação '{action}':")
        # traceback.print_exc() # Logger já captura a exceção
        return {"status": "error", "action": "manage_files_failed", "data": {"message": f"Erro inesperado ao executar a ação '{action}': {e}"}} 