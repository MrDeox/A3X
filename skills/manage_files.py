import os
import glob
import traceback
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

def _read_file(filepath: str) -> dict:
    """Lê o conteúdo de um arquivo."""
    resolved_path = _resolve_path(filepath)
    if not resolved_path:
        return {"status": "error", "action": "read_file_failed", "data": {"message": f"Acesso negado ou caminho inválido: '{filepath}'"}}

    try:
        if not resolved_path.is_file():
            return {"status": "error", "action": "read_file_failed", "data": {"message": f"Arquivo não encontrado ou não é um arquivo: '{filepath}'"}}

        with open(resolved_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Limita o tamanho do conteúdo retornado na mensagem para não poluir logs/prompt
        max_len_preview = 500
        content_preview = content[:max_len_preview] + ("..." if len(content) > max_len_preview else "")

        return {
            "status": "success",
            "action": "file_read",
            "data": {
                "filepath": filepath, # Retorna o path original solicitado
                "content": content, # Retorna o conteúdo completo
                "message": f"Conteúdo do arquivo '{filepath}' lido com sucesso (Prévia: {content_preview})"
            }
        }
    except FileNotFoundError: # Deveria ser pego por is_file(), mas por segurança
        return {"status": "error", "action": "read_file_failed", "data": {"message": f"Arquivo não encontrado: '{filepath}'"}}
    except PermissionError:
        logger.error(f"Erro de permissão ao ler arquivo: {filepath}", exc_info=True)
        return {"status": "error", "action": "read_file_failed", "data": {"message": f"Permissão negada para ler o arquivo: '{filepath}'"}}
    except IsADirectoryError:
         return {"status": "error", "action": "read_file_failed", "data": {"message": f"O caminho fornecido é um diretório, não um arquivo: '{filepath}'"}}
    except Exception as e:
        logger.exception(f"Erro inesperado ao ler arquivo '{filepath}':")
        return {"status": "error", "action": "read_file_failed", "data": {"message": f"Erro inesperado ao ler arquivo '{filepath}': {e}"}}

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

def _list_directory(directory: str) -> dict:
    """Lista o conteúdo de um diretório."""
    resolved_path = _resolve_path(directory)
    if not resolved_path:
        return {"status": "error", "action": "list_failed", "data": {"message": f"Acesso negado ou caminho inválido para listar: '{directory}'"}}

    try:
        if not resolved_path.exists():
             return {"status": "error", "action": "list_failed", "data": {"message": f"Diretório não encontrado: '{directory}'"}}
        if not resolved_path.is_dir():
             return {"status": "error", "action": "list_failed", "data": {"message": f"O caminho fornecido não é um diretório: '{directory}'"}}

        items = list(resolved_path.iterdir())
        # Converte os Paths para strings relativas ao workspace para a saída
        relative_items = []
        for item in items:
            try:
                relative_path = str(item.relative_to(WORKSPACE_ROOT))
                if item.is_dir():
                    relative_items.append(relative_path + "/")
                else:
                    relative_items.append(relative_path)
            except ValueError:
                # Se, por alguma razão, um item listado não estiver no workspace
                # (links simbólicos?), apenas use o nome.
                 relative_items.append(item.name + ("/" if item.is_dir() else ""))


        num_items = len(relative_items)
        message = f"{num_items} item(s) encontrado(s) em '{directory}'."

        # Limita a lista mostrada na mensagem para não poluir
        max_show = 15
        if num_items > 0:
            sample_items_display = sorted(relative_items)[:max_show]
            message += f" Exemplo: {', '.join(sample_items_display)}"
            if num_items > max_show:
                message += f"... (e mais {num_items - max_show})"


        return {
            "status": "success",
            "action": "directory_listed",
            "data": {
                "directory": directory,
                "items": sorted(relative_items), # Retorna a lista completa e ordenada
                "message": message
            }
        }

    except PermissionError:
        logger.error(f"Erro de permissão ao listar diretório: {directory}", exc_info=True)
        return {"status": "error", "action": "list_failed", "data": {"message": f"Permissão negada para listar o diretório: '{directory}'"}}
    except NotADirectoryError: # Deveria ser pego por is_dir(), mas por segurança
        return {"status": "error", "action": "list_failed", "data": {"message": f"O caminho fornecido não é um diretório: '{directory}'"}}
    except Exception as e:
        logger.exception(f"Erro inesperado ao listar diretório '{directory}':")
        return {"status": "error", "action": "list_failed", "data": {"message": f"Erro inesperado ao listar diretório '{directory}': {e}"}}


# Remover a função execute_delete_file se não for mais usada diretamente
# def execute_delete_file(file_name: str) -> dict: ...

def skill_manage_files(action_input: dict, agent_memory: dict = None, agent_history: list | None = None) -> dict:
    """
    Gerencia arquivos e diretórios dentro do workspace (ler, criar, adicionar, listar).

    Args:
        action_input (dict): Dicionário contendo a ação e parâmetros.
            Exemplos:
                {"action": "read", "filepath": "path/to/file.txt"}
                {"action": "create", "filepath": "path/to/new.txt", "content": "Olá", "overwrite": False}
                {"action": "append", "filepath": "path/to/file.txt", "content": " Mundo"}
                {"action": "list", "directory": "path/to/dir"} (relativo ao workspace)
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
        if action == "read":
            filepath = action_input.get("filepath")
            if not filepath:
                return {"status": "error", "action": "read_file_failed", "data": {"message": "Parâmetro 'filepath' obrigatório para a ação 'read'."}}
            return _read_file(filepath)

        elif action == "create":
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

        elif action == "list":
            directory = action_input.get("directory")
            if directory is None: # Permite listar o root se directory='' ou não fornecido? Vamos exigir.
                # Se quiser listar o root, passe directory="." ou directory=""
                 return {"status": "error", "action": "list_failed", "data": {"message": "Parâmetro 'directory' obrigatório para a ação 'list'."}}
            return _list_directory(directory)

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