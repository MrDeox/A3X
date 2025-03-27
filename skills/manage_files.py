import os
import glob

def skill_manage_files(entities: dict, original_command: str, intent: str = None, history: list = None) -> dict:
    """Gerencia arquivos (criar, listar, deletar)."""
    print("\n[Skill: Manage Files]")
    print(f"  Entidades recebidas: {entities}")
    if history:
        print(f"  Histórico recebido (últimos turnos): {history[-4:]}") # Mostra parte do histórico
    else:
        print("  Nenhum histórico fornecido")

    action = entities.get("action")
    file_name = entities.get("file_name")
    content = entities.get("content")
    file_extension = entities.get("file_extension")

    if not action:
        return {"status": "error", "action": "manage_files_failed", "data": {"message": "Não entendi qual ação realizar."}}

    try:
        if action == "create":
            if not file_name:
                return {"status": "error", "action": "manage_files_failed", "data": {"message": "Nome do arquivo não especificado."}}
            
            if not content:
                return {"status": "error", "action": "manage_files_failed", "data": {"message": "Conteúdo não especificado."}}
            
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "success", "action": "file_created", "data": {"message": f"Arquivo '{file_name}' criado com conteúdo."}}

        elif action == "append":
            if not file_name:
                return {"status": "error", "action": "manage_files_failed", "data": {"message": "Nome do arquivo não especificado."}}
            
            if not content:
                return {"status": "error", "action": "manage_files_failed", "data": {"message": "Conteúdo para adicionar não especificado."}}
            
            if not os.path.exists(file_name):
                return {"status": "error", "action": "manage_files_failed", "data": {"message": f"Arquivo '{file_name}' não encontrado para adicionar conteúdo."}}
            
            with open(file_name, "a", encoding="utf-8") as f:
                f.write(f"\n{content}")
            return {"status": "success", "action": "file_appended", "data": {"message": f"Conteúdo adicionado ao arquivo '{file_name}'."}}

        elif action == "list":
            if not file_extension:
                return {"status": "error", "action": "manage_files_failed", "data": {"message": "Extensão de arquivo não especificada."}}
            
            files = glob.glob(f"*{file_extension}")
            if not files:
                return {"status": "success", "action": "files_listed", "data": {"message": f"Nenhum arquivo{file_extension} encontrado."}}
            
            return {"status": "success", "action": "files_listed", "data": {"message": f"{len(files)} arquivo(s) encontrado(s), como: {', '.join(files[:3])}{'...' if len(files) > 3 else ''}"}}

        elif action == "delete":
            if not file_name:
                return {"status": "error", "action": "manage_files_failed", "data": {"message": "Nome do arquivo não especificado."}}
            
            if not os.path.exists(file_name):
                return {"status": "error", "action": "manage_files_failed", "data": {"message": f"Arquivo '{file_name}' não encontrado."}}
            
            # Requer confirmação para deletar
            return {
                "status": "confirmation_required",
                "action": "delete_file",
                "data": {
                    "file_name": file_name,
                    "confirmation_prompt": f"Tem certeza que deseja deletar o arquivo '{file_name}'?"
                }
            }

        else:
            return {"status": "error", "action": "manage_files_failed", "data": {"message": f"Ação '{action}' não suportada."}}

    except Exception as e:
        print(f"\n[Erro na Skill Manage Files] Ocorreu um erro: {e}")
        return {"status": "error", "action": "manage_files_failed", "data": {"message": f"Erro ao gerenciar arquivos: {e}"}}

def execute_delete_file(file_name: str) -> dict:
    """Executa a deleção de um arquivo após confirmação."""
    try:
        # Medida de segurança simples
        if os.path.isabs(file_name) or ".." in file_name:
            return {
                "status": "error",
                "action": "delete_file_failed",
                "data": {"message": "Desculpe, por segurança, só posso deletar arquivos diretamente no diretório atual."}
            }

        if not os.path.exists(file_name):
            return {
                "status": "error",
                "action": "delete_file_failed",
                "data": {"message": f"Erro: O arquivo '{file_name}' não existe."}
            }
        if not os.path.isfile(file_name):
            return {
                "status": "error",
                "action": "delete_file_failed",
                "data": {"message": f"Erro: '{file_name}' não é um arquivo."}
            }

        # Executa a deleção
        os.remove(file_name)
        return {
            "status": "success",
            "action": "file_deleted",
            "data": {
                "file_name": file_name,
                "message": f"Arquivo '{file_name}' deletado com sucesso."
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "action": "delete_file_failed",
            "data": {"message": f"Erro ao deletar o arquivo '{file_name}': {e}"}
        } 