import os
import glob
import traceback # Para debug, se necessário

# Remover a função execute_delete_file se não for mais usada diretamente
# def execute_delete_file(file_name: str) -> dict: ...

def skill_manage_files(action_input: dict, agent_memory: dict, agent_history: list | None = None) -> dict:
    """
    Gerencia arquivos (criar, adicionar, listar).
    Ação 'delete' está temporariamente desabilitada no fluxo ReAct.

    Args:
        action_input (dict): Dicionário contendo a ação e parâmetros.
            Ex: {"action": "list", "file_extension": "*.py"}
                {"action": "create", "file_name": "meu_arquivo.txt", "content": "Olá"}
                {"action": "append", "file_name": "meu_arquivo.txt", "content": "Mundo"}
        agent_memory (dict): Memória do agente (não usada nesta skill).
        agent_history (list | None): Histórico da conversa (não usado nesta skill).

    Returns:
        dict: Resultado da operação.
    """
    print("\n[Skill: Manage Files (ReAct)]")
    print(f"  Action Input: {action_input}")

    action = action_input.get("action")
    file_name = action_input.get("file_name")
    content = action_input.get("content")
    file_extension = action_input.get("file_extension") # Usado para list

    if not action:
        return {"status": "error", "action": "manage_files_failed", "data": {"message": "Parâmetro 'action' ausente no Action Input."}}

    try:
        # --- Ação: Criar Arquivo ---
        if action == "create":
            if not file_name or content is None: # Content pode ser string vazia
                 return {"status": "error", "action": "manage_files_failed", "data": {"message": "Parâmetros 'file_name' e 'content' são obrigatórios para 'create'."}}
            # Medida de segurança simples: impedir escrita fora do diretório atual
            if os.path.dirname(file_name):
                 return {"status": "error", "action": "manage_files_failed", "data": {"message": "A criação de arquivos só é permitida no diretório atual."}}
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "success", "action": "file_created", "data": {"message": f"Arquivo '{file_name}' criado com sucesso."}}

        # --- Ação: Adicionar Conteúdo ---
        elif action == "append":
            if not file_name or content is None:
                 return {"status": "error", "action": "manage_files_failed", "data": {"message": "Parâmetros 'file_name' e 'content' são obrigatórios para 'append'."}}
            if os.path.dirname(file_name): # Segurança
                 return {"status": "error", "action": "manage_files_failed", "data": {"message": "A modificação de arquivos só é permitida no diretório atual."}}
            if not os.path.exists(file_name):
                return {"status": "error", "action": "manage_files_failed", "data": {"message": f"Arquivo '{file_name}' não encontrado para adicionar conteúdo."}}
            with open(file_name, "a", encoding="utf-8") as f:
                # Adiciona newline se o arquivo não estiver vazio E não terminar com newline
                if os.path.getsize(file_name) > 0:
                     f.seek(0, os.SEEK_END)
                     f.seek(f.tell() - 1, os.SEEK_SET)
                     if f.read(1) != '\n':
                          f.write('\n')
                f.write(content)
            return {"status": "success", "action": "file_appended", "data": {"message": f"Conteúdo adicionado ao arquivo '{file_name}'."}}

        # --- Ação: Listar Arquivos ---
        elif action == "list":
            # Usa file_extension como padrão glob
            pattern = file_extension
            if not pattern:
                 # Se não fornecer padrão, lista tudo no diretório atual? Ou erro?
                 # Vamos listar tudo por padrão, mas com aviso.
                 print("[Manage Files WARN] Nenhum padrão fornecido para 'list', listando todos os arquivos/dirs no diretório atual.")
                 pattern = "*" # Lista tudo no diretório atual

            # Medida de segurança: só permite padrões no diretório atual
            if os.path.dirname(pattern):
                 return {"status": "error", "action": "manage_files_failed", "data": {"message": "A listagem de arquivos só é permitida no diretório atual."}}

            try:
                 files = glob.glob(pattern)
                 num_files = len(files)

                 if not files:
                     message = f"Nenhum arquivo ou diretório correspondente a '{pattern}' encontrado no diretório atual."
                 else:
                     # Limita a lista mostrada na mensagem para não poluir
                     max_show = 10
                     sample_files = files[:max_show]
                     # Adiciona '/' a diretórios para clareza
                     sample_files_display = [f + '/' if os.path.isdir(f) else f for f in sample_files]
                     
                     message = f"{num_files} arquivo(s)/diretório(s) encontrado(s) para '{pattern}': {', '.join(sample_files_display)}"
                     if num_files > max_show:
                          message += f"... (e mais {num_files - max_show})"

                 # Retorna a lista completa E a mensagem formatada
                 return {"status": "success", "action": "files_listed", "data": {"files": files, "message": message}}

            except Exception as glob_e:
                 print(f"[Erro Manage Files] Erro ao usar glob com '{pattern}': {glob_e}")
                 return {"status": "error", "action": "manage_files_failed", "data": {"message": f"Erro ao listar arquivos com padrão '{pattern}': {glob_e}"}}

        # --- Ação: Deletar Arquivo (Temporariamente Desabilitada no ReAct) ---
        elif action == "delete":
             if not file_name:
                  return {"status": "error", "action": "manage_files_failed", "data": {"message": "Parâmetro 'file_name' obrigatório para 'delete'."}}
             if os.path.dirname(file_name): # Segurança
                 return {"status": "error", "action": "manage_files_failed", "data": {"message": "A deleção de arquivos só é permitida no diretório atual."}}

             print(f"[Manage Files WARN] Ação 'delete' chamada, mas temporariamente desabilitada no fluxo ReAct (requer tratamento de confirmação pelo Agente).")
             # No futuro, o Agente precisaria:
             # 1. Chamar esta skill.
             # 2. Receber um status como "confirmation_required".
             # 3. Perguntar ao usuário "Tem certeza que deseja deletar X?".
             # 4. Se sim, chamar uma sub-função ou outra skill para efetivar a deleção.
             return {"status": "error", "action":"action_not_fully_implemented", "data": {"message": f"A deleção do arquivo '{file_name}' requer confirmação e ainda não está totalmente suportada neste modo."}}

        # --- Ação Desconhecida ---
        else:
            return {"status": "error", "action": "manage_files_failed", "data": {"message": f"Ação '{action}' não é suportada pela skill 'manage_files'."}}

    except Exception as e:
        print(f"\n[Erro na Skill Manage Files] Ocorreu um erro inesperado: {e}")
        traceback.print_exc() # Imprime traceback para debug detalhado
        return {"status": "error", "action": "manage_files_failed", "data": {"message": f"Erro inesperado ao executar a ação '{action}': {e}"}} 