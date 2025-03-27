# --- INÍCIO DO CÓDIGO PARA COPIAR ---
import os
import re
import ast
import subprocess
from .modify_code import _find_code_in_history_or_file # Garanta que esta importação está no topo do arquivo

def skill_execute_code(entities: dict, original_command: str, intent: str = None, history: list = None) -> dict:
    """Executa código Python de forma restrita e segura usando Firejail como sandbox."""
    print("\n[Skill: Execute Code (Restricted)]")
    print(f"  Entidades recebidas: {entities}")
    # Removido log de histórico para evitar logs muito grandes
    # print(f"  Histórico recebido (últimos turnos): {history[-5:] if history else []}")

    # Encontrar código para executar
    file_name = entities.get("file_name")
    code_to_execute, target_desc, language = _find_code_in_history_or_file(file_name, history)

    if not code_to_execute or language != 'python':
        return {
            "status": "error",
            "action": "execute_code_failed",
            "data": {"message": "Não foi possível encontrar código Python para executar."}
        }

    # Análise estática (AST) para validação de segurança
    try:
        tree = ast.parse(code_to_execute)
    except SyntaxError as e:
        return {
            "status": "error",
            "action": "execute_code_failed",
            "data": {"message": f"Erro de sintaxe no código: {str(e)}"}
        }

    # Verificar tipos de nós permitidos
    for node in ast.walk(tree):
        allowed_node_found = False # Reseta para cada nó

        # Caso 1: Tipos Básicos Permitidos (SEM Assign, SEM BinOp aqui)
        if isinstance(node, (ast.Module, ast.Expr, ast.Constant, ast.Name, ast.Load, ast.Store)):
            allowed_node_found = True
            continue

        # Caso 2: Chamadas de Função Permitidas (print com args simples ou BinOp)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'print':
                all_args_allowed = True
                for arg in node.args:
                    # Permitir Constantes, Nomes, ou Operações Binárias como args do print
                    if not isinstance(arg, (ast.Constant, ast.Name, ast.BinOp)):
                         all_args_allowed = False
                         break
                    # Se for BinOp, verificar se a operação interna é permitida
                    if isinstance(arg, ast.BinOp):
                        if not isinstance(arg.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                            all_args_allowed = False
                            break
                if all_args_allowed:
                    allowed_node_found = True
                    continue

        # Caso 3: Atribuição (Assign)
        if isinstance(node, ast.Assign):
            all_targets_allowed = True
            for target in node.targets:
                if not isinstance(target, ast.Name): # Só permite atribuir a nomes simples
                    all_targets_allowed = False
                    break
            # Poderíamos adicionar verificação no node.value aqui se necessário
            if all_targets_allowed:
                 allowed_node_found = True
                 continue

        # Caso 4: Operações Binárias Permitidas (fora dos args do print)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                 # Poderíamos verificar node.left/right aqui se necessário
                 allowed_node_found = True
                 continue

        # Se o nó atual não foi coberto pelos casos acima, não é permitido
        if not allowed_node_found:
            print(f"[DEBUG] Nó AST não permitido encontrado: {type(node).__name__}") # Log para depuração
            return {
                "status": "error",
                "action": "execute_code_failed", # Corrigido action
                "data": {"message": f"Erro de segurança: Código contém construção não permitida: {type(node).__name__}."}
            }

    # Se o loop terminar sem retornar erro, o código é considerado seguro (para esta versão)
    print("[Skill: Execute Code] Análise AST passou.")

    # Execução segura via subprocess e firejail
    print("[Skill: Execute Code] Executando via Firejail...")
    try:
        cmd = [
            'firejail', '--quiet', '--noprofile', '--net=none', '--private',
            'python3', '-c', code_to_execute
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, check=False
        )
        captured_output = result.stdout
        captured_error = result.stderr

        if result.returncode == 0:
            return {
                "status": "success",
                "action": "code_executed",
                "data": {
                    "message": f"Código de {target_desc} executado via Firejail.",
                    "output": captured_output,
                    "stderr": captured_error
                }
            }
        else:
            return {
                "status": "error",
                "action": "execute_code_failed",
                "data": {
                    "message": f"Erro durante a execução (Firejail) do código de {target_desc}.",
                    "error_details": f"Exit Code: {result.returncode}",
                    "output": captured_output,
                    "stderr": captured_error
                }
            }
    except subprocess.TimeoutExpired:
        return { "status": "error", "action": "execute_code_failed", "data": {"message": "Erro: Tempo limite de execução excedido (5 segundos)."}}
    except FileNotFoundError:
         return { "status": "error", "action": "execute_code_failed", "data": {"message": "Erro: Comando 'firejail' não encontrado. Ele está instalado e no PATH?"}}
    except Exception as e:
        return { "status": "error", "action": "execute_code_failed", "data": { "message": "Erro inesperado ao tentar executar o código via Firejail.", "error_details": str(e)}}
# --- FIM DO CÓDIGO PARA COPIAR ---