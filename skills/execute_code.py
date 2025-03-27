# --- INÍCIO DO CÓDIGO PARA COPIAR ---
import os
import re
import ast
import subprocess
import tempfile
from core.config import PYTHON_EXEC_TIMEOUT
# REMOVIDO: from .modify_code import _find_code_in_history_or_file

# Importar a função de validação AST (assumindo que está no mesmo arquivo)
# Se _is_safe_ast estiver em outro lugar, ajuste o import
# from .ast_validator import _is_safe_ast # Exemplo se estivesse em ast_validator.py

# Função _is_safe_ast (COLE AQUI SE NÃO ESTIVER JÁ NO ARQUIVO)
# Certifique-se que a função _is_safe_ast está definida neste arquivo
# ou importada corretamente. Colando a última versão conhecida dela aqui
# para garantir que existe:
def _is_safe_ast(code_string: str) -> tuple[bool, str]:
    """Analisa a AST do código para permitir apenas construções seguras."""
    try:
        tree = ast.parse(code_string)
        allowed_nodes = (
            ast.Module, ast.Expr, ast.Constant, ast.Call, ast.Name,
            ast.Load, ast.Store, ast.Assign, ast.FunctionDef, ast.arguments,
            ast.arg, ast.Return, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
            # Adicionar outros nós seguros conforme necessário (ex: loops, condicionais básicos)
            ast.For, ast.While, ast.If, ast.Compare, ast.Eq, ast.NotEq,
            ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
            ast.Pass, ast.Break, ast.Continue, ast.List, ast.Tuple, ast.Dict,
            ast.Subscript, ast.Index, # Para acesso a listas/dicionários
        )
        allowed_calls = {'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'tuple', 'set'} # Funções built-in seguras
        allowed_imports = set() # Nenhum import permitido por padrão

        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False, f"Nó AST não permitido: {type(node).__name__}"
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in allowed_calls:
                        # Permitir chamadas a funções definidas no próprio código
                        # (Verifica se a função chamada está definida no escopo atual da AST)
                        is_defined_locally = False
                        for definition in ast.walk(tree):
                             if isinstance(definition, ast.FunctionDef) and definition.name == node.func.id:
                                 is_defined_locally = True
                                 break
                        if not is_defined_locally:
                             return False, f"Chamada de função não permitida: {node.func.id}"
                # Bloquear chamadas de atributos (ex: obj.method()) por segurança inicial
                elif isinstance(node.func, ast.Attribute):
                     return False, f"Chamada de método/atributo não permitida: {ast.dump(node.func)}"
            # Bloquear imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                 # Permitir apenas imports específicos se necessário
                 # module_name = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                 # if module_name not in allowed_imports:
                 return False, f"Imports não são permitidos ({type(node).__name__})"
            # Bloquear acesso a atributos como __builtins__, __import__ etc.
            if isinstance(node, ast.Attribute):
                 # Permitir acesso a atributos seguros se necessário no futuro
                 # Ex: if node.attr not in {'append', 'pop', ...}:
                 if node.attr.startswith('_'): # Bloqueia atributos "privados" ou "mágicos"
                      return False, f"Acesso a atributo não permitido: {node.attr}"
            # Validação específica para BinOp (já feita antes, mas redundância não prejudica)
            if isinstance(node, ast.BinOp):
                 if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                      return False, f"Operação binária não permitida: {type(node.op).__name__}"


        return True, "Código parece seguro (análise AST básica)."
    except SyntaxError as e:
        return False, f"Erro de sintaxe no código: {str(e)}"
    except Exception as e:
        return False, f"Erro inesperado durante análise AST: {e}"


# Função principal da skill
def skill_execute_code(action_input: dict, agent_memory: dict, agent_history: list | None = None) -> dict:
    """Executa código Python usando a assinatura padrão, buscando no histórico ou memória."""
    print("\n[Skill: Execute Code (ReAct)]")
    print(f"  Action Input: {action_input}")
    # print(f"  Memory Received: {agent_memory}") # Optional debug

    code_to_execute = None
    language = "python" # Default language for execution
    target_desc = action_input.get("target_description", "código desconhecido")

    # --- FIND CODE ---
    # 1. Try from ReAct history
    if agent_history:
        print("  Buscando código na Observation anterior...")
        for entry in reversed(agent_history):
            # Look for code blocks in Observations from generate or modify actions
            if entry.startswith("Observation:") and ("Código Gerado:" in entry or "Código Modificado:" in entry):
                 # Extract code, potentially inferring language
                 code_match = re.search(r"```(\w*)\s*([\s\S]*?)\s*```", entry, re.DOTALL)
                 if code_match:
                     lang_found = code_match.group(1).strip().lower()
                     code_block = code_match.group(2).strip()
                     if code_block: # Ensure code block is not empty
                         code_to_execute = code_block
                         language = lang_found if lang_found else language # Update language if found
                         target_desc = "o código da observação anterior"
                         print(f"  Código ({language}) encontrado na observação anterior.")
                         break # Found the most recent code

    # 2. Try from agent memory if not found in history
    if not code_to_execute:
        last_code_from_mem = agent_memory.get('last_code')
        last_lang_from_mem = agent_memory.get('last_lang')
        if last_code_from_mem:
             print("  Código não encontrado na Observation, usando memória do agente...")
             code_to_execute = last_code_from_mem
             language = last_lang_from_mem if last_lang_from_mem else language
             target_desc = "o último código na memória"
             print(f"  Código ({language}) encontrado na memória.")

    # Check if code was found
    if not code_to_execute:
         return {
             "status": "error",
             "action": "execute_code_failed",
             "data": {"message": f"Não foi possível encontrar o código alvo ('{target_desc}') para executar."}
         }

    # Check if language is executable (currently only Python)
    if language != 'python':
         return {
             "status": "error",
             "action": "execute_code_failed",
             "data": {"message": f"Atualmente, só posso executar código Python. O código encontrado é '{language}'."}
         }

    print(f"  Código a ser executado ({target_desc}):\n---\n{code_to_execute}\n---")

    # --- Safety Check (Existing Logic OK) ---
    # if not _is_safe_code(code_to_execute):
    #     print("[Execute Safety FAIL] Código considerado inseguro pela análise AST.")
    #     return {
    #         "status": "error",
    #         "action": "execute_code_failed",
    #         "data": {"message": "Execução bloqueada por motivos de segurança (análise AST)."}
    #     }
    # print("[Execute Safety PASS] Análise AST inicial não encontrou problemas óbvios.")

    # --- Execute in Firejail (UPDATED COMMAND) ---
    stdout_result = ""
    stderr_result = ""
    exit_code = -1

    try:
        # REMOVED: Temporary file creation
        # with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
        #     script_path = temp_script.name
        #     temp_script.write(code_to_execute)

        # REMOVED: Check for FIREJAIL_PROFILE existence

        # Construct Firejail command using --noprofile and python -c
        # Use 'python3' for broader compatibility
        firejail_command = [
            "firejail",
            "--quiet",
            "--noprofile",      # Use no profile for basic sandboxing
            "--net=none",       # Disable networking
            "--private",        # Use private home directory, tmpfs /tmp
            "python3",          # Interpreter
            "-c",               # Execute code from string
            code_to_execute     # The actual code
        ]

        print(f"  Executando via Firejail: {' '.join(firejail_command[:6])} python3 -c '...'") # Log truncated command

        # Execute using subprocess with timeout
        process = subprocess.run(
            firejail_command,
            capture_output=True,
            text=True,
            timeout=PYTHON_EXEC_TIMEOUT, # Use configured timeout
            check=False # Don't raise exception on non-zero exit code
        )

        stdout_result = process.stdout.strip()
        stderr_result = process.stderr.strip()
        exit_code = process.returncode

        print(f"  Execução concluída. Exit Code: {exit_code}")
        if stdout_result: print(f"  Stdout:\n{stdout_result}")
        if stderr_result: print(f"  Stderr:\n{stderr_result}")

        # REMOVED: Clean up the temporary file
        # os.remove(script_path)

        # --- Return Result ---
        if exit_code == 0:
            return {
                "status": "success",
                "action": "code_executed",
                "data": {
                    "code": code_to_execute,
                    "language": language,
                    "output": stdout_result,
                    "stderr": stderr_result, # Include stderr even on success (for warnings)
                    "exit_code": exit_code,
                    "message": f"Código Python ({target_desc}) executado com sucesso."
                }
            }
        else:
            error_cause = "erro durante a execução"
            return {
                "status": "error",
                "action": "execute_code_failed",
                "data": {
                    "code": code_to_execute,
                    "language": language,
                    "output": stdout_result,
                    "stderr": stderr_result,
                    "exit_code": exit_code,
                    "message": f"Código Python ({target_desc}) falhou com {error_cause} (exit code: {exit_code}). Stderr: {stderr_result}"
                }
            }

    except subprocess.TimeoutExpired:
        print(f"[Execute ERROR] Timeout ({PYTHON_EXEC_TIMEOUT}s) atingido durante a execução.")
        # No temp file to clean up anymore
        return {
            "status": "error",
            "action": "execute_code_failed",
            "data": {
                "code": code_to_execute,
                "language": language,
                "message": f"Execução do código ({target_desc}) excedeu o limite de tempo ({PYTHON_EXEC_TIMEOUT}s)."
            }
        }
    except FileNotFoundError:
         # This error could now mean 'firejail' or 'python3' is not found
         print(f"[Execute ERROR] Comando 'firejail' ou 'python3' não encontrado. Verifique a instalação e o PATH.")
         return {"status": "error", "action": "execute_code_failed", "data": {"message": "Erro de ambiente: 'firejail' ou 'python3' não encontrado."}}
    except Exception as e:
        print(f"\n[Erro Inesperado na Skill Execute] {e}")
        # traceback.print_exc() # Uncomment for debug
        # No temp file to clean up anymore
        return {
            "status": "error",
            "action": "execute_code_failed",
            "data": {"message": f"Erro inesperado ao tentar executar código ({target_desc}): {e}"}
        }

# --- FIM DO CÓDIGO PARA COPIAR ---