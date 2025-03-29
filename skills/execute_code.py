# --- INÍCIO DO CÓDIGO PARA COPIAR ---
import os
import re
import ast
import subprocess
import traceback
import logging
from pathlib import Path
from core.config import PYTHON_EXEC_TIMEOUT # Default timeout

# Initialize logger
logger = logging.getLogger(__name__)

# Define o diretório raiz do projeto (ajuste se necessário)
WORKSPACE_ROOT = Path("/home/arthur/Projects/A3X").resolve()

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
            # Validação específica para BinOp
            if isinstance(node, ast.BinOp):
                 if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                      return False, f"Operação binária não permitida: {type(node.op).__name__}"


        return True, "Código parece seguro (análise AST básica)."
    except SyntaxError as e:
        return False, f"Erro de sintaxe no código: {str(e)}"
    except Exception as e:
        logger.error(f"Erro inesperado durante análise AST: {e}", exc_info=True)
        return False, f"Erro inesperado durante análise AST: {e}"


def skill_execute_code(action_input: dict) -> dict:
    """
    Executa um bloco de código (atualmente apenas Python) em um sandbox Firejail.

    Args:
        action_input (dict): Dicionário contendo a ação e parâmetros.
            Exemplo:
                {"action": "execute", "code": "print('Hello')", "language": "python", "timeout": 10}

    Returns:
        dict: Dicionário padronizado com o resultado da execução:
              {"status": "success/error", "action": "code_executed/execution_failed",
               "data": {"message": "...", "stdout": "...", "stderr": "...", "returncode": ...}}
    """
    logger.debug(f"Recebido action_input para execute_code: {action_input}")

    code_to_execute = action_input.get("code")
    language = action_input.get("language", "python").lower() # Default to python, ensure lowercase
    timeout = action_input.get("timeout", PYTHON_EXEC_TIMEOUT) # Use provided or default timeout

    if not code_to_execute:
        return {
            "status": "error",
            "action": "execution_failed",
            "data": {"message": "Parâmetro 'code' ausente no Action Input."}
        }

    # Validar timeout (deve ser um número positivo)
    try:
        timeout_sec = float(timeout)
        if timeout_sec <= 0:
             raise ValueError("Timeout must be positive")
    except (ValueError, TypeError):
         logger.warning(f"Timeout inválido fornecido: {timeout}. Usando default: {PYTHON_EXEC_TIMEOUT}s")
         timeout_sec = PYTHON_EXEC_TIMEOUT

    # Validar linguagem suportada
    if language != 'python':
        return {
            "status": "error",
            "action": "execution_failed",
            "data": {"message": f"Linguagem não suportada: '{language}'. Atualmente, apenas 'python' é suportado."}
        }

    logger.info(f"Tentando executar código {language} com timeout de {timeout_sec}s.")
    # Logar apenas uma parte do código para não poluir logs
    code_preview = code_to_execute[:100] + ("..." if len(code_to_execute) > 100 else "")
    logger.debug(f"Código (prévia):\n---\n{code_preview}\n---")

    # --- Safety Check (AST) ---
    is_safe, safety_message = _is_safe_ast(code_to_execute)
    if not is_safe:
        logger.warning(f"Execução bloqueada pela análise AST: {safety_message}")
        return {
            "status": "error",
            "action": "execution_failed",
            "data": {"message": f"Execução bloqueada por motivos de segurança (análise AST): {safety_message}"}
        }
    logger.debug(f"Resultado da análise AST: {safety_message}")

    # --- Execute in Firejail ---
    stdout_result = ""
    stderr_result = ""
    exit_code = -1

    try:
        # Construir comando Firejail
        # Usar 'python3' para melhor compatibilidade
        firejail_command = [
            "firejail",
            "--quiet",
            "--noprofile",      # Sandbox básico sem perfil específico
            "--net=none",       # Desabilitar rede
            "--private",        # Diretório home privado, tmpfs /tmp
            "python3",          # Interpretador
            "-c",               # Executar código da string
            code_to_execute     # O código em si
        ]

        logger.debug(f"Executando comando: {' '.join(firejail_command[:6])} python3 -c '...'" ) # Log truncado

        # Executar com subprocess.run
        process = subprocess.run(
            firejail_command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False # Não lança exceção em erro, verificamos o exit_code
        )

        stdout_result = process.stdout.strip()
        stderr_result = process.stderr.strip()
        exit_code = process.returncode

        logger.info(f"Execução concluída. Exit Code: {exit_code}")
        if stdout_result: logger.debug(f"Stdout: {stdout_result}")
        if stderr_result: logger.debug(f"Stderr: {stderr_result}")

        # --- Return Result ---
        if exit_code == 0:
            return {
                "status": "success",
                "action": "code_executed",
                "data": {
                    "stdout": stdout_result,
                    "stderr": stderr_result, # Incluir stderr mesmo em sucesso (para warnings)
                    "returncode": exit_code,
                    "message": "Código executado com sucesso."
                }
            }
        else:
            # Tentar dar uma mensagem de erro mais específica
            error_cause = f"erro durante a execução (exit code: {exit_code})"
            if "SyntaxError:" in stderr_result:
                error_cause = "erro de sintaxe"
            elif "Traceback (most recent call last):" in stderr_result:
                 error_cause = "erro de runtime"

            return {
                "status": "error",
                "action": "execution_failed",
                "data": {
                    "stdout": stdout_result,
                    "stderr": stderr_result,
                    "returncode": exit_code,
                    "message": f"Falha na execução do código: {error_cause}. Stderr: {stderr_result[:200]}..." # Limita stderr na msg
                }
            }

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout ({timeout_sec}s) atingido durante a execução.")
        return {
            "status": "error",
            "action": "execution_failed",
            "data": {"message": f"Execução do código excedeu o limite de tempo ({timeout_sec}s)."}
        }
    except FileNotFoundError:
         # Pode ser 'firejail' ou 'python3'
         logger.error("Comando 'firejail' ou 'python3' não encontrado. Verifique a instalação e o PATH.", exc_info=True)
         return {"status": "error", "action": "execution_failed", "data": {"message": "Erro de ambiente: 'firejail' ou 'python3' não encontrado."}}
    except Exception as e:
        logger.exception("Erro inesperado ao tentar executar código:")
        return {
            "status": "error",
            "action": "execution_failed",
            "data": {"message": f"Erro inesperado ao tentar executar código: {e}"}
        }

# --- FIM DO CÓDIGO PARA COPIAR ---