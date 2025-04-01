# --- INÍCIO DO CÓDIGO PARA COPIAR ---
import ast
import subprocess
import logging
from pathlib import Path
from core.config import PYTHON_EXEC_TIMEOUT # Default timeout
from core.code_safety import is_safe_ast

# Initialize logger
logger = logging.getLogger(__name__)

def skill_execute_code(action_input: dict) -> dict:
    """
    Executa um bloco de código (atualmente apenas Python) em um sandbox Firejail.
    Realiza uma análise AST básica para segurança antes da execução.

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
    code_preview = code_to_execute[:100] + ("..." if len(code_to_execute) > 100 else "")
    logger.debug(f"Código (prévia):\n---\n{code_preview}\n---")

    # --- Safety Check (AST) ---
    is_safe, safety_message = is_safe_ast(code_to_execute)
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