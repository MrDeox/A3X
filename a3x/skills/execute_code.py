# --- INÍCIO DO CÓDIGO PARA COPIAR ---
import subprocess
import logging
from a3x.core.config import PYTHON_EXEC_TIMEOUT  # Default timeout
from a3x.core.code_safety import is_safe_ast
from a3x.core.skills import skill  # <<< Update import
from a3x.core.db_utils import add_episodic_record # <<< Corrected import name

# Initialize logger
logger = logging.getLogger(__name__)


# Renamed function, added decorator, and updated signature
@skill(
    name="execute_code",
    description="Executes a block of Python code in a secure sandbox environment (Firejail).",
    parameters={
        "code": (str, ...),
        "language": (str, "python"),
        "timeout": (float, PYTHON_EXEC_TIMEOUT),
    },
)
def execute_code(
    code: str, language: str = "python", timeout: float = PYTHON_EXEC_TIMEOUT
) -> dict:
    """
    Executa um bloco de código Python em um sandbox Firejail.
    Realiza uma análise AST básica para segurança antes da execução.

    Args:
        code (str): The Python code to execute.
        language (str, optional): The programming language (must be 'python'). Defaults to "python".
        timeout (float, optional): Maximum execution time in seconds. Defaults to PYTHON_EXEC_TIMEOUT.

    Returns:
        dict: Standardized dictionary with the result of the execution.
    """
    logger.debug(
        f"Executing skill 'execute_code'. Language: {language}, Timeout: {timeout}"
    )

    # Pydantic/Skill framework handles non-empty code and basic type validation via decorator
    # We still need to validate language and timeout value logic

    language = language.lower()  # Ensure lowercase for comparison
    code_to_execute = code  # Use the direct argument

    # Validate timeout value (must be positive)
    timeout_sec = PYTHON_EXEC_TIMEOUT  # Default value
    try:
        if timeout is not None:
            parsed_timeout = float(timeout)
            if parsed_timeout > 0:
                timeout_sec = parsed_timeout
            else:
                logger.warning(
                    f"Provided timeout ({timeout}) is not positive. Using default: {PYTHON_EXEC_TIMEOUT}s"
                )
        # If timeout is None, the default PYTHON_EXEC_TIMEOUT is already set
    except (ValueError, TypeError):
        logger.warning(
            f"Invalid timeout type provided: {type(timeout)}. Using default: {PYTHON_EXEC_TIMEOUT}s"
        )
        # Default PYTHON_EXEC_TIMEOUT is already set

    # Validate supported language
    if language != "python":
        logger.error(f"Unsupported language specified: '{language}'")
        return {
            "status": "error",
            "action": "execution_failed",
            "data": {
                "message": f"Language not supported: '{language}'. Only 'python' is currently supported."
            },
        }

    logger.info(f"Attempting to execute {language} code with timeout {timeout_sec}s.")
    code_preview = code_to_execute[:100] + ("..." if len(code_to_execute) > 100 else "")
    logger.debug(f"Code Preview:\n---\n{code_preview}\n---")

    # --- Safety Check (AST) ---
    is_safe, safety_message = is_safe_ast(code_to_execute)
    if not is_safe:
        logger.warning(f"Execution blocked by AST analysis: {safety_message}")
        outcome = f"failure: AST analysis blocked - {safety_message}"
        metadata = {"reason": "ast_block", "message": safety_message}
        # <<< Registrar Experiência >>>
        try:
            add_episodic_record(context="execute_code skill", action=code_to_execute, outcome=outcome, metadata=metadata)
        except Exception as db_err:
            logger.error(f"Failed to record AST block experience: {db_err}")
        return {
            "status": "error",
            "action": "execution_failed",
            "data": {
                "message": f"Execution blocked for safety reasons (AST analysis): {safety_message}"
            },
        }
    logger.debug(f"AST analysis result: {safety_message}")

    # --- Execute in Firejail ---
    stdout_result = ""
    stderr_result = ""
    exit_code = -1

    try:
        # Construir comando Firejail
        firejail_command = [
            "firejail",
            "--quiet",
            "--noprofile",  # Sandbox básico sem perfil específico
            "--net=none",  # Desabilitar rede
            "--private",  # Diretório home privado, tmpfs /tmp
            "python3",  # Interpretador
            "-c",  # Executar código da string
            code_to_execute,  # O código em si
        ]

        logger.debug(
            f"Executando comando: {' '.join(firejail_command[:6])} python3 -c '...'"
        )  # Log truncado

        # Executar com subprocess.run
        process = subprocess.run(
            firejail_command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,  # Não lança exceção em erro, verificamos o exit_code
        )

        stdout_result = process.stdout.strip()
        stderr_result = process.stderr.strip()
        exit_code = process.returncode

        logger.info(f"Execução concluída. Exit Code: {exit_code}")
        if stdout_result:
            logger.debug(f"Stdout: {stdout_result}")
        if stderr_result:
            logger.debug(f"Stderr: {stderr_result}")

        # --- Return Result ---
        if exit_code == 0:
            outcome = "success"
            metadata = {"returncode": exit_code, "stdout_len": len(stdout_result), "stderr_len": len(stderr_result)}
            # <<< Registrar Experiência >>>
            try:
                add_episodic_record(context="execute_code skill", action=code_to_execute, outcome=outcome, metadata=metadata)
            except Exception as db_err:
                logger.error(f"Failed to record success experience: {db_err}")
            return {
                "status": "success",
                "action": "code_executed",
                "data": {
                    "stdout": stdout_result,
                    "stderr": stderr_result,  # Incluir stderr mesmo em sucesso (para warnings)
                    "returncode": exit_code,
                    "message": "Código executado com sucesso.",
                },
            }
        else:
            error_cause = f"erro durante a execução (exit code: {exit_code})"
            if "SyntaxError:" in stderr_result:
                error_cause = "erro de sintaxe"
            elif "Traceback (most recent call last):" in stderr_result:
                error_cause = "erro de runtime"

            outcome = f"failure: {error_cause}"
            metadata = {"returncode": exit_code, "error_cause": error_cause, "stderr_preview": stderr_result[:200]}
            # <<< Registrar Experiência >>>
            try:
                add_episodic_record(context="execute_code skill", action=code_to_execute, outcome=outcome, metadata=metadata)
            except Exception as db_err:
                logger.error(f"Failed to record failure experience: {db_err}")
            return {
                "status": "error",
                "action": "execution_failed",
                "data": {
                    "stdout": stdout_result,
                    "stderr": stderr_result,
                    "returncode": exit_code,
                    "message": f"Falha na execução do código: {error_cause}. Stderr: {stderr_result[:200]}...",  # Limita stderr na msg
                },
            }

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout ({timeout_sec}s) atingido durante a execução.")
        outcome = f"failure: timeout ({timeout_sec}s)"
        metadata = {"timeout_value": timeout_sec}
        # <<< Registrar Experiência >>>
        try:
            add_episodic_record(context="execute_code skill", action=code_to_execute, outcome=outcome, metadata=metadata)
        except Exception as db_err:
            logger.error(f"Failed to record timeout experience: {db_err}")
        return {
            "status": "error",
            "action": "execution_failed",
            "data": {
                "message": f"Execução do código excedeu o limite de tempo ({timeout_sec}s)."
            },
        }
    except FileNotFoundError:
        logger.error(
            "Comando 'firejail' ou 'python3' não encontrado. Verifique a instalação e o PATH.",
            exc_info=True,
        )
        outcome = "failure: environment error (firejail/python3 not found)"
        metadata = {"error_type": "FileNotFoundError"}
        # <<< Registrar Experiência >>>
        try:
            add_episodic_record(context="execute_code skill", action="N/A - Environment Error", outcome=outcome, metadata=metadata)
        except Exception as db_err:
            logger.error(f"Failed to record environment error experience: {db_err}")
        return {
            "status": "error",
            "action": "execution_failed",
            "data": {
                "message": "Erro de ambiente: 'firejail' ou 'python3' não encontrado."
            },
        }
    except Exception as e:
        logger.exception("Erro inesperado ao tentar executar código:")
        outcome = f"failure: unexpected error - {type(e).__name__}"
        metadata = {"error_type": type(e).__name__, "error_message": str(e)[:200]}
        # <<< Registrar Experiência >>>
        try:
            add_episodic_record(context="execute_code skill", action=code_to_execute if 'code_to_execute' in locals() else "N/A - Early Error", outcome=outcome, metadata=metadata)
        except Exception as db_err:
            logger.error(f"Failed to record unexpected error experience: {db_err}")
        return {
            "status": "error",
            "action": "execution_failed",
            "data": {"message": f"Erro inesperado ao tentar executar código: {e}"},
        }


# --- FIM DO CÓDIGO PARA COPIAR ---
