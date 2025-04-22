# --- INÍCIO DO CÓDIGO PARA COPIAR ---
import subprocess
import logging
from typing import Dict, Any, Optional, Tuple
import shutil # Added for shutil.which
import tempfile # Added for temporary files
import os # Added for path manipulation
from a3x.core.skills import skill  # <<< Update import
from a3x.core.db_utils import add_episodic_record # <<< Corrected import name
from a3x.core.context import Context # Added import
from a3x.core.context import SharedTaskContext
from a3x.core.context_accessor import ContextAccessor
from a3x.core.code_safety import is_safe_ast
import a3x.core.config as config # <<< Added import for config flag

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize a ContextAccessor instance for use in skills
_context_accessor = ContextAccessor()

# Renamed function, added decorator, and updated signature
@skill(
    name="execute_code",
    description="Executes a given code snippet in a specified language within a sandboxed environment.",
    parameters={
        "code": {"type": str, "description": "The code snippet to execute."},
        "language": {"type": Optional[str], "default": "python", "description": "The programming language of the snippet (default: python)."},
        "timeout": {"type": Optional[int], "default": 60, "description": "Maximum execution time in seconds (default: 60)."},
        "shared_task_context": {"type": "Optional[a3x.core.context.SharedTaskContext]", "description": "The shared context for accessing task-related data.", "optional": True}
    }
)
def execute_code(
    context: Context,
    code: str,
    language: str = "python",
    timeout: int = 60,
    shared_task_context: Optional[SharedTaskContext] = None
) -> dict:
    """
    Executa um bloco de código Python em um sandbox Firejail.
    Realiza uma análise AST básica para segurança antes da execução.
    Pode usar o `shared_task_context` para resolver placeholders como `$LAST_READ_FILE`.

    Args:
        context (SkillContext): The execution context provided by the agent.
        code (str): The Python code to execute.
        language (str, optional): The programming language (must be 'python'). Defaults to "python".
        timeout (float, optional): Maximum execution time in seconds. Defaults to ... (implicit default).
        shared_task_context (SharedTaskContext, optional): The shared context for the current task.

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
    
    # --- ADDED: Resolve placeholders using SharedTaskContext --- 
    resolved_successfully = True
    if shared_task_context:
        placeholders = {
            "$LAST_READ_FILE": "last_file_read_path",
            # Add more placeholders as needed
        }
        for placeholder, context_key in placeholders.items():
            if placeholder in code_to_execute:
                resolved_value = shared_task_context.get(context_key)
                if resolved_value:
                    # Simple string replacement - might need more robust templating later
                    logger.info(f"Resolving placeholder '{placeholder}' with value from context key '{context_key}'.")
                    code_to_execute = code_to_execute.replace(placeholder, str(resolved_value))
                else:
                    logger.warning(f"Placeholder '{placeholder}' found, but key '{context_key}' not found in shared context.")
                    # Decide how to handle: error out or execute with placeholder?
                    # For now, let's error out to be safe.
                    resolved_successfully = False
                    return {
                        "status": "error",
                        "action": "execution_failed_placeholder_unresolved",
                        "data": {"message": f"Could not resolve placeholder '{placeholder}' from shared context key '{context_key}'."}
                    }
    if not resolved_successfully:
        # This return should ideally not be reachable due to the return inside the loop, but safeguard.
        return { "status": "error", "action": "execution_failed_placeholder", "data": {"message": "Failed to resolve one or more placeholders from shared context."} }
    # --- END ADDED --- 

    # Validate timeout value (must be positive)
    timeout_sec = 60  # Explicit default
    try:
        if timeout is not None:
            parsed_timeout = float(timeout)
            if parsed_timeout > 0:
                timeout_sec = parsed_timeout
            else:
                logger.warning(
                    f"Provided timeout ({timeout}) is not positive. Using default: {timeout_sec}s"
                )
        # If timeout is None, the default timeout_sec is already set
    except (ValueError, TypeError):
        logger.warning(
            f"Invalid timeout type provided: {type(timeout)}. Using default: {timeout_sec}s"
        )
        # Default timeout_sec is already set

    # Validate supported language
    if language != "python":
        logger.error(f"Unsupported language specified: '{language}'")
        return {
            "status": "error",
            "action": "execution_failed_language",
            "data": {
                "message": f"Language not supported: '{language}'. Only 'python' is currently supported."
            },
        }

    logger.info(f"Attempting to execute {language} code with timeout {timeout_sec}s.")
    code_preview = code_to_execute[:100] + ("..." if len(code_to_execute) > 100 else "")
    logger.debug(f"Code Preview:\n---\n{code_preview}\n---")

    # --- Safety Check using refined AST from core.code_safety ---
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

    # --- Execute Code ---
    stdout_result = ""
    stderr_result = ""
    exit_code = -1
    execution_method = "unknown" # To track how it was run

    firejail_path = shutil.which("firejail")
    use_firejail = config.USE_FIREJAIL_SANDBOX and firejail_path

    if use_firejail:
        execution_method = "firejail"
        logger.info("Firejail found and enabled. Executing code in sandboxed environment.")
        try:
            # Construir comando Firejail
            firejail_command = [
                firejail_path, # Use found path
                "--quiet",
                "--noprofile",
                "--net=none",
                "--private",
                "--seccomp",
                "--nonewprivs",
                "--noroot",
                "python3",
                "-c",
                code_to_execute,
            ]

            logger.debug(
                f"Executando comando via Firejail: {' '.join(firejail_command[:8])} python3 -c '...'"
            )

            process = subprocess.run(
                firejail_command,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            stdout_result = process.stdout.strip()
            stderr_result = process.stderr.strip()
            exit_code = process.returncode

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout ({timeout_sec}s) atingido durante a execução com Firejail.")
            outcome = f"failure: timeout ({timeout_sec}s)"
            metadata = {"timeout_value": timeout_sec, "method": execution_method}
            # <<< Registrar Experiência >>>
            try:
                add_episodic_record(context="execute_code skill", action=code_to_execute, outcome=outcome, metadata=metadata)
            except Exception as db_err:
                logger.error(f"Failed to record timeout experience: {db_err}")
            return {
                "status": "error",
                "action": "execution_failed",
                "data": {
                    "message": f"Execução do código (via Firejail) excedeu o limite de tempo ({timeout_sec}s)."
                },
            }
        except Exception as e:
             logger.exception(f"Erro inesperado durante execução com Firejail: {e}")
             outcome = f"failure: unexpected error ({type(e).__name__})"
             metadata = {"error": str(e), "method": execution_method}
             # <<< Registrar Experiência >>>
             try:
                 add_episodic_record(context="execute_code skill", action=code_to_execute if 'code_to_execute' in locals() else "N/A - Early Error", outcome=outcome, metadata=metadata)
             except Exception as db_err:
                 logger.error(f"Failed to record unexpected error experience: {db_err}")
             return {
                 "status": "error",
                 "action": "execution_failed",
                 "data": {"message": f"Erro inesperado durante execução com Firejail: {e}"},
             }

    else: # Firejail not found OR disabled by config
        execution_method = "direct_python"
        if not config.USE_FIREJAIL_SANDBOX:
            logger.warning("Firejail usage is disabled by configuration (USE_FIREJAIL_SANDBOX=False). Falling back to direct Python execution (less secure).")
        elif not firejail_path:
            logger.warning("Firejail executable not found in PATH. Falling back to direct Python execution (less secure).")
        else:
             # Should not happen given the logic, but include for completeness
             logger.warning("Firejail disabled for unknown reason. Falling back to direct Python execution (less secure).")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                script_path = os.path.join(tmpdir, "temp_script.py")
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(code_to_execute)

                logger.debug(f"Executing script directly: python3 {script_path}")

                process = subprocess.run(
                    ["python3", script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                    cwd=tmpdir # Run script from its own directory
                )
                stdout_result = process.stdout.strip()
                stderr_result = process.stderr.strip()
                exit_code = process.returncode

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout ({timeout_sec}s) atingido durante a execução direta.")
            outcome = f"failure: timeout ({timeout_sec}s)"
            metadata = {"timeout_value": timeout_sec, "method": execution_method}
            # <<< Registrar Experiência >>>
            try:
                add_episodic_record(context="execute_code skill", action=code_to_execute, outcome=outcome, metadata=metadata)
            except Exception as db_err:
                logger.error(f"Failed to record timeout experience: {db_err}")
            return {
                "status": "error",
                "action": "execution_failed",
                "data": {
                    "message": f"Execução do código (via direct python) excedeu o limite de tempo ({timeout_sec}s)."
                },
            }
        except FileNotFoundError:
             # This would likely mean 'python3' is not found
             logger.error("Comando 'python3' não encontrado. Verifique a instalação e o PATH.", exc_info=True)
             outcome = "failure: python3 not found"
             metadata = {"method": execution_method}
             # <<< Registrar Experiência >>>
             try:
                 add_episodic_record(context="execute_code skill", action="N/A - Environment Error", outcome=outcome, metadata=metadata)
             except Exception as db_err:
                 logger.error(f"Failed to record environment error experience: {db_err}")
             return {
                 "status": "error",
                 "action": "execution_failed_runtime_missing",
                 "data": {"message": "Comando 'python3' não encontrado."},
             }
        except Exception as e:
             logger.exception(f"Erro inesperado durante execução direta: {e}")
             outcome = f"failure: unexpected error ({type(e).__name__})"
             metadata = {"error": str(e), "method": execution_method}
             # <<< Registrar Experiência >>>
             try:
                 add_episodic_record(context="execute_code skill", action=code_to_execute if 'code_to_execute' in locals() else "N/A - Early Error", outcome=outcome, metadata=metadata)
             except Exception as db_err:
                 logger.error(f"Failed to record unexpected error experience: {db_err}")
             return {
                 "status": "error",
                 "action": "execution_failed",
                 "data": {"message": f"Erro inesperado durante execução direta: {e}"},
             }

    # --- Process Results (Common logic for both methods) ---
    logger.info(f"Execução ({execution_method}) concluída. Exit Code: {exit_code}")
    if stdout_result:
        logger.debug(f"Stdout: {stdout_result}")
    if stderr_result:
        logger.debug(f"Stderr: {stderr_result}")

    if exit_code == 0:
        outcome = "success"
        metadata = {"returncode": exit_code, "stdout_len": len(stdout_result), "stderr_len": len(stderr_result), "method": execution_method}
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
                "stderr": stderr_result,
                "returncode": exit_code,
                "message": f"Código executado com sucesso (via {execution_method}).",
            },
        }
    else:
        error_cause = f"erro durante a execução (exit code: {exit_code})"
        if "SyntaxError:" in stderr_result:
            error_cause = "erro de sintaxe"
        elif "Traceback (most recent call last):" in stderr_result:
            error_cause = "erro de runtime"

        outcome = f"failure: {error_cause}"
        metadata = {"returncode": exit_code, "error_cause": error_cause, "stderr_preview": stderr_result[:200], "method": execution_method}
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
                "message": f"Falha na execução do código (via {execution_method}): {error_cause}. Stderr: {stderr_result[:200]}...",
            },
        }

    # Update context with execution results using ContextAccessor
    result_payload = {
        "status": "success" if exit_code == 0 else "error",
        "action": "code_executed",
        "data": {
            "stdout": stdout_result,
            "stderr": stderr_result,
            "returncode": exit_code,
            "message": "Código executado com sucesso." if exit_code == 0 else f"Falha na execução do código: {error_cause}. Stderr: {stderr_result[:200]}...",
        },
    }
    _context_accessor.set_last_execution_result(result_payload)
    logger.info(f"Updated context with last execution result")

    return result_payload


# --- FIM DO CÓDIGO PARA COPIAR ---
