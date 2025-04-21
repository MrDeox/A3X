# a3x/skills/code_execution.py

import logging
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from a3x.core.skills import skill, SkillContext
from a3x.core.config import PROJECT_ROOT # Assuming PROJECT_ROOT is defined in config

logger = logging.getLogger(__name__)

@skill(
    name="execute_python_in_sandbox",
    description="Executa um script Python dentro de um sandbox Firejail seguro e restrito.",
    parameters={
        "type": "object",
        "properties": {
            "script_path": {
                "type": "string",
                "description": "Caminho relativo ao root do projeto para o script Python a ser executado.",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Lista de argumentos a serem passados para o script.",
                "default": [],
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Tempo máximo em segundos para a execução do script.",
                "default": 60,
            }
        },
        "required": ["script_path"],
    }
)
async def execute_python_in_sandbox(
    context: SkillContext,
    script_path: str,
    args: Optional[List[str]] = None,
    timeout_seconds: int = 60
) -> Dict[str, str | int]:
    """
    Executes a given Python script within a secure Firejail sandbox.

    Args:
        context: The skill execution context.
        script_path: The path to the Python script relative to the project root.
        args: A list of arguments to pass to the script.
        timeout_seconds: Maximum execution time in seconds.

    Returns:
        A dictionary containing the execution result:
        {
            "stdout": Standard output of the script,
            "stderr": Standard error of the script,
            "exit_code": Exit code of the script execution,
            "status": "success", "error", or "timeout"
        }
    """
    if args is None:
        args = []

    if not PROJECT_ROOT:
        logger.error("PROJECT_ROOT not configured. Cannot determine sandbox bind path.")
        return {"status": "error", "stderr": "PROJECT_ROOT configuration missing.", "stdout": "", "exit_code": -1}

    project_root_abs = Path(PROJECT_ROOT).resolve()
    script_abs = (project_root_abs / script_path).resolve()
    venv_python_path = project_root_abs / ".venv" / "bin" / "python" # Assuming venv name is .venv

    # --- Security Checks ---
    # 1. Ensure script is within the project directory
    try:
        script_abs.relative_to(project_root_abs)
    except ValueError:
        logger.error(f"Security Violation: Attempted to execute script outside project root: {script_path}")
        return {"status": "error", "stderr": "Security Error: Script path is outside the project directory.", "stdout": "", "exit_code": -1}

    # 2. Check if python executable exists in venv
    if not venv_python_path.is_file():
         logger.error(f"Python executable not found in virtual environment: {venv_python_path}")
         return {"status": "error", "stderr": f"Virtual environment Python not found at {venv_python_path}", "stdout": "", "exit_code": -1}

    # --- Build Firejail Command ---
    # Using /sandbox as the internal mount point for clarity - NOT NEEDED WITH WHITELIST
    # sandbox_internal_path = "/sandbox"
    # script_sandbox_path = Path(sandbox_internal_path) / script_abs.relative_to(project_root_abs)
    # venv_sandbox_python = Path(sandbox_internal_path) / venv_python_path.relative_to(project_root_abs)

    firejail_base_cmd = [
        "firejail",
        "--quiet",
        "--noprofile",
        "--net=none",
        # Remove --private and --bind
        # f"--private", 
        # f"--bind={project_root_abs},{sandbox_internal_path}", 
        # Whitelist the project root and the venv python explicitly
        f"--whitelist={project_root_abs}",
        # Add other necessary paths if needed (e.g., system libraries python might depend on)
        # f"--whitelist=/usr/lib/python3.13", # Example - adjust to your python version/location if needed
    ]

    # Command to execute inside the sandbox - paths are now absolute from the real root
    script_args_str = " ".join([f'"{arg}"' for arg in args]) # Basic quoting
    # Execute python directly using its absolute path, change CWD to project root
    inner_cmd = f"cd {project_root_abs} && {venv_python_path} {script_abs} {script_args_str}"
    
    full_cmd = firejail_base_cmd + ["sh", "-c", inner_cmd]

    logger.info(f"Executing in sandbox: {' '.join(full_cmd)}") # Log the command for debugging

    # --- Execute the Command ---
    try:
        process = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,  # Don't raise exception on non-zero exit code
            encoding='utf-8',
            errors='replace' # Handle potential decoding errors
        )
        logger.info(f"Sandbox execution finished. Exit code: {process.returncode}")
        return {
            "stdout": process.stdout,
            "stderr": process.stderr,
            "exit_code": process.returncode,
            "status": "success" if process.returncode == 0 else "error"
        }
    except subprocess.TimeoutExpired:
        logger.warning(f"Sandbox execution timed out after {timeout_seconds} seconds for script: {script_path}")
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout_seconds} seconds.",
            "exit_code": -1, # Indicate timeout
            "status": "timeout"
        }
    except FileNotFoundError:
         logger.error("`firejail` command not found. Is Firejail installed and in PATH?")
         return {"status": "error", "stderr": "Firejail command not found.", "stdout": "", "exit_code": -1}
    except Exception as e:
        logger.exception(f"Unexpected error executing script in sandbox: {script_path}")
        return {
            "stdout": "",
            "stderr": f"Unexpected sandbox execution error: {e}",
            "exit_code": -1,
            "status": "error"
        } 