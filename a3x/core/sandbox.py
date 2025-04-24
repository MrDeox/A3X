# a3x/core/sandbox.py
import logging
import subprocess
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Assuming these core components exist and are importable
import a3x.core.config as config
from a3x.core.context import SharedTaskContext
from a3x.core.code_safety import is_safe_ast
# Assuming db_utils exists and has the function
# If not, this import needs adjustment or the functionality needs to be passed in.
# from a3x.core.db_utils import add_episodic_record 

# Placeholder for the episodic record function if db_utils is not the right place
# This avoids a hard dependency during refactoring if the location is uncertain
# In a real scenario, we'd confirm the correct import path.
def add_episodic_record(context: str, action: str, outcome: str, metadata: Dict):
    # In a real implementation, this would interact with the database/logging system
    print(f"[Episodic Record] Context: {context}, Action: {action[:100]}..., Outcome: {outcome}, Metadata: {metadata}")
    pass


logger = logging.getLogger(__name__)

# Default timeout if not specified
DEFAULT_TIMEOUT_SECONDS = 60

def _resolve_placeholders(code: str, shared_context: Optional[SharedTaskContext]) -> Tuple[str, bool]:
    """Resolves placeholders like $LAST_READ_FILE in the code snippet."""
    if not shared_context:
        return code, True # No context, nothing to resolve

    code_to_execute = code
    resolved_successfully = True
    placeholders = {
        # Example placeholder - extend this dict as needed
        "$LAST_READ_FILE": "last_file_read_path", 
    }

    for placeholder, context_key in placeholders.items():
        if placeholder in code_to_execute:
            resolved_value = shared_context.get(context_key)
            if resolved_value:
                logger.info(f"Resolving placeholder '{placeholder}' with value from context key '{context_key}'.")
                # Simple string replacement - might need more robust templating
                code_to_execute = code_to_execute.replace(placeholder, str(resolved_value))
            else:
                logger.warning(f"Placeholder '{placeholder}' found, but key '{context_key}' not found in shared context.")
                resolved_successfully = False
                break # Stop resolving on first failure

    return code_to_execute, resolved_successfully

def _run_with_firejail(code: str, language: str, timeout: int) -> Dict[str, Any]:
    """Executes the code snippet using Firejail."""
    firejail_path = shutil.which("firejail")
    if not firejail_path:
        return {"status": "error", "stderr": "Firejail executable not found in PATH.", "stdout": "", "exit_code": -1, "method": "firejail_missing"}

    if language != "python":
         return {"status": "error", "stderr": f"Firejail execution only supports python, got {language}", "stdout": "", "exit_code": -1, "method": "firejail"}

    # Use python3 -c for direct code execution within firejail
    firejail_command = [
        firejail_path,
        "--quiet", "--noprofile", "--net=none", "--private", 
        "--seccomp", "--nonewprivs", "--noroot",
        "python3", "-c", code,
    ]
    logger.info("Executing code via Firejail.")
    logger.debug(f"Firejail command: {' '.join(firejail_command[:8])} python3 -c '...'")

    try:
        process = subprocess.run(
            firejail_command,
            capture_output=True, text=True, timeout=timeout, check=False,
            encoding='utf-8', errors='replace'
        )
        return {
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip(),
            "exit_code": process.returncode,
            "status": "success" if process.returncode == 0 else "error",
            "method": "firejail"
        }
    except subprocess.TimeoutExpired:
        logger.warning(f"Firejail execution timed out after {timeout} seconds.")
        return {"status": "timeout", "stderr": f"Execution timed out after {timeout} seconds.", "stdout": "", "exit_code": -1, "method": "firejail"}
    except Exception as e:
        logger.exception(f"Unexpected error during Firejail execution: {e}")
        return {"status": "error", "stderr": f"Unexpected Firejail error: {e}", "stdout": "", "exit_code": -1, "method": "firejail"}

def _run_direct(code: str, language: str, timeout: int) -> Dict[str, Any]:
    """Executes python code directly (less secure fallback)."""
    if language != "python":
        return {"status": "error", "stderr": f"Direct execution only supports python, got {language}", "stdout": "", "exit_code": -1, "method": "direct"}
        
    logger.warning("Executing code directly without Firejail sandbox (less secure).")
    try:
        # Execute using a temporary file to handle multi-line scripts better than -c
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "temp_sandbox_script.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            logger.debug(f"Executing script directly: python3 {script_path}")
            process = subprocess.run(
                ["python3", script_path],
                capture_output=True, text=True, timeout=timeout, check=False,
                cwd=tmpdir, # Run script from its own directory
                encoding='utf-8', errors='replace'
            )
            return {
                "stdout": process.stdout.strip(),
                "stderr": process.stderr.strip(),
                "exit_code": process.returncode,
                "status": "success" if process.returncode == 0 else "error",
                "method": "direct"
            }
    except subprocess.TimeoutExpired:
        logger.warning(f"Direct execution timed out after {timeout} seconds.")
        return {"status": "timeout", "stderr": f"Execution timed out after {timeout} seconds.", "stdout": "", "exit_code": -1, "method": "direct"}
    except Exception as e:
        logger.exception(f"Unexpected error during direct execution: {e}")
        return {"status": "error", "stderr": f"Unexpected direct execution error: {e}", "stdout": "", "exit_code": -1, "method": "direct"}


def execute_code_in_sandbox(
    code: str,
    language: str = "python",
    timeout: Optional[int] = None,
    shared_context: Optional[SharedTaskContext] = None,
    skill_context: Optional[Dict] = None # Pass context for logging
) -> Dict[str, Any]:
    """
    Executes a code snippet in a sandboxed environment, prioritizing Firejail.

    Args:
        code (str): The code snippet to execute.
        language (str): The programming language (currently only "python").
        timeout (Optional[int]): Maximum execution time in seconds. Uses DEFAULT_TIMEOUT_SECONDS if None.
        shared_context (Optional[SharedTaskContext]): Context for resolving placeholders.
        skill_context (Optional[Dict]): Optional context from the skill caller for logging.

    Returns:
        dict: A dictionary containing the execution result:
              {"stdout", "stderr", "exit_code", "status", "method"}
              Status can be "success", "error", "timeout", "ast_block", "placeholder_error".
              Method indicates how it was run ("firejail", "direct", "ast_block", "placeholder_error", "unsupported_language").
    """
    language = language.lower()
    timeout_sec = timeout if timeout is not None and timeout > 0 else DEFAULT_TIMEOUT_SECONDS
    
    calling_skill_name = skill_context.get("skill_name", "unknown_skill") if skill_context else "unknown_context"
    log_context_base = f"sandbox execution called by {calling_skill_name}"

    if language != "python":
        logger.error(f"Unsupported language specified: '{language}'")
        return {"status": "error", "stderr": f"Language not supported: {language}", "stdout": "", "exit_code": -1, "method": "unsupported_language"}

    # 1. Resolve Placeholders
    code_to_execute, resolved = _resolve_placeholders(code, shared_context)
    if not resolved:
         logger.error("Failed to resolve placeholders in code snippet.")
         return {"status": "error", "stderr": "Failed to resolve placeholders using shared context.", "stdout": "", "exit_code": -1, "method": "placeholder_error"}

    # 2. Safety Check (AST) - Apply this regardless of execution method for defense-in-depth
    # Especially important if falling back to direct execution.
    is_safe, safety_message = is_safe_ast(code_to_execute)
    if not is_safe:
        logger.warning(f"Execution blocked by AST analysis: {safety_message}")
        outcome = f"failure: AST analysis blocked - {safety_message}"
        metadata = {"reason": "ast_block", "message": safety_message}
        try:
            add_episodic_record(context=log_context_base, action=code_to_execute, outcome=outcome, metadata=metadata)
        except Exception as db_err:
            logger.error(f"Failed to record AST block experience: {db_err}")
        return {"status": "ast_block", "stderr": f"Execution blocked by AST analysis: {safety_message}", "stdout": "", "exit_code": -1, "method": "ast_block"}
    logger.debug(f"AST analysis passed: {safety_message}")

    # 3. Execute (Try Firejail first if enabled)
    result: Dict[str, Any] = {}
    if config.USE_FIREJAIL_SANDBOX:
        result = _run_with_firejail(code_to_execute, language, timeout_sec)
        # If firejail executable wasn't found, result contains 'firejail_missing' method
        # We might want to fallback ONLY if firejail is missing, not if it failed for other reasons.
        # Let's adjust: Fallback only if firejail itself is missing, otherwise trust its result/error.
        if result.get("method") != "firejail_missing":
            logger.info(f"Firejail execution attempt completed with status: {result.get('status')}")
            # Don't fallback if firejail ran but failed/timed out.
        else: 
            # Firejail executable not found, try direct method as fallback
            logger.warning("Firejail executable not found. Falling back to direct execution.")
            result = _run_direct(code_to_execute, language, timeout_sec)
            
    else: # Firejail disabled in config
        logger.info("Firejail is disabled. Using direct execution.")
        result = _run_direct(code_to_execute, language, timeout_sec)

    # 4. Log Experience
    outcome = f"{result.get('status')}: exit_code={result.get('exit_code')}"
    metadata = {
        "method": result.get("method"),
        "exit_code": result.get("exit_code"),
        "timeout_value": timeout_sec,
        "stderr_preview": result.get("stderr", "")[:100] # Log snippet of stderr
    }
    try:
        add_episodic_record(context=log_context_base, action=code_to_execute, outcome=outcome, metadata=metadata)
    except Exception as db_err:
        logger.error(f"Failed to record execution experience: {db_err}")

    # Ensure 'method' is always in the final result dict for clarity
    if "method" not in result:
        result["method"] = "unknown" # Should not happen, but safeguard

    return result

# Example usage (for testing purposes, would not be in final module)
# async def main():
#     from a3x.core.context import SharedTaskContext
#     ctx = SharedTaskContext(task_id="test_task")
#     ctx.set("last_file_read_path", "/tmp/my_test_file.txt")
#     # Create dummy file
#     with open("/tmp/my_test_file.txt", "w") as f: f.write("Hello from test file!")
    
#     test_code_placeholder = "with open('$LAST_READ_FILE', 'r') as f:\n    print(f.read())"
#     test_code_simple = "print('Hello from sandbox!')\nimport sys\nsys.exit(0)"
#     test_code_error = "print('Error incoming!')\nraise ValueError('Test error')"
#     test_code_unsafe = "import os\nos.system('echo unsafe')"

#     print("--- Testing Placeholder Resolution ---")
#     result = execute_code_in_sandbox(test_code_placeholder, shared_context=ctx)
#     print(result)

#     print("--- Testing Simple Execution ---")
#     result_simple = execute_code_in_sandbox(test_code_simple)
#     print(result_simple)

#     print("--- Testing Error Execution ---")
#     result_error = execute_code_in_sandbox(test_code_error)
#     print(result_error)

#     print("--- Testing Unsafe Execution (AST Block) ---")
#     result_unsafe = execute_code_in_sandbox(test_code_unsafe)
#     print(result_unsafe)

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     import asyncio
#     asyncio.run(main()) # If functions were async, otherwise just call main() 