import json
import logging
from pathlib import Path
import os
from typing import Dict, Any
import asyncio
from filelock import AsyncFileLock, Timeout

# Commenting out hardcoded BASE_DIR. Paths should be handled relative to execution or configured.
# BASE_DIR = Path("a3x/a3net") 

# Configure logging for this utility
logger = logging.getLogger(__name__)

# Ensure the datasets directory exists
# Assuming run.py or main setup handles base path correctly
# Using relative path from where the script might be run or imported
try:
    # Try resolving relative to the utils directory first
    current_dir = Path(__file__).parent
    BASE_DIR = current_dir.parent # a3x/a3net/
    DATASETS_DIR = BASE_DIR / "datasets"
    LOCK_DIR = BASE_DIR / "locks"
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive)
    # This might be less reliable depending on CWD
    BASE_DIR = Path("a3x/a3net")
    DATASETS_DIR = BASE_DIR / "datasets"
    LOCK_DIR = BASE_DIR / "locks"
    logger.warning("Could not resolve path using __file__, using relative path 'a3x/a3net'. Ensure CWD is workspace root.")


DATASETS_DIR.mkdir(parents=True, exist_ok=True)
LOCK_DIR.mkdir(parents=True, exist_ok=True)

async def registrar_exemplo_de_aprendizado(task_name: str, input_data: str, label: str):
    """
    Registra um par input/label em um arquivo .jsonl específico para a tarefa,
    de forma assíncrona e com bloqueio de arquivo.

    Args:
        task_name (str): O nome da tarefa (usado como nome do arquivo).
        input_data (str): O dado de entrada.
        label (str): O rótulo/saída correspondente. Can be any JSON-serializable type.

    Returns:
        bool: True se o exemplo foi registrado com sucesso, False caso contrário.
    """
    if not task_name or not isinstance(task_name, str):
        logger.error("Nome da tarefa inválido ou ausente para registrar exemplo.")
        return False
    if not input_data or not isinstance(input_data, str):
        # Input is expected to be text for most current use cases
        logger.error(f"Dado de entrada inválido ou ausente para tarefa '{task_name}'.")
        return False

    label_to_store = None
    if label is None:
         logger.warning(f"Rótulo ausente para tarefa '{task_name}'. Registrando com rótulo nulo.")
         label_to_store = None # Store actual None
    else:
        try:
            # Check if label is already a string, if not, dump it
            if isinstance(label, str):
                 # Attempt to parse if it looks like JSON, otherwise keep as string
                 try:
                     parsed_label = json.loads(label)
                     label_to_store = parsed_label
                 except json.JSONDecodeError:
                     label_to_store = label # Keep as original string
            else:
                # Try to serialize non-string labels
                json.dumps(label) # Test serializability
                label_to_store = label # Store the original object if serializable
        except TypeError:
            logger.error(f"Rótulo não serializável para JSON para tarefa '{task_name}'. Tipo: {type(label)}, Rótulo: {label!r}")
            return False

    # Sanitize task_name to prevent path traversal issues and ensure valid filename
    safe_task_name = "".join(c for c in task_name if c.isalnum() or c in ('_', '-')).rstrip()
    if not safe_task_name:
        logger.error(f"Nome da tarefa '{task_name}' resultou em um nome de arquivo inválido após sanitização.")
        return False

    file_path = DATASETS_DIR / f"{safe_task_name}.jsonl"
    lock_path = LOCK_DIR / f"{safe_task_name}.jsonl.lock"
    # Use a shared lock object if possible, but separate locks per file are safer
    # Pass timeout during instantiation
    lock = AsyncFileLock(lock_path, timeout=5.0)

    example_obj = {"input": input_data, "label": label_to_store}
    example_line = json.dumps(example_obj, ensure_ascii=False) + '\n' # Use ensure_ascii=False for broader compatibility

    try:
        # Correct pattern: Use the lock object directly as context manager
        async with lock: # Remove timeout from here
            # Append the new example using standard blocking I/O inside run_in_executor
            # This prevents blocking the main async loop for file operations
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: _append_to_file(file_path, example_line))

            logger.info(f"Exemplo registrado com sucesso para tarefa '{safe_task_name}' em {file_path}")
            # Lock is released automatically by 'async with lock:'
            return True
    except Timeout: # Catch the specific Timeout exception from filelock
        logger.error(f"Timeout ({lock.timeout}s) ao tentar adquirir lock para o arquivo {lock_path}. Falha ao registrar exemplo para tarefa '{safe_task_name}'.") # Use lock.timeout
        return False
    except IOError as e:
        logger.error(f"Erro de I/O ao registrar exemplo para tarefa '{safe_task_name}' em {file_path}: {e}")
        return False
    except Exception as e:
        logger.exception(f"Erro inesperado ao registrar exemplo para tarefa '{safe_task_name}': {e}") # Use logger.exception for traceback
        return False

def _append_to_file(path: Path, line: str):
    """Helper function to append a line to a file (blocking)."""
    try:
        with open(path, mode='a', encoding='utf-8') as f:
            f.write(line)
    except IOError as e:
        logger.error(f"Erro de I/O na função helper _append_to_file para {path}: {e}")
        raise # Re-raise the exception to be caught by the caller

# Example usage (for testing purposes)
async def main_test():
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing data_logger...")
    task = "teste_simples"
    success1 = await registrar_exemplo_de_aprendizado(task, "Qual a capital da França?", "Paris")
    success2 = await registrar_exemplo_de_aprendizado(task, "Quanto é 2+2?", {"resultado": 4, "tipo": "matematica"})
    success3 = await registrar_exemplo_de_aprendizado(task, "Input without label", None)
    success4 = await registrar_exemplo_de_aprendizado("invalid/task name", "input", "label") # Should fail sanitization
    success5 = await registrar_exemplo_de_aprendizado("long_task_test", "data input " * 10, "label output")

    logger.info(f"Resultados: {success1}, {success2}, {success3}, {success4}, {success5}")

    # Simulate concurrent writes (simple test)
    await asyncio.gather(
        registrar_exemplo_de_aprendizado("concurrent_test", "write 1", "label 1"),
        registrar_exemplo_de_aprendizado("concurrent_test", "write 2", "label 2"),
        registrar_exemplo_de_aprendizado("concurrent_test", "write 3", "label 3")
    )

if __name__ == '__main__':
    # This allows running the file directly for basic testing
    asyncio.run(main_test())

# --- Optional: Add duplicate checking logic if needed ---
# def check_if_duplicate(file_path: Path, example_data: Dict[str, Any]) -> bool:
#     if not file_path.exists():
#         return False
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 try:
#                     existing_example = json.loads(line.strip())
#                     # Simple check based on input and label matching
#                     if existing_example.get("input") == example_data.get("input") and \
#                        existing_example.get("label") == example_data.get("label"):
#                         return True
#                 except json.JSONDecodeError:
#                     logger.warning(f"[DataLogger] Skipping corrupted line in {file_path}: {line.strip()}")
#                     continue
#         return False
#     except Exception as e:
#         logger.error(f"[DataLogger] Error checking for duplicates in {file_path}: {e}")
#         return False # Assume not duplicate if check fails

# Modify `registrar_exemplo_de_aprendizado` to include check:
# ...
# if check_if_duplicate(file_path, example_data):
#     logger.info(f"[DataLogger] Duplicate example for task '{task_name}' skipped.")
#     return True # Indicate success even if skipped duplicate
# with open(...) as f: ...
# ... 