import os
import json
import logging
from typing import List, Dict, Any

# Define the path relative to the project root
# Assuming project structure like A3X/a3x/core/...
try:
    # Find project root assuming this file is in a3x/core/
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError: # __file__ might not be defined in some contexts
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

LOG_DIR = os.path.join(_PROJECT_ROOT, "memory", "llm_logs")
LOG_FILE_PATH = os.path.join(LOG_DIR, 'decision_reflections.jsonl')

logger = logging.getLogger(__name__)

def load_recent_reflection_logs(n: int = 10) -> List[Dict[str, Any]]:
    """Lê os últimos n logs da skill de reflexão e retorna como lista de dicionários.

    Args:
        n: Número de logs recentes a serem lidos.

    Returns:
        Lista de dicionários, cada um representando um log JSON. Retorna lista vazia se o
        arquivo não existir ou ocorrerem erros.
    """
    logs = []
    if not os.path.exists(LOG_FILE_PATH):
        logger.warning(f"Reflection log file not found: {LOG_FILE_PATH}")
        return logs

    try:
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            # Read all lines into memory (potentially inefficient for very large files)
            # For large files, consider reading lines in reverse or using deque
            all_lines = f.readlines()

        # Get the last n lines
        recent_lines = all_lines[-n:]

        for i, line in enumerate(recent_lines):
            try:
                log_entry = json.loads(line.strip())
                logs.append(log_entry)
            except json.JSONDecodeError as json_err:
                # Log error with line number relative to the *end* of the file
                line_num_from_end = len(recent_lines) - i
                logger.error(f"Failed to parse JSON from log file {LOG_FILE_PATH} (approx. line -{line_num_from_end}): {json_err}")
                logger.debug(f"Invalid line content: {line.strip()}")
            except Exception as parse_err:
                 line_num_from_end = len(recent_lines) - i
                 logger.error(f"Unexpected error parsing log file {LOG_FILE_PATH} (approx. line -{line_num_from_end}): {parse_err}")

    except IOError as io_err:
        logger.error(f"Failed to read reflection log file {LOG_FILE_PATH}: {io_err}")
    except Exception as e:
        logger.exception(f"Unexpected error loading reflection logs from {LOG_FILE_PATH}:")

    # Ensure the list returned has at most n items, even if fewer were read/parsed
    return logs[-n:] # Slice again just in case parsing errors reduced the count

# Example usage (if run directly)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Attempting to load last 5 reflection logs from: {LOG_FILE_PATH}")
    recent_logs = load_recent_reflection_logs(5)
    if recent_logs:
        logger.info(f"Successfully loaded {len(recent_logs)} logs.")
        # Print timestamps for verification
        for log in recent_logs:
             print(f" - Timestamp: {log.get('timestamp')}")
    else:
        logger.info("No recent logs found or loaded.") 