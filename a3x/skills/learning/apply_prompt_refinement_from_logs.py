"""
Skill para aplicar automaticamente refinamentos ao prompt de outra skill,
baseado nas sugestões geradas pela análise de logs de reflexão.
"""

import logging
import re
import os
import datetime
from typing import Dict, Any, Optional, List
import ast
from pathlib import Path
import shutil

# Core imports
from a3x.core.skills import skill
# Import context type for hinting
from a3x.core.agent import _ToolExecutionContext 
# Remove unused LLM import
# from a3x.core.llm_interface import call_llm
from a3x.core.config import PROJECT_ROOT
# Remove unused imports
# from a3x.skills.core.call_skill_by_name import call_skill_by_name
# from a3x.core.memory_manager import MemoryManager # Module does not exist
# from a3x.core.utils.file_utils import get_most_recent_file # Function does not exist

# Path to the target skill file
# TODO: Make this configurable or discoverable?
TARGET_SKILL_FILE = "skills/simulate/simulate_decision_reflection.py"
PROMPT_START_MARKER = "# --- START SIMULATION PROMPT ---"
PROMPT_END_MARKER = "# --- END SIMULATION PROMPT ---"

logger = logging.getLogger(__name__)

# Path definitions
PROMPT_FILE_DIR = Path(__file__).parent.parent / "prompts"
PROMPT_FILE_PATH = PROMPT_FILE_DIR / "decision_prompt.txt"
PROMPT_BACKUP_DIR = Path(PROJECT_ROOT) / ".a3x" / "prompt_backups"

# --- Skill Definition ---
@skill(
    name="apply_prompt_refinement_from_logs",
    description="Applies a suggested prompt refinement to a target skill's configuration/prompt file.",
    parameters={
        "context": {"type": _ToolExecutionContext, "description": "Execution context for file system access."},
        "target_skill_name": {"type": str, "description": "The name of the skill whose prompt should be updated."},
        "suggested_prompt": {"type": str, "description": "The new suggested prompt content."},
        "prompt_file_path": {"type": Optional[str], "default": None, "description": "Optional explicit path to the prompt file if different from default location."}
    }
)
async def apply_prompt_refinement_from_logs(
    context: _ToolExecutionContext,
    target_skill_name: str,
    suggested_prompt: str,
    prompt_file_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Applies the suggested refined prompt to the main decision_prompt.txt file,
    creating a backup of the old version first.
    """
    context.logger.info(f"Attempting to apply prompt refinement from log: {prompt_file_path}")

    # Ensure backup directory exists
    try:
        PROMPT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        context.logger.error(f"Failed to create prompt backup directory {PROMPT_BACKUP_DIR}: {e}")
        ctx.logger.error(f"Failed to create prompt backup directory {PROMPT_BACKUP_DIR}: {e}")
        return {"status": "error", "message": f"Failed to create backup directory: {e}"}

    # 1. Check if the target prompt file exists
    if not PROMPT_FILE_PATH.is_file():
        ctx.logger.error(f"Target prompt file not found: {PROMPT_FILE_PATH}")
        return {"status": "error", "message": f"Target prompt file {PROMPT_FILE_PATH.name} not found."}

    # 2. Create a timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{PROMPT_FILE_PATH.stem}_{timestamp}{PROMPT_FILE_PATH.suffix}.bak"
    backup_path = PROMPT_BACKUP_DIR / backup_filename
    try:
        shutil.copy2(PROMPT_FILE_PATH, backup_path)
        ctx.logger.info(f"Created backup of current prompt: {backup_path.name}")
    except Exception as e:
        ctx.logger.error(f"Failed to create backup of prompt file {PROMPT_FILE_PATH}: {e}")
        return {"status": "error", "message": f"Failed to backup prompt file: {e}"}

    # 3. Write the new suggested prompt to the original file
    try:
        with open(PROMPT_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(suggested_prompt)
        ctx.logger.info(f"Successfully applied refined prompt to {PROMPT_FILE_PATH.name}.")
        # Manage old backups after successful write
        _manage_old_prompt_backups(PROMPT_BACKUP_DIR, PROMPT_FILE_PATH.stem, ctx.logger)
        return {"status": "success", "message": f"Prompt {PROMPT_FILE_PATH.name} updated successfully."}
    except IOError as e:
        ctx.logger.error(f"Failed to write refined prompt to {PROMPT_FILE_PATH}: {e}")
        # Attempt to restore backup on write failure
        # restored = _restore_latest_backup(PROMPT_FILE_PATH, PROMPT_BACKUP_DIR, ctx.logger)
        return {"status": "error", "message": f"Failed to write new prompt: {e}. Backup created at {backup_path.name}."} # Removed mention of restore attempt
    except Exception as e:
        ctx.logger.exception(f"Unexpected error writing refined prompt to {PROMPT_FILE_PATH}:")
        # Attempt to restore backup on unexpected failure
        # restored = _restore_latest_backup(PROMPT_FILE_PATH, PROMPT_BACKUP_DIR, ctx.logger)
        return {"status": "error", "message": f"Unexpected error writing prompt: {e}. Backup created at {backup_path.name}."} # Removed mention of restore attempt

# --- Helper Functions ---

def _manage_old_prompt_backups(backup_dir: Path, file_stem: str, logger: logging.Logger):
    """Keeps only the most recent N backups for a given prompt file stem."""
    max_backups = 5 # Keep last 5 backups
    try:
        backups = sorted(
            backup_dir.glob(f"{file_stem}_*.bak"),
            key=os.path.getmtime,
            reverse=True, # Newest first
        )

        if len(backups) > max_backups:
            backups_to_delete = backups[max_backups:]
            logger.info(f"Managing old prompt backups for '{file_stem}'. Found {len(backups)}, keeping {max_backups}.")
            for old_backup in backups_to_delete:
                try:
                    old_backup.unlink()
                    logger.debug(f"Deleted old prompt backup: {old_backup.name}")
                except OSError as del_err:
                    logger.warning(f"Failed to delete old prompt backup '{old_backup.name}': {del_err}")
    except Exception as e:
        logger.exception(f"Error managing old prompt backups in {backup_dir} for {file_stem}: {e}")


# def _restore_latest_backup(target_path: Path, backup_dir: Path, logger: logging.Logger) -> bool:
#     """Restores the most recent .bak file from the backup directory to the target path."""
#     try:
#         # Find the most recent backup file - COMMENTED OUT - requires get_most_recent_file
#         # latest_backup = get_most_recent_file(backup_dir, f"{target_path.stem}_*.bak")
#         latest_backup = None # Placeholder
#         if latest_backup and latest_backup.is_file():
#             shutil.copy2(latest_backup, target_path)
#             logger.info(f"Restored prompt {target_path.name} from backup {latest_backup.name}.")
#             return True
#         else:
#             logger.warning(f"No suitable backup found in {backup_dir} to restore for {target_path.name}.")
#             return False
#     except Exception as e:
#         logger.exception(f"Error restoring backup for {target_path.name}: {e}")
#         return False

# TODO:
# - Implement actual ctx.run_skill or equivalent mechanism.
# - Implement actual ctx.read_file / ctx.write_file or use FileManager skill.
# - Refine error handling and logging.
# - Consider more robust prompt extraction/replacement if markers fail. 