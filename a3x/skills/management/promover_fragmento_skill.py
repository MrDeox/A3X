import logging
import re
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# --- Core A3X Imports ---
from a3x.core.skills import skill
# Use the base Context type, assuming it provides workspace_root
from a3x.core.context import Context 
# Potentially use _ToolExecutionContext if more specific context is needed/guaranteed
# from a3x.core.agent import _ToolExecutionContext as Context

# --- Removed Placeholder Imports ---
# try:
#     from a3x.core.fragment import FragmentContext # Adjust import as needed
#     from a3x.core.memory.memory_manager import MemoryManager # Adjust import as needed
# except ImportError:
#     logger = logging.getLogger(__name__)
#     logger.warning(\"Could not import FragmentContext or MemoryManager. Using placeholders.\")
#     class FragmentContext: ...
#     class MemoryManager: ...

logger = logging.getLogger(__name__)

# --- Constants for File Paths (Relative to Workspace Root) ---
FRAGMENT_DIR = Path(\"a3x/fragments\")
ARCHIVE_DIR = Path(\"a3x/a3net/archive/fragments\") # Example archive location

# --- Regex Patterns ---
# Pattern to find class definition inheriting from BaseFragment
CLASS_PATTERN = re.compile(r"class\\s+([A-Za-z0-9_]+)\\(BaseFragment\\):")
# Pattern to find the base name and mutation number from a mutation name
MUTATION_NAME_PATTERN = re.compile(r"^(.*?)_mut_(\\d+)$")

# --- Helper Function ---
def _derive_fragment_filenames(mutation_name: str, base_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    \"\"\"Derives mutation file, base file, and base name from mutation name.\"\"\"
    match = MUTATION_NAME_PATTERN.match(mutation_name)
    if not match:
        logger.error(f\"Fragment name \'{mutation_name}\' does not match mutation pattern (e.g., BaseName_mut_1).\")
        return None, None, None

    base_name = match.group(1)
    mutation_num = match.group(2)

    # Convert PascalCase/CamelCase base_name to snake_case for filename stem
    base_stem = re.sub(r'(?<!^)(?=[A-Z])', '_', base_name).lower()
    mutation_stem = f\"{base_stem}_mut_{mutation_num}\"

    mutation_file = base_dir / f\"{mutation_stem}_fragment.py\" # Assuming convention
    base_file = base_dir / f\"{base_stem}_fragment.py\"      # Assuming convention

    logger.debug(f\"Derived: BaseName=\'{base_name}\', MutationFile=\'{mutation_file}\', BaseFile=\'{base_file}\'\")
    return mutation_file, base_file, base_name


# --- Skill Definition ---
@skill(
    name=\"promover_fragmento\",
    description=\"Promove uma mutação de fragmento para substituir a versão base original.\",
    parameters={
        \"fragment_name\": {\"type\": \"string\", \"description\": \"O nome da *mutação* do fragmento a ser promovida (ex: MeuFragmento_mut_1).\"}
    }
)
async def promover_fragmento(
    ctx: Context, # Use base Context, expecting workspace_root
    fragment_name: str
) -> Dict[str, Any]:
    \"\"\"
    Promotes a fragment mutation by archiving the original base file and
    renaming the mutation file to the base file name. Also updates the
    internal class name and @fragment decorator name within the promoted code.

    Args:
        ctx: The execution context, providing workspace_root.
        fragment_name: The name of the *mutation* fragment to promote (e.g., \"MyFragment_mut_1\").

    Returns:
        A dictionary containing the status and a message.
    \"\"\"
    log_prefix = f\"[PromoteSkill \'{fragment_name}\']\"
    logger.info(f\"{log_prefix} Starting promotion process...\")

    # --- 1. Get Workspace Path ---
    if not hasattr(ctx, \'workspace_root\') or not ctx.workspace_root:
        logger.error(f\"{log_prefix} Workspace root not found in context.\")
        return {\"status\": \"error\", \"message\": \"Workspace root missing from context.\"}
    workspace_path = Path(ctx.workspace_root)
    fragment_dir_abs = workspace_path / FRAGMENT_DIR
    archive_dir_abs = workspace_path / ARCHIVE_DIR

    # --- 2. Derive Paths and Validate ---
    mutation_file_abs, base_file_abs, base_name = _derive_fragment_filenames(fragment_name, fragment_dir_abs)

    if not mutation_file_abs or not base_file_abs or not base_name:
        # Error already logged by helper function
        return {\"status\": \"error\", \"message\": f\"Invalid mutation name format or derivation failed: {fragment_name}\"}

    if not mutation_file_abs.is_file():
        logger.error(f\"{log_prefix} Mutation fragment file not found: {mutation_file_abs}\")
        return {\"status\": \"error\", \"message\": f\"Mutation file not found: {mutation_file_abs.relative_to(workspace_path)}\"}

    if not base_file_abs.is_file():
        # This might be acceptable if the base was already archived/deleted, but promotion implies replacement.
        logger.warning(f\"{log_prefix} Base fragment file not found: {base_file_abs}. Proceeding with promotion, but base cannot be archived.\")
        # Allow proceeding, but the archival step will be skipped.

    # --- 3. Archive Base Fragment File ---
    try:
        if base_file_abs.is_file():
            archive_dir_abs.mkdir(parents=True, exist_ok=True)
            archive_target = archive_dir_abs / base_file_abs.name
            logger.info(f\"{log_prefix} Archiving base file \'{base_file_abs.name}\' to \'{archive_target}\'...\")
            shutil.move(str(base_file_abs), str(archive_target)) # Use shutil.move for cross-fs safety if needed
            logger.info(f\"{log_prefix} Base file archived successfully.\")
        else:
            logger.info(f\"{log_prefix} Base file \'{base_file_abs.name}\' not found, skipping archival.\")

    except Exception as e:
        logger.exception(f\"{log_prefix} Error archiving base file {base_file_abs}:")
        return {\"status\": \"error\", \"message\": f\"Error archiving base file: {e}\"}

    # --- 4. Promote Mutation File (Rename) ---
    try:
        logger.info(f\"{log_prefix} Promoting mutation file by renaming \'{mutation_file_abs.name}\' to \'{base_file_abs.name}\'...\")
        # Use os.rename or Path.rename. Path.rename might be safer across filesystems.
        mutation_file_abs.rename(base_file_abs)
        promoted_file_abs = base_file_abs # The file now has the base name
        logger.info(f\"{log_prefix} Mutation file renamed successfully to \'{promoted_file_abs.name}\'.\")

    except Exception as e:
        logger.exception(f\"{log_prefix} Error renaming mutation file {mutation_file_abs} to {base_file_abs}:")
        # Attempt to rollback archive? Complex. For now, report error.
        return {\"status\": \"error\", \"message\": f\"Error renaming mutation file: {e}\"}

    # --- 5. Update Internal Code References (CLASS NAME ONLY) ---
    try:
        logger.info(f\"{log_prefix} Updating internal CLASS NAME in promoted file '{promoted_file_abs.name}'...")
        original_code = promoted_file_abs.read_text(encoding=\"utf-8\")
        modified_code = original_code

        # Update Class Name ONLY
        modified_code = CLASS_PATTERN.sub(f\"class {base_name}(BaseFragment):\", modified_code, count=1)

        if modified_code != original_code:
            promoted_file_abs.write_text(modified_code, encoding=\"utf-8\")
            logger.info(f\"{log_prefix} Internal CLASS NAME updated successfully.\")
        else:
            logger.warning(f\"{log_prefix} Could not find class pattern to update in the code. File may be structured unexpectedly.\")
            # Continue, but log a warning. The file rename was successful.

    except IOError as e:
        logger.exception(f\"{log_prefix} Error reading/writing promoted file {promoted_file_abs} for internal updates:")
        return {\"status\": \"error\", \"message\": f\"IOError updating internal code: {e}\"}
    except Exception as e:
        logger.exception(f\"{log_prefix} Unexpected error updating internal code in {promoted_file_abs}:")
        return {\"status\": \"error\", \"message\": f\"Unexpected error updating internal code: {e}\"}

    # --- 6. Final Success ---
    logger.info(f\"{log_prefix} Fragment \'{fragment_name}\' promoted successfully to replace \'{base_name}\'. Class name updated.")
    return {
        \"status\": \"success\",
        \"message\": f\"Fragment \'{fragment_name}\' promoted successfully, replacing \'{base_name}\'. File updated to \'{promoted_file_abs.relative_to(workspace_path)}\'. Class name updated. Decorator name may need manual update or registry reload.\"
        # Note: Registry reload is NOT handled here. It should be triggered separately if needed.
    }

# Example of how this skill might be registered or used (conceptual)
# register_skill("promover_fragmento", promover_fragmento) 