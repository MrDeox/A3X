import os
import logging
import importlib
import pkgutil
import sys
from typing import Dict, Any, List, Set

from a3x.core.skills import skill, load_skills, SKILL_REGISTRY
from a3x.core.config import PROJECT_ROOT
# from a3x.core.config import get_project_root # REMOVED - Function doesn't exist there

logger = logging.getLogger(__name__)

# Define paths relative to the project root
# PROJECT_ROOT = get_project_root() # REMOVED
# Calculate project root dynamically (assuming this file is in a3x/skills/core/)
_current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..', '..')) 

AUTO_GENERATED_DIR = os.path.join(PROJECT_ROOT, "a3x", "skills", "auto_generated")
AUTO_GENERATED_INIT_FILE = os.path.join(AUTO_GENERATED_DIR, "__init__.py")
SKILLS_PACKAGE_NAME = "a3x.skills"
AUTO_GENERATED_PACKAGE_NAME = "a3x.skills.auto_generated"

@skill(
    name="reload_auto_generated_skills",
    description="Detecta novas skills em 'a3x/skills/auto_generated', atualiza o __init__.py e recarrega todas as skills.",
    parameters={} # No parameters needed
)
async def reload_auto_generated_skills(ctx: Any) -> Dict[str, Any]:
    """
    Scans the auto-generated skills directory, updates its __init__.py if needed,
    clears the skill registry, and reloads all skills using load_skills().
    """
    ctx.logger.info("Starting reload of auto-generated skills...")
    detected_py_files: List[str] = []
    imported_modules: Set[str] = set()
    modules_to_add: List[str] = []
    init_updated: bool = False
    skills_reloaded: bool = False
    status: str = "success"
    message: str = ""

    try:
        # --- Step 1: Scan auto-generated directory ---
        ctx.logger.info(f"Scanning directory: {AUTO_GENERATED_DIR}")
        if not os.path.isdir(AUTO_GENERATED_DIR):
            os.makedirs(AUTO_GENERATED_DIR, exist_ok=True)
            ctx.logger.info(f"Created directory as it did not exist: {AUTO_GENERATED_DIR}")
            # No files to process if directory was just created

        detected_py_files = [
            f for f in os.listdir(AUTO_GENERATED_DIR)
            if f.endswith(".py") and f != "__init__.py"
        ]
        detected_modules = {os.path.splitext(f)[0] for f in detected_py_files}
        ctx.logger.info(f"Detected Python modules: {detected_modules or 'None'}")

        # --- Step 2: Check and update __init__.py ---
        ctx.logger.info(f"Checking __init__.py: {AUTO_GENERATED_INIT_FILE}")
        if not os.path.exists(AUTO_GENERATED_INIT_FILE):
            with open(AUTO_GENERATED_INIT_FILE, "w", encoding="utf-8") as f_init:
                f_init.write("# Auto-generated __init__.py for A³X skills\n")
            ctx.logger.info(f"Created empty __init__.py.")
            current_init_content = ""
        else:
             with open(AUTO_GENERATED_INIT_FILE, "r", encoding="utf-8") as f_init:
                 current_init_content = f_init.read()

        # Find currently imported modules (simple parsing)
        for line in current_init_content.splitlines():
            line = line.strip()
            if line.startswith("from . import") or line.startswith("import ."): # Handle both styles just in case
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "." and parts[2] == "import":
                    imported_modules.add(parts[3])
                elif len(parts) >= 2 and parts[0] == "import" and parts[1].startswith("."):
                     imported_modules.add(parts[1][1:]) # remove leading dot

        ctx.logger.debug(f"Modules currently imported in __init__.py: {imported_modules}")

        # Determine which detected modules are missing from __init__.py
        modules_to_add = sorted(list(detected_modules - imported_modules))

        if modules_to_add:
            ctx.logger.info(f"Adding missing import statements for: {modules_to_add}")
            lines_to_add = [f"from . import {mod}\n" for mod in modules_to_add]

            with open(AUTO_GENERATED_INIT_FILE, "a", encoding="utf-8") as f_init:
                # Add a newline if file doesn't end with one, before appending
                if current_init_content and not current_init_content.endswith("\n"):
                     f_init.write("\n")
                f_init.writelines(lines_to_add)
            init_updated = True
            ctx.logger.info(f"Successfully updated {AUTO_GENERATED_INIT_FILE}.")
        else:
            ctx.logger.info("__init__.py is already up-to-date. No changes needed.")

        # --- Step 3: Reload Skills ---
        ctx.logger.info("Clearing existing skill registry...")
        count_before = len(SKILL_REGISTRY)
        SKILL_REGISTRY.clear()
        ctx.logger.info(f"Registry cleared (was {count_before} skills). Reloading skills...")

        # Reload the skills package. This will re-execute __init__.py files
        # and re-run @skill decorators.
        try:
            # Ensure the main skills package is loaded/reloaded
            # load_skills handles the import logic including subpackages
            load_skills(SKILLS_PACKAGE_NAME)
            count_after = len(SKILL_REGISTRY)
            ctx.logger.info(f"Skills reloaded successfully via load_skills('{SKILLS_PACKAGE_NAME}'). Total skills now: {count_after}")
            skills_reloaded = True
            message = f"Detected {len(detected_py_files)} file(s). Updated __init__: {init_updated}. Reloaded {count_after} skills."

        except Exception as load_err:
            status = "error"
            message = f"Failed to reload skills via load_skills: {load_err}"
            ctx.logger.exception(f"Error during load_skills('{SKILLS_PACKAGE_NAME}'):")
            skills_reloaded = False # Explicitly set false on error

    except Exception as e:
        status = "error"
        message = f"An unexpected error occurred: {e}"
        ctx.logger.exception("Error during skill reload process:")

    return {
        "detected_files": detected_py_files,
        "init_updated": init_updated,
        "skills_reloaded": skills_reloaded,
        "status": status,
        "message": message
    }

# Helper function to ensure project root is correct (if not already in config)
# You might need to adjust this based on your actual config structure
# Example placeholder if get_project_root is not directly available:
# def get_project_root():
#     return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')) 