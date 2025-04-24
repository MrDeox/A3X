import os
import logging
import importlib
import pkgutil
import sys
from typing import Dict, Any, List, Set

from a3x.core.skills import skill, discover_skills, SKILL_REGISTRY
from a3x.core.config import PROJECT_ROOT, SKILL_PACKAGES
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
    description="Recarrega dinamicamente skills que foram auto-geradas pelo agente, permitindo que novas capacidades sejam usadas sem reiniciar.",
    parameters={}
)
async def reload_auto_generated_skills(ctx: Any) -> Dict[str, Any]:
    """Dynamically reloads skills from the auto_generated skills directory."""
    log_prefix = "[ReloadAutoSkills]"
    logger.info(f"{log_prefix} Attempting to reload auto-generated skills from: {AUTO_GENERATED_DIR}")

    # Check if the directory exists
    if not os.path.isdir(AUTO_GENERATED_DIR):
        msg = f"Auto-generated skills directory not found: {AUTO_GENERATED_DIR}"
        logger.warning(f"{log_prefix} {msg}")
        return {"status": "warning", "data": {"message": msg}}

    # Get the list of currently registered skill names before reloading
    # Use the global SKILL_REGISTRY dictionary keys
    skills_before = set(SKILL_REGISTRY.keys())
    count_before = len(skills_before)

    # Call the main skill loading function, targeting only the auto-generated package
    # The package name should correspond to the directory structure relative to the root
    # where Python searches for modules (e.g., relative to PROJECT_ROOT if it's in sys.path).
    # Assuming 'a3x.skills.auto_generated' is the correct package name.
    auto_skills_package = "a3x.skills.auto_generated"
    logger.info(f"{log_prefix} Triggering skill discovery, focusing on package: {auto_skills_package}")
    try:
        # <<< CHANGED call from load_all_skills to discover_skills >>>
        # Pass the specific directory to discover_skills
        discover_skills(skill_directory=AUTO_GENERATED_DIR)
        # load_all_skills([auto_skills_package]) # Pass as a list
        logger.info(f"{log_prefix} Skill discovery process completed for {auto_skills_package}.")
    except Exception as e:
        msg = f"Error occurred during skill discovery for {auto_skills_package}: {e}"
        logger.exception(f"{log_prefix} {msg}")
        return {"status": "error", "data": {"message": msg}}

    # Check which skills were newly registered
    skills_after = set(SKILL_REGISTRY.keys())
    newly_registered = list(skills_after - skills_before)
    count_after = len(skills_after)

    logger.info(f"{log_prefix} Reload process finished. Before: {count_before}, After: {count_after}. Newly registered: {len(newly_registered)}")
    if newly_registered:
        logger.info(f"{log_prefix} New skills: {newly_registered}")
        msg = f"Successfully reloaded auto-generated skills. {len(newly_registered)} new skill(s) registered: {newly_registered}"
    else:
        logger.info(f"{log_prefix} No new skills were registered.")
        msg = "Reload completed, but no new auto-generated skills were found or registered."

    return {"status": "success", "data": {"message": msg, "new_skills": newly_registered}}

# Helper function to ensure project root is correct (if not already in config)
# You might need to adjust this based on your actual config structure
# Example placeholder if get_project_root is not directly available:
# def get_project_root():
#     return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')) 