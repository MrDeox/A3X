# skills/__init__.py
import logging
import pkgutil
import importlib
import time
# No longer need atexit import here unless used elsewhere

logger = logging.getLogger(__name__)

# Import specific skills directly if needed for type hinting or specific logic
from .management import reorganizar_fragmentos_skill  # Register reorganizar_fragmentos skill

# Autodiscover and import all modules in this package
# This ensures @skill decorators are executed
__all__ = []
# Explicitly skip files that are being removed/consolidated
SKIPPED_MODULES = {
    "manage_files",
    "write_file",
    "read_file",
    "list_files",
    "delete_file",
    # "perception", # Restore - let's see if it crashes first
    # "web_search", # Restore - let's see if it crashes first
}

# Keep existing skill registration logic here
# ... (skill_registry, SkillInfo, etc.) ...
_skill_registry = {}

class SkillInfo:
    # ... (existing SkillInfo class) ...
    pass

def skill(name=None, description=None, parameters=None, examples=None):
    # ... (existing skill decorator) ...
    pass

def get_skill(name):
    # ... (existing get_skill function) ...
    pass

def list_skills():
    # ... (existing list_skills function) ...
    pass

def load_skills(reload=False):
    global _skill_registry
    if reload:
        _skill_registry = {} # Clear registry on reload


    for _, name, is_pkg in pkgutil.walk_packages(__path__, prefix=__name__ + '.'):
        # Skip __init__ modules themselves if necessary, though walk_packages usually handles this
        if name.endswith('.__init__'):
            continue

        relative_name = name.split('.')[-1]

        # time.sleep(0.01) # Reduced sleep
        logger.info(f"---> Attempting to import skill module: {relative_name} (is_pkg={is_pkg})")
        try:

            module = importlib.import_module(name)

            logger.info(f"---> Successfully imported skill module: {relative_name}")
            # Optional: Reload nested skills if the module supports it
            # if reload and hasattr(module, 'load_skills'):
            #     module.load_skills(reload=True)
        except Exception as e:
            logger.error(f"Failed to import skill module {name}: {e}", exc_info=True)

        # time.sleep(0.01) # Reduced sleep



    # Removed atexit check here



    # Explicitly import specific modules after dynamic loading
    # <<< REMOVED explicit import of file_manager to prevent circular dependency >>>
    # try:
    #     print("*** DEBUG: --> Trying explicit import: .file_manager", flush=True)
    #     from . import file_manager
    #     logger.info("Imported file_manager skill.")
    #     print("*** DEBUG: <-- Finished explicit import: .file_manager", flush=True)
    # except ImportError as e:
    #     logger.warning(f"Could not import file_manager skill: {e}")
    #     print(f"*** WARNING: Could not import .file_manager: {e}", flush=True)
    # except Exception as e:
    #     logger.error(f"Error importing file_manager skill: {e}", exc_info=True)
    #     print(f"*** ERROR: Could not import .file_manager: {e}", flush=True)



# This file now primarily serves to mark the 'skills' directory as a Python package.
# Skill loading and registration logic has been moved to skills.loader

# Optionally, expose key functions from the loader for convenience
# from .loader import get_skill, list_skills 

# It's generally better practice to require explicit imports from the loader module, 
# e.g., from a3x.skills.loader import get_skill
# rather than re-exporting here.

logger.debug("a3x.skills package initialized. Load skills via skills.loader.load_skills().")

# Descoberta e registro de skills deve ser feita explicitamente na inicialização da aplicação principal (ex: onde ToolRegistry é criado).
