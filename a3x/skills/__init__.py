# skills/__init__.py
import logging
import pkgutil
import importlib
import time
# No longer need atexit import here unless used elsewhere

logger = logging.getLogger(__name__)

# Import specific skills directly if needed for type hinting or specific logic
# from . import specific_skill_module

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

    print("*** DEBUG: Entering walk_packages loop in skills/__init__.py ***", flush=True)
    time.sleep(0.01) # Reduced sleep
    for _, name, is_pkg in pkgutil.walk_packages(__path__, prefix=__name__ + '.'):
        # Skip __init__ modules themselves if necessary, though walk_packages usually handles this
        if name.endswith('.__init__'):
            continue

        relative_name = name.split('.')[-1]
        print(f"*** DEBUG: walk_packages found: {relative_name} (is_pkg={is_pkg}) ***", flush=True)
        # time.sleep(0.01) # Reduced sleep
        logger.info(f"---> Attempting to import skill module: {relative_name} (is_pkg={is_pkg})")
        try:
            print(f"*** DEBUG: --- Importing: {name} ---", flush=True)
            # time.sleep(0.01) # Reduced sleep
            module = importlib.import_module(name)
            print(f"*** DEBUG: +++ Imported: {name} +++", flush=True)
            # time.sleep(0.01) # Reduced sleep
            logger.info(f"---> Successfully imported skill module: {relative_name}")
            # Optional: Reload nested skills if the module supports it
            # if reload and hasattr(module, 'load_skills'):
            #     module.load_skills(reload=True)
        except Exception as e:
            logger.error(f"Failed to import skill module {name}: {e}", exc_info=True)
            print(f"*** ERROR: Failed to import {name}: {e} ***", flush=True) # Add error print
        # time.sleep(0.01) # Reduced sleep

    print("*** DEBUG: Exited walk_packages loop in skills/__init__.py ***", flush=True)
    time.sleep(0.01) # Reduced sleep

    # Removed atexit check here

    print("*** DEBUG: Passed walk_packages. About to try importing .file_manager ***", flush=True)
    time.sleep(0.01) # Reduced sleep

    # Explicitly import specific modules after dynamic loading
    try:
        print("*** DEBUG: --> Trying explicit import: .file_manager", flush=True)
        from . import file_manager
        logger.info("Imported file_manager skill.")
        print("*** DEBUG: <-- Finished explicit import: .file_manager", flush=True)
    except ImportError as e:
        logger.warning(f"Could not import file_manager skill: {e}")
        print(f"*** WARNING: Could not import .file_manager: {e}", flush=True)
    except Exception as e:
        logger.error(f"Error importing file_manager skill: {e}", exc_info=True)
        print(f"*** ERROR: Could not import .file_manager: {e}", flush=True)

    try:
        print("*** DEBUG: --> Trying explicit import: .simulate", flush=True)
        from . import simulate # Assuming simulate is a package
        logger.info("Imported simulate package.")
        print("*** DEBUG: <-- Finished explicit import: .simulate", flush=True)
    except ImportError as e:
        logger.warning(f"Could not import simulate package: {e}")
        print(f"*** WARNING: Could not import .simulate: {e}", flush=True)
    except Exception as e:
        logger.error(f"Error importing simulate package: {e}", exc_info=True)
        print(f"*** ERROR: Could not import .simulate: {e}", flush=True)

    try:
        print("*** DEBUG: --> Trying explicit import: .monetization", flush=True)
        from . import monetization # Assuming monetization is a package
        logger.info("Imported monetization package.")
        print("*** DEBUG: <-- Finished explicit import: .monetization", flush=True)
    except ImportError as e:
        logger.warning(f"Could not import monetization package: {e}")
        print(f"*** WARNING: Could not import .monetization: {e}", flush=True)
    except Exception as e:
        logger.error(f"Error importing monetization package: {e}", exc_info=True)
        print(f"*** ERROR: Could not import .monetization: {e}", flush=True)

    try:
        print("*** DEBUG: --> Trying explicit import: .learning", flush=True)
        from . import learning # Assuming learning is a package
        logger.info("Imported learning package.")
        print("*** DEBUG: <-- Finished explicit import: .learning", flush=True)
    except ImportError as e:
        logger.warning(f"Could not import learning package: {e}")
        print(f"*** WARNING: Could not import .learning: {e}", flush=True)
    except Exception as e:
        logger.error(f"Error importing learning package: {e}", exc_info=True)
        print(f"*** ERROR: Could not import .learning: {e}", flush=True)

    print("*** DEBUG: Finished explicit imports in skills/__init__.py ***", flush=True)
    logger.info("Skills package initialized.")


# Initial load when the package is imported
print("*** DEBUG: Starting initial load_skills() call in skills/__init__.py ***", flush=True)
load_skills()
print("*** DEBUG: Finished initial load_skills() call in skills/__init__.py ***", flush=True)
