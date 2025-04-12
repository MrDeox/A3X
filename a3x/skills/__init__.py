# skills/__init__.py
import logging
import pkgutil
import importlib

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

# Restore automatic discovery
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    # Ignore modules starting with _, those being consolidated, or modules within the core directory
    if not module_name.startswith("_") and module_name not in SKIPPED_MODULES and not module_name.startswith("core."):
        logger.info(f"---> Attempting to import skill module: {module_name} (is_pkg={is_pkg})") # Log BEFORE
        try:
            _module = importlib.import_module(f"{__package__}.{module_name}")
            # Optionally, add module names to __all__ if needed
            # __all__.append(module_name)
            logger.info(f"---> Successfully imported skill module: {module_name}") # Log AFTER success
            # logger.debug(f"Successfully imported skill module: {module_name}") # Original debug log
        except Exception as e:
            logger.error(
                f"---> Failed to import skill module '{module_name}': {e}", exc_info=True # Log Exception
            )

# Optional: Log the final list of discovered modules if needed
logger.debug(f"Skills package initialized. Autodiscovered modules (via walk_packages).")

# Explicitly import the consolidated module to ensure its class is instantiated
# and its @skill methods are registered.
try:
    importlib.import_module(".file_manager", __package__)
    logger.debug("Explicitly imported consolidated file_manager skill module.")
except Exception as e:
    logger.error(
        f"Failed to explicitly import consolidated skill module 'file_manager': {e}",
        exc_info=True,
    )

# Explicitly import the simulate sub-package to ensure its skills are registered.
try:
    importlib.import_module(".simulate", __package__)
    logger.debug("Explicitly imported simulate skill sub-package.")
except Exception as e:
    logger.error(
        f"Failed to explicitly import skill sub-package 'simulate': {e}",
        exc_info=True,
    )

# Explicitly import the monetization sub-package to ensure its skills are registered.
try:
    importlib.import_module(".monetization", __package__)
    logger.debug("Explicitly imported monetization skill sub-package.")
except Exception as e:
    logger.error(
        f"Failed to explicitly import skill sub-package 'monetization': {e}",
        exc_info=True,
    )

# Remove the explicit import block for perception
# logger.info("Attempting explicit import of .perception...") # Log BEFORE trying
# try:
#     importlib.import_module(".perception", __package__)
#     logger.debug("Explicitly imported perception skill sub-package.") # Log AFTER success
# except Exception as e:
#     logger.error(
#         f"Failed to explicitly import skill sub-package 'perception': {e}",
#         exc_info=True,
#     )

# Explicitly import the auto_generated sub-package to ensure its skills are registered.
try:
    importlib.import_module(".auto_generated", __package__)
    logger.debug("Explicitly imported auto_generated skill sub-package.")
except ModuleNotFoundError:
    # Expected if the directory doesn't exist or has no __init__.py yet
    logger.debug("Sub-package 'auto_generated' not found or not yet initialized, skipping import.")
except Exception as e:
    logger.error(
        f"Failed to explicitly import skill sub-package 'auto_generated': {e}",
        exc_info=True,
    )
