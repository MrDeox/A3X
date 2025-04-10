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
}

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    # Ignore modules starting with _ or those being consolidated
    if not module_name.startswith("_") and module_name not in SKIPPED_MODULES:
        try:
            _module = importlib.import_module(f".{module_name}", __package__)
            # Optionally, add module names to __all__ if needed
            # __all__.append(module_name)
            # logger.debug(f"Successfully imported skill module: {module_name}")
        except Exception as e:
            logger.error(
                f"Failed to import skill module '{module_name}': {e}", exc_info=True
            )

# Optional: Log the final list of discovered modules if needed
# logger.debug(f"Skills package initialized. Autodiscovered modules (approx): {__all__}")

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
