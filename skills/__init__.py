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
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    if not module_name.startswith('_'): # Ignore modules starting with _
        try:
            _module = importlib.import_module(f'.{module_name}', __package__)
            # Optionally, add module names to __all__ if needed
            # __all__.append(module_name)
            # logger.debug(f"Successfully imported skill module: {module_name}")
        except Exception as e:
            logger.error(f"Failed to import skill module '{module_name}': {e}", exc_info=True)

# Optional: Log the final list of discovered modules if needed
# logger.debug(f"Skills package initialized. Autodiscovered modules (approx): {__all__}")
