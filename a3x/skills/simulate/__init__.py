# This file makes the 'simulate' directory a Python package.

import logging

logger = logging.getLogger(__name__)

# Explicitly import modules within this sub-package to ensure their skills are registered.
try:
    from . import simulate_arthur_response
except ImportError as e:
    logger.warning(f"Could not import simulate_arthur_response: {e}")
try:
    from . import simulate_decision_reflection
except ImportError as e:
    logger.warning(f"Could not import simulate_decision_reflection: {e}")

logger.debug("Imported simulate skills package.") 