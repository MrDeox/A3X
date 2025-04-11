# a3x/skills/web/__init__.py
import logging

# Import skill modules in this package to ensure registration by the @skill decorator
# from . import interact_with_browser # No longer needed directly if consolidated
from . import autonomous_web_navigator

# Add other web skills here in the future, e.g.:
# from . import find_element

logger = logging.getLogger(__name__)
logger.debug("Web skills package initialized, relying on decorators for registration.") 