# skills/__init__.py

# Import individual skill modules here so the @skill decorator registers them
# when this package is loaded by core.tools.load_skills()

# from . import core_skills # Example: Load core skills like list_files, etc. <-- REMOVED
from . import gumroad_skill # Keep existing Gumroad skill import
from . import browser_skill # <-- ADDED: Import the new browser skill

# You can add more skill modules below as needed
# from . import another_skill
