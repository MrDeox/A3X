# a3x/skills/core/__init__.py

# Import core skill modules to trigger registration via @skill decorator

# Example (Add imports for your core skills here):
# from . import some_core_skill
from . import call_skill_by_name
from . import append_to_file_path
from . import auto_expand_capabilities
from . import propose_skill_from_gap
from . import reload_auto_generated_skills
import logging

logger = logging.getLogger(__name__)

print("DEBUG: Imported core skills package.")

# Optionally, you can define __all__ if needed for specific export control
# __all__ = ["call_skill_by_name", "append_to_file_path", ...] 