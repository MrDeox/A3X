import logging
from typing import Dict, Any, Optional

# <<< Import base and decorator >>>
from .base import ManagerFragment # Inherits from BaseFragment
from .registry import fragment
from a3x.core.tool_executor import _ToolExecutionContext, execute_tool
from a3x.skills.file_manager import FileManagerSkill # Import the skill class

logger = logging.getLogger(__name__)

# --- Define Skills Managed by this Manager --- 
# This can be defined here or potentially derived if skills register their manager
FILE_OPS_SKILLS = [
    "read_file",
    "write_file",
    "list_directory",
    "append_to_file",
    "delete_path",
]

# <<< Apply the decorator >>>
@fragment(
    name="FileOpsManager",
    description="Coordinates file operations by selecting and executing the appropriate file skill.",
    category="Management",
    managed_skills=FILE_OPS_SKILLS
)
class FileOpsManager(ManagerFragment): # Inherit from ManagerFragment
    """Manager Fragment responsible for handling file operations."""

    # MANAGED_SKILLS = FILE_OPS_SKILLS # Set as class attribute for potential discovery fallback

    async def execute(
        self,
        ctx: _ToolExecutionContext,
        # ... existing code ...
    ) -> Dict[str, Any]:
        # ... existing code ...
        return {} 