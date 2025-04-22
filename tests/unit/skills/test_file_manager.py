from unittest.mock import MagicMock, patch, mock_open, call
from pathlib import Path
from a3x.skills.file_system.file_manager import FileManagerSkill
from a3x.core.context import _ToolExecutionContext 