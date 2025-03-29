import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

# Import the skill function and WORKSPACE_ROOT
from skills.manage_files import skill_manage_files, WORKSPACE_ROOT

# Define a consistent mock workspace root for tests
MOCK_WORKSPACE_ROOT = Path("/home/arthur/Projects/A3X").resolve()

# Helper to create expected resolved paths
def mock_resolve(path_str):
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (MOCK_WORKSPACE_ROOT / p).resolve()

# == General/Invalid Action Tests ==

def test_invalid_action():
    """Test calling the skill with an unsupported action."""
    action_input = {"action": "fly", "destination": "moon"}
    result = skill_manage_files(action_input)
    assert result['status'] == "error"
    assert result['action'] == "manage_files_failed"
    assert "Ação 'fly' não é suportada" in result['data']['message']

def test_missing_action():
    """Test calling the skill without the 'action' parameter."""
    action_input = {"filepath": "some/file.txt"}
    result = skill_manage_files(action_input)
    assert result['status'] == "error"
    assert result['action'] == "manage_files_failed"
    assert "Parâmetro 'action' ausente" in result['data']['message']

# == Delete Action Tests (Placeholder/Error Cases) ==

def test_delete_action_disabled():
    """Test that the 'delete' action is currently disabled/not implemented."""
    filepath = "file_to_delete.tmp"
    action_input = {"action": "delete", "filepath": filepath}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "action_not_implemented"
    assert "ainda não está implementada/habilitada" in result['data']['message']
    assert filepath in result['data']['message']

def test_missing_filepath_for_delete():
     """Test calling 'delete' without 'filepath'."""
     # This test assumes 'delete' action exists but filepath is missing.
     # If 'delete' is completely removed, this test might need adjustment.
     action_input = {"action": "delete"}
     result = skill_manage_files(action_input)
     assert result['status'] == "error"
     # The action result depends on whether the 'delete' case exists in the main function.
     # If it doesn't exist, it might fall through to 'invalid_action'.
     # If it exists but checks for filepath first, it should be 'delete_failed'.
     # Assuming the latter for now:
     assert result['action'] == "delete_failed"
     assert "Parâmetro 'filepath' obrigatório" in result['data']['message']
