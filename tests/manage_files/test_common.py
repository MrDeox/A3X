# Add project root to sys.path if needed (handled in conftest.py)

# <<< MODIFIED: Import specific functions and WORKSPACE_ROOT >>>
import pytest
from unittest.mock import patch

# <<< REMOVED OBSOLETE TESTS >>>
# def test_invalid_action():
#     ...
# def test_missing_action():
#     ...
# def test_delete_action_disabled():
#     ...
# def test_missing_filepath_for_delete():
#     ...

# Common fixtures for manage_files tests


@pytest.fixture
def mock_workspace(tmp_path):
    """Creates a temporary workspace directory."""
    ws_path = tmp_path / "mock_workspace"
    ws_path.mkdir()
    return ws_path


@pytest.fixture
def patch_workspace_root(mock_workspace):
    """Patches the WORKSPACE_ROOT constant as used in skills.file_manager."""
    # Corrected target: Patch the constant where it's used in the module under test
    with patch("skills.file_manager.WORKSPACE_ROOT", str(mock_workspace)):
        yield str(mock_workspace)


# Example common test (can be expanded)
def test_workspace_root_patching(patch_workspace_root):
    """Verify that the WORKSPACE_ROOT is correctly patched within skills.file_manager."""
    # Re-import within the test from the module where it's patched and used
    from skills.file_manager import WORKSPACE_ROOT as patched_root

    assert patched_root == patch_workspace_root


# Add other common tests or helper functions if applicable
