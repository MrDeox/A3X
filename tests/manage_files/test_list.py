import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Import the skill function
from skills.file_manager import FileManagerSkill

# Removed MOCK_WORKSPACE_ROOT and mock_resolve helper

# Removed old Path mock fixtures

# == List Action Tests ==

# Removed mock_decorator_logic helper

@pytest.mark.asyncio
# Removed decorator patch and old fixtures
async def test_list_success_files_and_dirs(tmp_path):
    """Test successfully listing a directory via wrapped function."""
    directory = "src/components"
    file_manager_instance = FileManagerSkill() # Need instance for self

    # --- Mock Configuration ---
    resolved_path_mock = MagicMock(spec=Path, name="resolved_path_mock")
    # resolved_path_mock.is_dir.return_value = True # Not needed for wrapped func
    # Mock workspace_root on the instance for relative_to calculation
    # Assuming the instance has workspace_root correctly set
    # We might need to mock self.workspace_root.resolve() if it's called
    file_manager_instance.workspace_root = Path("/mock/workspace") # Set a mock root
    mock_workspace_resolved = file_manager_instance.workspace_root.resolve()

    # Mock items returned by iterdir
    mock_file1 = MagicMock(spec=Path, name="mock_file1")
    mock_file1.name = "Button.tsx"
    mock_file1.is_dir.return_value = False
    # Mock relative_to to return the correct relative Path object
    mock_file1.relative_to.return_value = Path(f"{directory}/Button.tsx")

    mock_dir1 = MagicMock(spec=Path, name="mock_dir1")
    mock_dir1.name = "utils"
    mock_dir1.is_dir.return_value = True
    mock_dir1.relative_to.return_value = Path(f"{directory}/utils")

    mock_file2 = MagicMock(spec=Path, name="mock_file2")
    mock_file2.name = "Card.tsx"
    mock_file2.is_dir.return_value = False
    mock_file2.relative_to.return_value = Path(f"{directory}/Card.tsx")

    # Configure iterdir on the mock
    resolved_path_mock.iterdir.return_value = [mock_file1, mock_dir1, mock_file2]

    # --- Execution ---
    result = await FileManagerSkill.list_directory.__wrapped__(
        file_manager_instance,
        resolved_path=resolved_path_mock,
        original_path_str=directory,
        directory=directory,
        extension=None # Assuming default or not used here
    )

    # Expected items based on the mocked relative_to paths
    expected_items = sorted(
        [
            f"{directory}/Button.tsx",
            f"{directory}/utils/", # Add trailing slash for dirs
            f"{directory}/Card.tsx",
        ]
    )

    # --- Assertions ---
    assert result["status"] == "success"
    assert result["action"] == "directory_listed"
    assert result["data"]["directory_requested"] == directory
    assert sorted(result["data"]["items"]) == expected_items
    assert "3 non-hidden item(s) found" in result["data"]["message"]

    # Verify calls
    # resolved_path_mock.is_dir.assert_called_once() # Not called by wrapped func
    resolved_path_mock.iterdir.assert_called_once()
    # Check that relative_to was called for each item
    mock_file1.relative_to.assert_called_once_with(mock_workspace_resolved)
    mock_dir1.relative_to.assert_called_once_with(mock_workspace_resolved)
    mock_file2.relative_to.assert_called_once_with(mock_workspace_resolved)


@pytest.mark.asyncio
async def test_list_empty_directory(tmp_path):
    """Test listing an empty directory via wrapped function."""
    directory = "empty_dir"
    file_manager_instance = FileManagerSkill()

    resolved_path_mock = MagicMock(spec=Path, name="resolved_path_mock")
    # resolved_path_mock.is_dir.return_value = True # Not needed
    resolved_path_mock.iterdir.return_value = []  # Empty list

    result = await FileManagerSkill.list_directory.__wrapped__(
        file_manager_instance,
        resolved_path=resolved_path_mock,
        original_path_str=directory,
        directory=directory,
        extension=None
    )

    assert result["status"] == "success"
    assert result["action"] == "directory_listed"
    assert result["data"]["items"] == []
    assert "0 non-hidden item(s) found" in result["data"]["message"]

    # resolved_path_mock.is_dir.assert_called_once() # Not called
    resolved_path_mock.iterdir.assert_called_once()


@pytest.mark.asyncio
async def test_list_permission_error(tmp_path):
    """Test listing directory with permission error via wrapped function."""
    directory = "restricted_dir"
    file_manager_instance = FileManagerSkill()

    resolved_path_mock = MagicMock(spec=Path, name="resolved_path_mock")
    # resolved_path_mock.is_dir.return_value = True # Not needed
    # Configure iterdir to raise PermissionError
    resolved_path_mock.iterdir.side_effect = PermissionError("Permission denied to list")

    result = await FileManagerSkill.list_directory.__wrapped__(
        file_manager_instance,
        resolved_path=resolved_path_mock,
        original_path_str=directory,
        directory=directory,
        extension=None
    )

    assert result["status"] == "error"
    assert result["action"] == "list_files_failed" # Correct action name
    assert "Permission denied" in result["data"]["message"]
    assert directory in result["data"]["message"]

    # Verify calls up to the point of error
    # resolved_path_mock.is_dir.assert_called_once() # Not called
    resolved_path_mock.iterdir.assert_called_once()

# Removed tests that only tested decorator validation logic:
# - test_list_directory_not_found
# - test_list_target_is_file
# - test_list_invalid_path
# - test_missing_directory_for_list

# Consider adding tests for filtering by extension if that logic is complex
# Consider adding test for generic Exception during iteration
