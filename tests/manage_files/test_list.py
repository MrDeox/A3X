import pytest
from unittest.mock import MagicMock
from pathlib import Path

# Import the skill function and WORKSPACE_ROOT
from skills.list_files import list_files

# Define a consistent mock workspace root for tests
MOCK_WORKSPACE_ROOT = Path("/home/arthur/Projects/A3X").resolve()


# Helper to create expected resolved paths
def mock_resolve(path_str):
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (MOCK_WORKSPACE_ROOT / p).resolve()


# Fixtures from original file (simplified for clarity, could use conftest.py)
@pytest.fixture
def mock_path_exists(mocker):
    return mocker.patch("pathlib.Path.exists")


@pytest.fixture
def mock_path_is_file(mocker):
    return mocker.patch("pathlib.Path.is_file")


@pytest.fixture
def mock_path_is_dir(mocker):
    return mocker.patch("pathlib.Path.is_dir")


@pytest.fixture
def mock_path_iterdir(mocker):
    return mocker.patch("pathlib.Path.iterdir")


# == List Action Tests ==


def test_list_success_files_and_dirs(
    mocker, mock_path_exists, mock_path_is_dir, mock_path_iterdir
):
    """Test successfully listing a directory with files and subdirs."""
    directory = "src/components"
    # resolved_path = mock_resolve(directory) # F841

    # Mock items returned by iterdir
    mock_file1 = MagicMock(spec=Path)
    mock_file1.name = "Button.tsx"
    mock_file1.is_dir.return_value = False
    # Simulate relative_to returning the path relative to WORKSPACE_ROOT
    mock_file1.relative_to.return_value = Path("src/components/Button.tsx")

    mock_dir1 = MagicMock(spec=Path)
    mock_dir1.name = "utils"
    mock_dir1.is_dir.return_value = True
    mock_dir1.relative_to.return_value = Path("src/components/utils")

    mock_file2 = MagicMock(spec=Path)
    mock_file2.name = "Card.tsx"
    mock_file2.is_dir.return_value = False
    mock_file2.relative_to.return_value = Path("src/components/Card.tsx")

    mock_path_exists.return_value = True
    mock_path_is_dir.return_value = True
    mock_path_iterdir.return_value = [mock_file1, mock_dir1, mock_file2]

    action_input = {"directory": directory}
    result = list_files(**action_input)

    # Expected items are relative to WORKSPACE_ROOT, dirs have trailing slash
    expected_items = sorted(
        [
            "src/components/Button.tsx",
            "src/components/utils/",
            "src/components/Card.tsx",
        ]
    )

    assert result["status"] == "success"
    assert result["action"] == "directory_listed"
    assert result["data"]["directory_requested"] == directory
    assert sorted(result["data"]["items"]) == expected_items
    assert "non-hidden item(s) found" in result["data"]["message"]
    assert "3" in result["data"]["message"]


def test_list_empty_directory(
    mocker, mock_path_exists, mock_path_is_dir, mock_path_iterdir
):
    """Test listing an empty directory."""
    directory = "empty_dir"
    # resolved_path = mock_resolve(directory) # F841

    mock_path_exists.return_value = True
    mock_path_is_dir.return_value = True
    mock_path_iterdir.return_value = []  # Empty list

    action_input = {"directory": directory}
    result = list_files(**action_input)

    assert result["status"] == "success"
    assert result["action"] == "directory_listed"
    assert result["data"]["items"] == []
    assert "non-hidden item(s) found" in result["data"]["message"]
    assert "0" in result["data"]["message"]


def test_list_directory_not_found(mocker, mock_path_exists):
    """Test listing a non-existent directory."""
    directory = "non_existent_dir"
    # resolved_path = mock_resolve(directory) # F841

    mock_path_exists.return_value = False  # Directory does not exist

    action_input = {"directory": directory}
    result = list_files(**action_input)

    assert result["status"] == "error"
    assert result["action"] == "list_files_failed"
    assert f"Directory not found: '{directory}'" == result["data"]["message"]


def test_list_target_is_file(mocker, mock_path_exists, mock_path_is_dir):
    """Test listing a path that is a file, not a directory."""
    directory = "src/app.py"
    # resolved_path = mock_resolve(directory) # F841

    mock_path_exists.return_value = True
    mock_path_is_dir.return_value = False  # It's a file

    action_input = {"directory": directory}
    result = list_files(**action_input)

    assert result["status"] == "error"
    assert result["action"] == "list_files_failed"
    assert (
        f"The specified path is not a directory: '{directory}'"
        == result["data"]["message"]
    )


def test_list_permission_error(
    mocker, mock_path_exists, mock_path_is_dir, mock_path_iterdir
):
    """Test listing directory with permission error."""
    directory = "restricted_dir"
    # resolved_path = mock_resolve(directory) # F841

    mock_path_exists.return_value = True
    mock_path_is_dir.return_value = True
    mock_path_iterdir.side_effect = PermissionError("Cannot list directory")

    action_input = {"directory": directory}
    result = list_files(**action_input)

    assert result["status"] == "error"
    assert result["action"] == "list_files_failed"
    assert "Permission denied" in result["data"]["message"]


def test_list_invalid_path(mocker):
    """Test list with a path outside the workspace."""
    directory = "../../root_dir"
    action_input = {"directory": directory}
    result = list_files(**action_input)

    assert result["status"] == "error"
    assert result["action"] == "list_files_failed"
    assert "Path validation failed" in result["data"]["message"]
    assert "Path inválido" in result["data"]["message"]


def test_missing_directory_for_list():
    """Test calling 'list' action without 'directory'. Uses default."""
    action_input = {}
    result = list_files(**action_input)
    # This should actually succeed if default ('.') is used
    # Re-evaluate this test case based on skill_list_files logic
    # For now, assuming it should succeed with default path
    assert result["status"] == "success"
    assert result["action"] == "directory_listed"
