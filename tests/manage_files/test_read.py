import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

# Import the skill function and WORKSPACE_ROOT
from skills.read_file import skill_read_file

# Define a consistent mock workspace root for tests
MOCK_WORKSPACE_ROOT = Path("/home/arthur/Projects/A3X").resolve()

# Helper to create expected resolved paths
def mock_resolve(path_str):
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (MOCK_WORKSPACE_ROOT / p).resolve()

# Fixtures for mocking Path methods used by the validator and the skill
@pytest.fixture
def mock_path_exists(mocker):
    return mocker.patch('pathlib.Path.exists', return_value=True) # Default to exists

@pytest.fixture
def mock_path_is_file(mocker):
    return mocker.patch('pathlib.Path.is_file', return_value=True) # Default to is file

@pytest.fixture
def mock_path_is_dir(mocker):
    return mocker.patch('pathlib.Path.is_dir', return_value=False) # Default to not dir

@pytest.fixture
def mock_path_getsize(mocker):
    # Mock stat().st_size used internally by skill_read_file
    mock_stat_result = MagicMock()
    mock_stat_result.st_size = 100 # Default size
    mock_stat_result.st_mode = 16877 # Simulate file mode (e.g., S_IFREG | 0644)
    return mocker.patch('pathlib.Path.stat', return_value=mock_stat_result)

@pytest.fixture
def mock_open_func(mocker):
    # Use the real mock_open to simulate file reading
    m = mock_open()
    return mocker.patch('builtins.open', m)

# == Read Action Tests (Updated for validator) ==

def test_read_success(mocker, mock_path_exists, mock_path_is_file, mock_path_getsize, mock_open_func):
    """Test successful file read after validator."""
    filepath = "data/my_file.txt"
    expected_content = "Hello, world!"
    resolved_path_obj = mock_resolve(filepath) # Validator uses this

    # Ensure mocks are set for the validator AND the skill logic
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    mock_path_getsize.return_value.st_size = len(expected_content)

    # Configure the mock_open context manager
    mock_file_handle = mock_open_func.return_value.__enter__.return_value
    mock_file_handle.read.return_value = expected_content

    action_input = {"file_name": filepath} # Validator expects 'file_name'
    result = skill_read_file(action_input)

    # Assertions remain mostly the same, but check action/message from skill
    assert result['status'] == "success"
    assert result['action'] == "file_read" # Check skill's success action name
    assert result['data']['filepath'] == filepath # Skill should report original path
    assert result['data']['content'] == expected_content
    assert "lido com sucesso" in result['data']['message'] # Check skill's message
    mock_open_func.assert_called_once_with(resolved_path_obj, "r", encoding="utf-8")
    assert mock_path_getsize.call_count == 2

def test_read_not_found(mocker, mock_path_exists, mock_path_is_file):
    """Test reading a non-existent file (handled by validator)."""
    filepath = "non_existent.txt"
    resolved_path = mock_resolve(filepath)

    # Simulate file not existing
    mock_path_exists.return_value = False
    mock_path_is_file.return_value = False # Doesn't matter if exists is False

    action_input = {"file_name": filepath}
    result = skill_read_file(action_input)

    assert result['status'] == "error"
    assert result['action'] == "path_validation_failed" # Check validator's action name
    assert f"Path not found: '{filepath}'" in result['data']['message'] # Check validator's message

def test_read_is_directory(mocker, mock_path_exists, mock_path_is_file, mock_path_is_dir):
    """Test attempting to read a directory (handled by validator)."""
    filepath = "data/"
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = True
    mock_path_is_file.return_value = False # It's not a file
    mock_path_is_dir.return_value = True

    action_input = {"file_name": filepath}
    result = skill_read_file(action_input)

    assert result['status'] == "error"
    assert result['action'] == "path_validation_failed" # Check validator's action name
    assert f"Path is not a file: '{filepath}'" in result['data']['message'] # Check validator's message

def test_read_permission_error(mocker, mock_path_exists, mock_path_is_file, mock_path_getsize, mock_open_func):
    """Test reading a file with permission error (handled by skill)."""
    filepath = "restricted.txt"
    resolved_path_obj = mock_resolve(filepath)

    # Validator checks pass
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    mock_path_getsize.return_value.st_size = 50 # Assume size check passes

    # Mock open to raise PermissionError
    mock_open_func.side_effect = PermissionError("Permission denied by OS")

    action_input = {"file_name": filepath}
    result = skill_read_file(action_input)

    assert result['status'] == "error"
    assert result['action'] == "read_file_failed" # Check skill's action name
    assert "Permissão negada para ler o arquivo:" in result['data']['message']
    mock_open_func.assert_called_once_with(resolved_path_obj, "r", encoding="utf-8")

def test_read_outside_workspace(mocker):
    """Test reading a file outside workspace (handled by validator)."""
    filepath = "../../../../etc/passwd"

    action_input = {"file_name": filepath}
    result = skill_read_file(action_input)

    assert result['status'] == "error"
    assert result['action'] == "path_validation_failed" # Check validator's action name
    assert "resolves outside the designated workspace" in result['data']['message'] # Check validator's message

def test_missing_filepath_for_read():
    """Test calling read without required filepath (handled by validator)."""
    action_input = {}
    result = skill_read_file(action_input)
    assert result['status'] == "error"
    assert result['action'] == "path_validation_failed" # Check validator's action name
    assert "Required path parameter 'file_name' is missing" in result['data']['message'] # Check validator's message
