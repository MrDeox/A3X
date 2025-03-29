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

# Fixtures from original file (simplified for clarity, could use conftest.py)
@pytest.fixture
def mock_path_exists(mocker):
    return mocker.patch('pathlib.Path.exists')

@pytest.fixture
def mock_path_is_file(mocker):
    return mocker.patch('pathlib.Path.is_file')

@pytest.fixture
def mock_path_is_dir(mocker):
    return mocker.patch('pathlib.Path.is_dir')

@pytest.fixture
def mock_open_func(mocker):
    return mocker.patch('builtins.open', mock_open())

@pytest.fixture(autouse=True)
def mock_workspace_check(mocker):
    # Mock _is_path_within_workspace to simplify testing
    def _mock_check(path):
        try:
            abs_path = Path(path).resolve()
            # Use MOCK_WORKSPACE_ROOT for consistent test environment
            return str(abs_path).startswith(str(MOCK_WORKSPACE_ROOT))
        except Exception:
            return False
    mocker.patch('skills.manage_files._is_path_within_workspace', side_effect=_mock_check)


# == Read Action Tests ==

def test_read_success(mocker, mock_path_exists, mock_path_is_file, mock_open_func):
    """Test successful file read."""
    filepath = "data/my_file.txt"
    expected_content = "Hello, world!"
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    mock_path_is_dir.return_value = False # Ensure it's not seen as a dir
    # Configure mock_open fixture to return specific content
    mock_open_func.return_value.read.return_value = expected_content

    action_input = {"action": "read", "filepath": filepath}
    result = skill_manage_files(action_input)

    assert result['status'] == "success"
    assert result['action'] == "file_read"
    assert result['data']['filepath'] == filepath
    assert result['data']['content'] == expected_content
    assert "lido com sucesso" in result['data']['message']
    # Check if open was called with the correct resolved path
    mock_open_func.assert_called_once_with(resolved_path, "r", encoding="utf-8")

def test_read_not_found(mocker, mock_path_exists, mock_path_is_file):
    """Test reading a non-existent file."""
    filepath = "non_existent.txt"
    resolved_path = mock_resolve(filepath)

    # Simulate file not existing or not being a file
    mock_path_exists.return_value = False
    mock_path_is_file.return_value = False

    action_input = {"action": "read", "filepath": filepath}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "read_file_failed"
    assert "não encontrado ou não é um arquivo" in result['data']['message'] # Updated message check

def test_read_is_directory(mocker, mock_path_exists, mock_path_is_file, mock_path_is_dir):
    """Test attempting to read a directory."""
    filepath = "data/"
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = True
    mock_path_is_file.return_value = False # It's not a file
    mock_path_is_dir.return_value = True # Crucially, it IS a directory

    action_input = {"action": "read", "filepath": filepath}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "read_file_failed"
    # The check in the code is `if not resolved_path.exists() or not resolved_path.is_file():`
    # So, even if it exists, is_file() being False triggers this error.
    assert "não encontrado ou não é um arquivo" in result['data']['message']


def test_read_permission_error(mocker, mock_path_exists, mock_path_is_file, mock_open_func):
    """Test reading a file with permission error."""
    filepath = "restricted.txt"
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    mock_open_func.side_effect = PermissionError("Permission denied")

    action_input = {"action": "read", "filepath": filepath}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "read_file_failed"
    assert "Permissão negada" in result['data']['message']

def test_read_outside_workspace(mocker):
    """Test reading a file outside the allowed workspace."""
    # We rely on the mocked _is_path_within_workspace fixture
    filepath = "../../../../etc/passwd" # Example of path traversal attempt

    action_input = {"action": "read", "filepath": filepath}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "read_file_failed"
    assert "Acesso negado ou caminho inválido" in result['data']['message']

def test_missing_filepath_for_read():
    """Test calling 'read' action without 'filepath'."""
    action_input = {"action": "read"}
    result = skill_manage_files(action_input)
    assert result['status'] == "error"
    assert result['action'] == "read_file_failed"
    assert "Parâmetro 'filepath' obrigatório" in result['data']['message']
