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
def mock_path_mkdir(mocker):
    return mocker.patch('pathlib.Path.mkdir')

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

# == Append Action Tests ==

def test_append_success(mocker, mock_path_exists, mock_path_is_file, mock_path_is_dir, mock_path_mkdir, mock_open_func):
    """Test successful append to an existing file."""
    filepath = "logs/activity.log"
    content_to_append = "User logged in"
    expected_write = content_to_append + "\n" # Skill adds newline
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = True  # File exists
    mock_path_is_file.return_value = True # It is a file
    mock_path_is_dir.return_value = False # It is not a directory

    action_input = {"action": "append", "filepath": filepath, "content": content_to_append}
    result = skill_manage_files(action_input)

    assert result['status'] == "success"
    assert result['action'] == "file_appended"
    assert result['data']['filepath'] == filepath
    assert "Conteúdo adicionado" in result['data']['message']
    mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True) # mkdir is called for append too
    mock_open_func.assert_called_once_with(resolved_path, "a", encoding="utf-8")
    mock_open_func().write.assert_called_once_with(expected_write)

def test_append_file_not_found(mocker, mock_path_exists, mock_path_is_file, mock_path_is_dir):
    """Test appending to a non-existent file (should fail)."""
    filepath = "logs/new_log.log"
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = False # File does not exist
    mock_path_is_file.return_value = False # Also not a file
    mock_path_is_dir.return_value = False # And not a dir

    action_input = {"action": "append", "filepath": filepath, "content": "Initial content"}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "append_failed"
    # Corrected assertion: Check for the actual 'not found' message
    assert "não encontrado para adicionar conteúdo" in result['data']['message']

def test_append_target_is_directory(mocker, mock_path_exists, mock_path_is_file, mock_path_is_dir):
    """Test appending to a path that is a directory."""
    filepath = "data/" # Path is a directory
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = True
    mock_path_is_file.return_value = False # NOT a file
    mock_path_is_dir.return_value = True # It IS a directory

    action_input = {"action": "append", "filepath": filepath, "content": "text"}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "append_failed"
    # Corrected assertion: Check for the specific 'cannot append to directory' message
    assert "Não é possível adicionar conteúdo a um diretório" in result['data']['message']

def test_append_permission_error(mocker, mock_path_exists, mock_path_is_file, mock_path_is_dir, mock_path_mkdir, mock_open_func):
    """Test append with permission error."""
    filepath = "readonly.txt"
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    mock_path_is_dir.return_value = False
    mock_path_mkdir.return_value = None # Prevent mkdir from causing issues
    mock_open_func.side_effect = PermissionError("Cannot append")

    action_input = {"action": "append", "filepath": filepath, "content": "more data"}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "append_failed"
    assert "Permissão negada" in result['data']['message'] # General permission error check

def test_append_invalid_path(mocker):
    """Test append with a path outside the workspace."""
    filepath = "../../secrets.txt"
    action_input = {"action": "append", "filepath": filepath, "content": "test"}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "append_failed"
    assert "Acesso negado ou caminho inválido" in result['data']['message']

def test_missing_parameters_for_append():
    """Test calling 'append' action with missing parameters."""
    action_input_no_path = {"action": "append", "content": "abc"}
    result_no_path = skill_manage_files(action_input_no_path)
    assert result_no_path['status'] == "error"
    assert result_no_path['action'] == "append_failed"
    assert "Parâmetros 'filepath' e 'content' são obrigatórios" in result_no_path['data']['message']

    action_input_no_content = {"action": "append", "filepath": "new.txt"}
    result_no_content = skill_manage_files(action_input_no_content)
    assert result_no_content['status'] == "error"
    assert result_no_content['action'] == "append_failed"
    assert "Parâmetros 'filepath' e 'content' são obrigatórios" in result_no_content['data']['message']
