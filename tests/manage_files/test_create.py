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

# == Create Action Tests ==

def test_create_success(mocker, mock_path_exists, mock_path_is_dir, mock_path_mkdir, mock_open_func):
    """Test successful file creation."""
    filepath = "output/new_file.log"
    content = "Log entry 1"
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = False # File does not exist initially
    mock_path_is_dir.return_value = False # Path is not a directory

    action_input = {"action": "create", "filepath": filepath, "content": content, "overwrite": False}
    result = skill_manage_files(action_input)

    assert result['status'] == "success"
    assert result['action'] == "file_created"
    assert result['data']['filepath'] == filepath
    assert "criado com sucesso" in result['data']['message']
    mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_open_func.assert_called_once_with(resolved_path, "w", encoding="utf-8")
    # Check if write was called on the file handle
    handle = mock_open_func() # Get the mock file handle
    handle.write.assert_called_once_with(content)

def test_create_overwrite_fail(mocker, mock_path_exists, mock_path_is_dir):
    """Test creating a file that exists without overwrite=True."""
    filepath = "existing_file.txt"
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = True # File exists
    mock_path_is_dir.return_value = False # It's a file, not a dir

    action_input = {"action": "create", "filepath": filepath, "content": "new data", "overwrite": False}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "create_file_failed"
    assert "já existe" in result['data']['message']
    assert "overwrite: True" in result['data']['message']

def test_create_overwrite_success(mocker, mock_path_exists, mock_path_is_dir, mock_path_mkdir, mock_open_func):
    """Test successfully overwriting an existing file."""
    filepath = "output/another_file.log"
    content = "Overwritten content"
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = True # File exists
    mock_path_is_dir.return_value = False # It's a file

    action_input = {"action": "create", "filepath": filepath, "content": content, "overwrite": True}
    result = skill_manage_files(action_input)

    assert result['status'] == "success"
    assert result['action'] == "file_overwritten"
    assert result['data']['filepath'] == filepath
    assert "sobrescrito com sucesso" in result['data']['message']
    mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_open_func.assert_called_once_with(resolved_path, "w", encoding="utf-8")
    mock_open_func().write.assert_called_once_with(content)

def test_create_target_is_directory(mocker, mock_path_exists, mock_path_is_dir):
    """Test attempting to create a file where a directory exists."""
    filepath = "data/existing_dir"
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = True # Path exists
    mock_path_is_dir.return_value = True # It is a directory

    action_input = {"action": "create", "filepath": filepath, "content": "test", "overwrite": False}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "create_file_failed"
    assert "já existe um diretório com este nome" in result['data']['message']

def test_create_permission_error(mocker, mock_path_exists, mock_path_is_dir, mock_path_mkdir, mock_open_func):
    """Test create file with permission error during write."""
    filepath = "restricted_area/new_file.txt"
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = False
    mock_path_is_dir.return_value = False
    mock_open_func.side_effect = PermissionError("Cannot write")

    action_input = {"action": "create", "filepath": filepath, "content": "test", "overwrite": False}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "create_file_failed"
    assert "Permissão negada" in result['data']['message']

def test_create_invalid_path(mocker):
    """Test create with a path outside the workspace."""
    filepath = "../outside_project.txt"
    action_input = {"action": "create", "filepath": filepath, "content": "test", "overwrite": False}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "create_file_failed"
    assert "Acesso negado ou caminho inválido" in result['data']['message']

def test_missing_parameters_for_create():
    """Test calling 'create' action with missing parameters."""
    action_input_no_path = {"action": "create", "content": "abc"}
    result_no_path = skill_manage_files(action_input_no_path)
    assert result_no_path['status'] == "error"
    assert result_no_path['action'] == "create_file_failed"
    assert "Parâmetros 'filepath' e 'content' são obrigatórios" in result_no_path['data']['message']

    action_input_no_content = {"action": "create", "filepath": "new.txt"}
    result_no_content = skill_manage_files(action_input_no_content)
    assert result_no_content['status'] == "error"
    assert result_no_content['action'] == "create_file_failed"
    assert "Parâmetros 'filepath' e 'content' são obrigatórios" in result_no_content['data']['message']

def test_invalid_overwrite_type_for_create():
    """Test calling 'create' with non-boolean overwrite."""
    action_input = {"action": "create", "filepath": "f.txt", "content": "c", "overwrite": "yes"}
    result = skill_manage_files(action_input)
    assert result['status'] == "error"
    assert result['action'] == "create_file_failed"
    assert "Parâmetro 'overwrite' deve ser um booleano" in result['data']['message']
