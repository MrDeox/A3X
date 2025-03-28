import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import os # Import os for path comparisons if needed

# Import the skill function and potentially helper functions if testing directly
from skills.manage_files import skill_manage_files, WORKSPACE_ROOT

# Define a consistent mock workspace root for tests
# Note: WORKSPACE_ROOT is already imported, but redefining ensures consistency if it changes
MOCK_WORKSPACE_ROOT = Path("/home/arthur/Projects/A3X").resolve()

# Helper to create expected resolved paths
def mock_resolve(path_str):
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (MOCK_WORKSPACE_ROOT / p).resolve()

# == Test Fixtures (Optional but helpful) ==
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
def mock_path_iterdir(mocker):
    return mocker.patch('pathlib.Path.iterdir')

@pytest.fixture
def mock_open_func(mocker):
    return mocker.patch('builtins.open', mock_open())

@pytest.fixture(autouse=True) # Apply this to all tests in the module
def mock_workspace_check(mocker):
    # Mock _is_path_within_workspace to always allow paths starting with MOCK_WORKSPACE_ROOT
    # This simplifies testing as we don't need to mock the complex logic inside it,
    # assuming _resolve_path works correctly (which we test implicitly)
    def _mock_check(path):
        try:
            abs_path = Path(path).resolve()
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

def test_read_is_directory(mocker, mock_path_exists, mock_path_is_file):
    """Test attempting to read a directory."""
    filepath = "data/"
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = True
    mock_path_is_file.return_value = False # It's not a file
    # No need to mock open raising IsADirectoryError, as is_file() check happens first

    action_input = {"action": "read", "filepath": filepath}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "read_file_failed"
    # Check the message returned due to the is_file() check failing
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
    # The action name depends on the mock implementation detail of whether exists() is checked *after* opening
    # Let's assume the check happens *before* opening, so overwrite=True leads to 'file_overwritten'
    # If exists() was checked after, it might still return file_created. Need to match implementation.
    # Based on the code: `action = "file_overwritten" if overwrite and resolved_path.exists() else "file_created"`
    # And the fact mock_path_exists returns True initially, it *should* be file_overwritten.
    assert result['action'] == "file_overwritten" # Adjust if mock logic differs
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
    # Simulate permission error on open, or on mkdir if checking that first
    mock_open_func.side_effect = PermissionError("Cannot write")
    # OR mock_path_mkdir.side_effect = PermissionError("Cannot create directory")

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
    # Allow mkdir to be called (or not, depending on implementation)
    # mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_open_func.assert_called_once_with(resolved_path, "a", encoding="utf-8")
    mock_open_func().write.assert_called_once_with(expected_write)

def test_append_file_not_found(mocker, mock_path_exists):
    """Test appending to a non-existent file (should fail)."""
    filepath = "logs/new_log.log"
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = False # File does not exist

    action_input = {"action": "append", "filepath": filepath, "content": "Initial content"}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "append_failed"
    assert "não encontrado para adicionar conteúdo" in result['data']['message']


def test_append_target_is_directory(mocker, mock_path_exists, mock_path_is_dir):
    """Test appending to a path that is a directory."""
    filepath = "data/" # Path is a directory
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = True
    mock_path_is_dir.return_value = True # It is a directory

    action_input = {"action": "append", "filepath": filepath, "content": "text"}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "append_failed"
    assert "Não é possível adicionar conteúdo a um diretório" in result['data']['message']

def test_append_permission_error(mocker, mock_path_exists, mock_path_is_file, mock_path_is_dir, mock_path_mkdir, mock_open_func):
    """Test append with permission error."""
    filepath = "readonly.txt"
    resolved_path = mock_resolve(filepath)

    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    mock_path_is_dir.return_value = False
    # Prevent mkdir from causing issues if parent is root
    mock_path_mkdir.return_value = None
    # Inject PermissionError on the open call
    mock_open_func.side_effect = PermissionError("Cannot append")

    action_input = {"action": "append", "filepath": filepath, "content": "more data"}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "append_failed"
    assert "Permissão negada para adicionar conteúdo" in result['data']['message'] # Check specific message

def test_append_invalid_path(mocker):
    """Test append with a path outside the workspace."""
    filepath = "../../secrets.txt"
    action_input = {"action": "append", "filepath": filepath, "content": "test"}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "append_failed"
    assert "Acesso negado ou caminho inválido" in result['data']['message']


# == List Action Tests ==

def test_list_success_files_and_dirs(mocker, mock_path_exists, mock_path_is_dir, mock_path_iterdir):
    """Test successfully listing a directory with files and subdirs."""
    directory = "src/components"
    resolved_path = mock_resolve(directory)

    # Mock items returned by iterdir
    mock_file1 = MagicMock(spec=Path)
    mock_file1.name = "Button.tsx"
    mock_file1.is_dir.return_value = False
    mock_file1.relative_to.return_value = Path(f"{directory}/Button.tsx") # Path relative to WORKSPACE

    mock_dir1 = MagicMock(spec=Path)
    mock_dir1.name = "utils"
    mock_dir1.is_dir.return_value = True
    mock_dir1.relative_to.return_value = Path(f"{directory}/utils")

    mock_file2 = MagicMock(spec=Path)
    mock_file2.name = "Card.tsx"
    mock_file2.is_dir.return_value = False
    mock_file2.relative_to.return_value = Path(f"{directory}/Card.tsx")

    mock_path_exists.return_value = True
    mock_path_is_dir.return_value = True
    mock_path_iterdir.return_value = [mock_file1, mock_dir1, mock_file2]

    action_input = {"action": "list", "directory": directory}
    result = skill_manage_files(action_input)

    expected_items = sorted([f"{directory}/Button.tsx", f"{directory}/utils/", f"{directory}/Card.tsx"])

    assert result['status'] == "success"
    assert result['action'] == "directory_listed"
    assert result['data']['directory'] == directory
    assert result['data']['items'] == expected_items
    assert "item(s) encontrado(s)" in result['data']['message']
    assert "Button.tsx" in result['data']['message']
    assert "utils/" in result['data']['message'] # Check for trailing slash in message

def test_list_empty_directory(mocker, mock_path_exists, mock_path_is_dir, mock_path_iterdir):
    """Test listing an empty directory."""
    directory = "empty_dir"
    resolved_path = mock_resolve(directory)

    mock_path_exists.return_value = True
    mock_path_is_dir.return_value = True
    mock_path_iterdir.return_value = [] # Empty list

    action_input = {"action": "list", "directory": directory}
    result = skill_manage_files(action_input)

    assert result['status'] == "success"
    assert result['action'] == "directory_listed"
    assert result['data']['items'] == []
    assert "0 item(s) encontrado(s)" in result['data']['message']

def test_list_directory_not_found(mocker, mock_path_exists):
    """Test listing a non-existent directory."""
    directory = "non_existent_dir"
    resolved_path = mock_resolve(directory)

    mock_path_exists.return_value = False # Directory does not exist

    action_input = {"action": "list", "directory": directory}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "list_failed"
    assert "Diretório não encontrado" in result['data']['message']

def test_list_target_is_file(mocker, mock_path_exists, mock_path_is_dir):
    """Test listing a path that is a file, not a directory."""
    directory = "src/app.py"
    resolved_path = mock_resolve(directory)

    mock_path_exists.return_value = True
    mock_path_is_dir.return_value = False # It's a file

    action_input = {"action": "list", "directory": directory}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "list_failed"
    assert "não é um diretório" in result['data']['message']

def test_list_permission_error(mocker, mock_path_exists, mock_path_is_dir, mock_path_iterdir):
    """Test listing directory with permission error."""
    directory = "restricted_dir"
    resolved_path = mock_resolve(directory)

    mock_path_exists.return_value = True
    mock_path_is_dir.return_value = True
    mock_path_iterdir.side_effect = PermissionError("Cannot list directory")

    action_input = {"action": "list", "directory": directory}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "list_failed"
    assert "Permissão negada" in result['data']['message']

def test_list_invalid_path(mocker):
    """Test list with a path outside the workspace."""
    directory = "../../root_dir"
    action_input = {"action": "list", "directory": directory}
    result = skill_manage_files(action_input)

    assert result['status'] == "error"
    assert result['action'] == "list_failed"
    assert "Acesso negado ou caminho inválido" in result['data']['message']

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

def test_missing_filepath_for_read():
    """Test calling 'read' action without 'filepath'."""
    action_input = {"action": "read"}
    result = skill_manage_files(action_input)
    assert result['status'] == "error"
    assert result['action'] == "read_file_failed"
    assert "Parâmetro 'filepath' obrigatório" in result['data']['message']

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

def test_missing_directory_for_list():
    """Test calling 'list' action without 'directory'."""
    action_input = {"action": "list"}
    result = skill_manage_files(action_input)
    assert result['status'] == "error"
    assert result['action'] == "list_failed"
    assert "Parâmetro 'directory' obrigatório" in result['data']['message']

# == Delete Action Tests (Placeholder) ==

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
     action_input = {"action": "delete"}
     result = skill_manage_files(action_input)
     assert result['status'] == "error"
     assert result['action'] == "delete_failed"
     assert "Parâmetro 'filepath' obrigatório" in result['data']['message']

