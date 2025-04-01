import pytest
import os
import tempfile
from unittest.mock import AsyncMock, patch, mock_open, MagicMock
from pathlib import Path

# Import the skill function
from skills.read_file import read_file
# Import the actual WORKSPACE_ROOT
from core.config import PROJECT_ROOT as WORKSPACE_ROOT # Use the real one

# Define a consistent mock workspace root for tests
# MOCK_WORKSPACE_ROOT = Path("/home/arthur/Projects/A3X").resolve()

# Helper to create expected resolved paths
# def mock_resolve(path_str):
#     p = Path(path_str)
#     if p.is_absolute():
#         return p.resolve()
#     return (MOCK_WORKSPACE_ROOT / p).resolve()

# Fixtures for mocking Path methods used by the validator and the skill
@pytest.fixture
def mock_path_exists(mocker):
    return mocker.patch.object(Path, 'exists')

@pytest.fixture
def mock_path_is_file(mocker):
    return mocker.patch.object(Path, 'is_file')

@pytest.fixture
def mock_path_is_dir(mocker):
    return mocker.patch.object(Path, 'is_dir')

@pytest.fixture
def mock_path_resolve(mocker):
    # Mock resolve to return a predictable path object
    def _mock_resolve(self):
        mock_resolved = MagicMock(spec=Path)
        mock_resolved.name = self.name
        mock_resolved.parent = self.parent
        mock_resolved.__str__.return_value = str(Path(WORKSPACE_ROOT) / self)
        try:
            mock_resolved.suffix = Path(str(self)).suffix
        except Exception:
            mock_resolved.suffix = ".mockdefault"
        # Ensure the resolved mock object itself has a working stat().st_size
        mock_resolved.stat.return_value = MagicMock(st_size=100) # Default size
        return mock_resolved
    return mocker.patch.object(Path, 'resolve', _mock_resolve)

@pytest.fixture
def mock_open_func(mocker):
    m = mock_open()
    return mocker.patch('builtins.open', m)

# == Read Action Tests ==

def test_read_success(mocker, mock_path_exists, mock_path_is_file, mock_path_resolve, mock_open_func):
    filepath = "data/my_file.txt"
    expected_content = "Hello, world!"
    resolved_path_obj = Path(WORKSPACE_ROOT) / filepath

    # Configure mocks
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    # Configure the mock returned by resolve fixture to have correct size
    mock_resolved = mock_path_resolve.return_value # Get the mock obj resolve returns
    mock_resolved.stat.return_value.st_size = len(expected_content)

    # Configure mock_open
    mock_file_handle = mock_open_func.return_value.__enter__.return_value
    mock_file_handle.read.return_value = expected_content

    action_input = {"file_path": filepath}
    result = read_file(**action_input)

    assert result['status'] == "success"
    assert result['action'] == "file_read"
    assert result['data']['filepath'] == filepath
    assert result['data']['content'] == expected_content
    assert "lido com sucesso" in result['data']['message']
    mock_open_func.assert_called_once_with(mocker.ANY, "r", encoding="utf-8") # Use ANY for path due to mock complexity

def test_read_not_found(mocker, mock_path_exists):
    filepath = "non_existent.txt"
    mock_path_exists.return_value = False # Configure exists to return False

    action_input = {"file_path": filepath}
    result = read_file(**action_input)

    assert result['status'] == "error"
    assert result['action'] == "path_validation_failed"
    assert f"Arquivo não encontrado: '{filepath}'" == result['data']['message']

def test_read_is_directory(mocker, mock_path_exists, mock_path_is_file, mock_path_is_dir):
    filepath = "data/"
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = False # It's not a file
    mock_path_is_dir.return_value = True  # It's a directory

    action_input = {"file_path": filepath}
    result = read_file(**action_input)

    assert result['status'] == "error"
    assert result['action'] == "path_validation_failed"
    assert f"O caminho fornecido não é um arquivo: '{filepath}'" == result['data']['message']

def test_read_permission_error(mocker, mock_path_exists, mock_path_is_file, mock_path_resolve, mock_open_func):
    filepath = "restricted.txt"
    resolved_path_obj = Path(WORKSPACE_ROOT) / filepath

    # Configure mocks for validation to pass
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    # Patch stat globally for this test to ensure size check passes before open
    mocker.patch.object(Path, 'stat', return_value=MagicMock(st_size=50))

    # Mock open to raise PermissionError
    mock_open_func.side_effect = PermissionError("Permission denied by OS")

    action_input = {"file_path": filepath}
    result = read_file(**action_input)

    assert result['status'] == "error"
    assert result['action'] == "read_file_failed"
    assert f"Permissão negada para ler o arquivo: '{filepath}'" == result['data']['message']
    mock_open_func.assert_called_once_with(mocker.ANY, "r", encoding="utf-8")

def test_read_outside_workspace(mocker):
    filepath = "../../etc/passwd"
    action_input = {"file_path": filepath}
    result = read_file(**action_input)

    assert result['status'] == "error"
    assert result['action'] == "path_validation_failed"
    # assert "Path inválido: tentativa de acesso fora do workspace." == result['data']['message'] # <-- OLD
    assert "Path inválido: não use paths absolutos ou '..'. Use paths relativos dentro do workspace." == result['data']['message'] # <-- NEW

def test_missing_filepath_for_read():
    action_input = {}
    # result = read_file(**action_input) # This will raise TypeError
    # Instead, test the validation directly or expect the specific error dictionary
    # For simplicity, let's call with None and expect the error dict
    result = read_file(file_path=None)

    assert result['status'] == "error"
    assert result['action'] == "path_validation_failed"
    assert "Nome do arquivo inválido ou não fornecido." == result['data']['message']
