# tests/test_read_file_skill.py
import sys
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# --- Add project root to sys.path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the skill function AFTER adjusting path
from skills.read_file import read_file
from core.config import PROJECT_ROOT # Import workspace root for reference

# Define the temporary directory for test files
TEMP_TEST_DIR = Path(PROJECT_ROOT) / "tests" / "temp_test_files"

# Ensure the temp directory exists (though we created it earlier)
@pytest.fixture(scope="module", autouse=True)
def ensure_temp_dir():
    TEMP_TEST_DIR.mkdir(parents=True, exist_ok=True)
    # Optional: Add cleanup logic here if needed after tests run
    yield
    # shutil.rmtree(TEMP_TEST_DIR) # Example cleanup

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"

# Helper function to create a temporary file with content
@pytest.fixture
def temp_test_file(tmp_path):
    test_dir = tmp_path / "temp_test_files"
    test_dir.mkdir()
    file_path = test_dir / "test_read.txt"
    content = "Este é um arquivo de teste para leitura.\nContém algumas linhas de texto.\n"
    file_path.write_text(content, encoding='utf-8')
    return file_path

# Fixture to patch WORKSPACE_ROOT to point to the temp dir parent
@pytest.fixture
def patch_workspace(temp_test_file):
    mock_workspace = temp_test_file.parent.parent # The tmp_path itself
    with patch('skills.read_file.WORKSPACE_ROOT', str(mock_workspace)):
        yield str(mock_workspace)

# --- Test Cases ---

def test_read_file_success(patch_workspace, temp_test_file):
    """Tests successful reading of a valid text file."""
    # Use the relative path from the *patched* workspace root
    file_path_rel = temp_test_file.relative_to(patch_workspace)
    expected_content = "Este é um arquivo de teste para leitura.\nContém algumas linhas de texto.\n"
    
    result = read_file(file_path=str(file_path_rel))

    assert result["status"] == "success"
    assert result["action"] == "file_read"
    assert "data" in result
    assert result["data"]["filepath"] == str(file_path_rel)
    assert result["data"]["content"] == expected_content
    assert "message" in result["data"]
    print(f"Success Test Result: {result}") # Print for confirmation

def test_read_file_not_found(patch_workspace):
    """Tests reading a file that does not exist."""
    file_path = "non_existent_file.txt"
    result = read_file(file_path=file_path)

    assert result["status"] == "error"
    assert result["action"] == "read_failed"
    assert "não encontrado" in result["data"]["message"]
    print(f"Not Found Test Result: {result}")

def test_read_directory(patch_workspace, temp_test_file):
    """Tests attempting to read a directory."""
    dir_path_rel = temp_test_file.parent.relative_to(patch_workspace)

    result = read_file(file_path=str(dir_path_rel))

    assert result["status"] == "error"
    assert result["action"] == "read_failed"
    # Expecting error from the decorator or the function
    assert "Cannot read a directory" in result["data"]["message"]
    print(f"Directory Test Result: {result}")

def test_read_outside_workspace(tmp_path):
    """Tests attempting to read a file outside the (mocked) workspace."""
    # Create a file outside the standard temp structure used by patch_workspace
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("Secret data")

    # WORKSPACE_ROOT is not patched here, so it uses the real one
    # We try to access the absolute path which should be outside.
    file_path_abs = str(outside_file.resolve())

    result = read_file(file_path=file_path_abs)

    assert result["status"] == "error"
    assert result["action"] == "read_failed"
    # Expecting validation error from the decorator
    assert "Path validation failed" in result["data"]["message"]
    assert "outside the workspace" in result["data"]["message"]
    print(f"Outside Workspace Test Result: {result}")

def test_read_file_permission_error(patch_workspace, temp_test_file):
    """Tests reading a file with a permission error."""
    file_path_rel = temp_test_file.relative_to(patch_workspace)

    # Patch open to raise PermissionError
    with patch("builtins.open", side_effect=PermissionError("Permission denied")):
        result = read_file(file_path=str(file_path_rel))

    assert result["status"] == "error"
    assert result["action"] == "read_failed"
    assert "Permissão negada" in result["data"]["message"]

# Para rodar: pytest tests/test_read_file_skill.py -v -s 