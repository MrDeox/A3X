import pytest
from pathlib import Path
import os

# Assume PROJECT_ROOT is determined and added to sys.path correctly by the test runner (e.g., pytest)
# Or explicitly add it here if needed for isolated runs:
# import sys
# PROJECT_ROOT = Path(__file__).parent.parent.parent # Adjust levels as necessary
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Assuming FileManagerSkill provides these as methods
    from a3x.skills.file_manager import FileManagerSkill
    # If they were standalone functions:
    # from a3x.skills.file_manager import read_file, write_file
except ImportError:
    pytest.skip("Skipping file_manager tests due to import errors.", allow_module_level=True)

# --- Fixtures ---

@pytest.fixture(scope="module")
def file_manager() -> FileManagerSkill:
    """Provides a FileManagerSkill instance for the test module."""
    # No workspace needed if paths are absolute or relative to CWD during test
    return FileManagerSkill()

@pytest.fixture
def temp_test_dir(tmp_path) -> Path:
    """Creates a temporary directory for test files."""
    test_dir = tmp_path / "file_manager_tests"
    test_dir.mkdir()
    return test_dir

# --- Test Cases ---

# == read_file Tests ==

@pytest.mark.asyncio
async def test_read_existing_file(file_manager, temp_test_dir):
    """Test reading content from an existing file."""
    file_path = temp_test_dir / "read_test.txt"
    expected_content = "Hello, A³X!\nLine 2."
    file_path.write_text(expected_content)

    # Use absolute path for the skill
    result = await file_manager.read_file(str(file_path.resolve()))

    assert result["status"] == "success"
    assert "content" in result["data"]
    assert result["data"]["content"] == expected_content
    assert result["data"]["path_used"] == str(file_path.resolve())

@pytest.mark.asyncio
async def test_read_non_existent_file(file_manager, temp_test_dir):
    """Test reading from a file that does not exist."""
    file_path = temp_test_dir / "non_existent.txt"

    result = await file_manager.read_file(str(file_path.resolve()))

    assert result["status"] == "error"
    assert "FileNotFoundError" in result.get("error_type", "")
    assert result["data"] is None # Or expect specific error data structure

@pytest.mark.asyncio
async def test_read_file_relative_path(file_manager, temp_test_dir):
    """Test reading using a path relative to the current working directory (during test)."""
    relative_path = "relative_read_test.txt"
    file_path_abs = temp_test_dir / relative_path
    expected_content = "Relative path content."
    file_path_abs.write_text(expected_content)

    # Change CWD for the test
    original_cwd = os.getcwd()
    os.chdir(temp_test_dir)
    try:
        result = await file_manager.read_file(relative_path) # Pass relative path
        assert result["status"] == "success"
        assert result["data"]["content"] == expected_content
        # Check if the skill resolved it to an absolute path
        assert result["data"]["path_used"] == str(file_path_abs.resolve())
    finally:
        os.chdir(original_cwd) # Restore CWD

# == write_file Tests ==

@pytest.mark.asyncio
async def test_write_new_file(file_manager, temp_test_dir):
    """Test writing content to a new file."""
    file_path = temp_test_dir / "write_new.txt"
    content_to_write = "This is a new file."

    assert not file_path.exists() # Ensure file doesn't exist initially

    result = await file_manager.write_file(
        filename=str(file_path.resolve()),
        content=content_to_write,
        overwrite=False # Default, but explicit
    )

    assert result["status"] == "success"
    assert file_path.exists()
    assert file_path.read_text() == content_to_write
    assert result["data"]["path_written"] == str(file_path.resolve())
    assert result["data"].get("backup_created") is None # No backup needed

@pytest.mark.asyncio
async def test_write_overwrite_existing_file(file_manager, temp_test_dir):
    """Test overwriting an existing file."""
    file_path = temp_test_dir / "write_overwrite.txt"
    initial_content = "Initial content."
    new_content = "Overwritten content."
    file_path.write_text(initial_content)

    result = await file_manager.write_file(
        filename=str(file_path.resolve()),
        content=new_content,
        overwrite=True,
        create_backup_flag=False # Test without backup for simplicity here
    )

    assert result["status"] == "success"
    assert file_path.read_text() == new_content
    assert result["data"].get("backup_created") is None

@pytest.mark.asyncio
async def test_write_fail_on_existing_file_no_overwrite(file_manager, temp_test_dir):
    """Test that writing fails if file exists and overwrite is False."""
    file_path = temp_test_dir / "write_fail.txt"
    initial_content = "Do not overwrite."
    file_path.write_text(initial_content)

    result = await file_manager.write_file(
        filename=str(file_path.resolve()),
        content="Attempted write.",
        overwrite=False
    )

    assert result["status"] == "error"
    assert "FileExistsError" in result.get("error_type", "")
    assert file_path.read_text() == initial_content # Content should be unchanged

@pytest.mark.asyncio
async def test_write_with_backup(file_manager, temp_test_dir):
    """Test writing with backup creation."""
    file_path = temp_test_dir / "write_backup.txt"
    initial_content = "Content to be backed up."
    new_content = "New content after backup."
    file_path.write_text(initial_content)

    result = await file_manager.write_file(
        filename=str(file_path.resolve()),
        content=new_content,
        overwrite=True,
        create_backup_flag=True # Explicitly request backup
    )

    assert result["status"] == "success"
    assert file_path.read_text() == new_content
    assert result["data"]["backup_created"] is not None
    backup_path_str = result["data"]["backup_created"]
    backup_path = Path(backup_path_str)
    assert backup_path.exists()
    assert backup_path.name.startswith(f"{file_path.name}.") # Check naming convention
    assert backup_path.name.endswith(".bak")
    assert backup_path.read_text() == initial_content

# TODO: Add tests for edge cases:
# - Writing to invalid paths (e.g., permission errors - might need more setup)
# - Reading/Writing very large files (if applicable)
# - Handling different encodings (if the skill supports it)
# - Writing to a non-existent directory (does it create or fail?)

# Example test for non-existent directory (assuming it should fail by default)
@pytest.mark.asyncio
async def test_write_to_non_existent_dir(file_manager, temp_test_dir):
    """Test writing to a path where the parent directory doesn't exist."""
    file_path = temp_test_dir / "non_existent_subdir" / "new_file.txt"

    result = await file_manager.write_file(
        filename=str(file_path.resolve()),
        content="Should fail."
    )

    assert result["status"] == "error"
    # The exact error might vary (FileNotFoundError, NotADirectoryError depending on implementation)
    assert "FileNotFoundError" in result.get("error_type", "") or "NotADirectoryError" in result.get("error_type", "")
    assert not file_path.parent.exists() 