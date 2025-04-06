import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from a3x.skills.file_manager import FileManagerSkill

# Test the wrapped function directly, bypassing the decorator

@pytest.mark.asyncio
@patch("builtins.open", new_callable=MagicMock) # Mock open for tests that should write
async def test_create_file_success(mock_open, tmp_path): # Add mock_open
    """Test successfully creating a new file by calling the wrapped function."""
    original_path = "new_folder/new_file.txt"
    content_to_write = "This is the first line.\n"
    file_manager_instance = FileManagerSkill() # Need an instance for the 'self' argument

    # --- Mock Configuration for the resolved_path object ---
    resolved_path_mock = MagicMock(spec=Path, name="resolved_path_mock")
    resolved_path_mock.parent = MagicMock(spec=Path, name="resolved_path_mock.parent")
    mock_file_handle = MagicMock(name="mock_file_handle")
    # Configure mock_open (builtins) instead of resolved_path_mock.open
    mock_open.return_value.__enter__.return_value = mock_file_handle

    # Configure the mock for checks inside the function
    resolved_path_mock.exists.return_value = False # File doesn't exist before write
    resolved_path_mock.is_dir.return_value = False # It's not a directory

    # --- Execution ---
    result = await FileManagerSkill.write_file.__wrapped__(
        file_manager_instance, # self
        resolved_path=resolved_path_mock,
        original_path_str=original_path,
        content=content_to_write,
        overwrite=False,
        path=original_path
    )

    # --- Assertions ---
    assert result["status"] == "success"
    assert result["action"] == "file_created"
    assert original_path in result["data"]["message"]
    assert result["data"]["filepath"] == original_path

    # Verify calls on the mock path object and mock_open
    resolved_path_mock.exists.assert_called_once() # Check inside the function
    resolved_path_mock.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    # Assert on mock_open (builtins)
    mock_open.assert_called_once_with(resolved_path_mock, "w", encoding="utf-8")
    mock_file_handle.write.assert_called_once_with(content_to_write)

# Apply similar changes to other tests... (Example for overwrite)

@pytest.mark.asyncio
@patch("builtins.open", new_callable=MagicMock) # Mock open
async def test_create_file_overwrite_success(mock_open, tmp_path): # Add mock_open
    """Test successfully overwriting an existing file via wrapped function."""
    original_path = "existing_file.md"
    content_to_write = "# New Markdown Content\n"
    file_manager_instance = FileManagerSkill()

    resolved_path_mock = MagicMock(spec=Path, name="resolved_path_mock")
    resolved_path_mock.parent = MagicMock(spec=Path, name="resolved_path_mock.parent")
    mock_file_handle = MagicMock(name="mock_file_handle")
    # Configure mock_open
    mock_open.return_value.__enter__.return_value = mock_file_handle

    # Configure for overwrite
    resolved_path_mock.exists.return_value = True # File *does* exist
    resolved_path_mock.is_dir.return_value = False

    result = await FileManagerSkill.write_file.__wrapped__(
        file_manager_instance,
        resolved_path=resolved_path_mock,
        original_path_str=original_path,
        content=content_to_write,
        overwrite=True, # Overwrite is True
        path=original_path
    )

    assert result["status"] == "success"
    assert result["action"] == "file_overwritten"
    assert original_path in result["data"]["message"]
    assert result["data"]["filepath"] == original_path

    resolved_path_mock.exists.assert_called() # Called multiple times potentially
    resolved_path_mock.is_dir.assert_called_once() # Called once in the check
    resolved_path_mock.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    # Assert on mock_open
    mock_open.assert_called_once_with(resolved_path_mock, "w", encoding="utf-8")
    mock_file_handle.write.assert_called_once_with(content_to_write)


@pytest.mark.asyncio
async def test_create_file_exists_no_overwrite(tmp_path):
    """Test failing create when file exists (no overwrite) via wrapped function."""
    original_path = "dont_overwrite.txt"
    content_to_write = "some data"
    file_manager_instance = FileManagerSkill()

    resolved_path_mock = MagicMock(spec=Path, name="resolved_path_mock")
    # Configure for exists, no overwrite
    resolved_path_mock.exists.return_value = True
    resolved_path_mock.is_dir.return_value = False
    # No need to mock parent or open as they shouldn't be called

    result = await FileManagerSkill.write_file.__wrapped__(
        file_manager_instance,
        resolved_path=resolved_path_mock,
        original_path_str=original_path,
        content=content_to_write,
        overwrite=False, # Overwrite is False
        path=original_path
    )

    assert result["status"] == "error"
    assert result["action"] == "write_file_failed"
    assert "already exists" in result["data"]["message"]

    # Verify checks were made, but write didn't happen
    resolved_path_mock.exists.assert_called_once()
    resolved_path_mock.is_dir.assert_called_once()
    resolved_path_mock.open.assert_not_called()


@pytest.mark.asyncio
async def test_create_file_target_is_directory(tmp_path):
    """Test failing create when target is a directory via wrapped function."""
    original_path = "existing_dir"
    content_to_write = "some data"
    file_manager_instance = FileManagerSkill()

    resolved_path_mock = MagicMock(spec=Path, name="resolved_path_mock")
    # Configure for target is directory
    resolved_path_mock.exists.return_value = True
    resolved_path_mock.is_dir.return_value = True # It IS a directory

    result = await FileManagerSkill.write_file.__wrapped__(
        file_manager_instance,
        resolved_path=resolved_path_mock,
        original_path_str=original_path,
        content=content_to_write,
        overwrite=False, # Overwrite doesn't matter here
        path=original_path
    )

    assert result["status"] == "error"
    assert result["action"] == "write_file_failed"
    assert "directory already exists" in result["data"]["message"]

    # Verify checks were made, but write didn't happen
    resolved_path_mock.exists.assert_called_once()
    resolved_path_mock.is_dir.assert_called_once()
    resolved_path_mock.open.assert_not_called()


@pytest.mark.asyncio
@patch("builtins.open", new_callable=MagicMock) # Mock open
async def test_create_file_permission_error(mock_open, tmp_path): # Add mock_open
    """Test handling PermissionError during write via wrapped function."""
    original_path = "protected/config.ini"
    content_to_write = "data"
    file_manager_instance = FileManagerSkill()

    resolved_path_mock = MagicMock(spec=Path, name="resolved_path_mock")
    resolved_path_mock.parent = MagicMock(spec=Path, name="resolved_path_mock.parent")
    # Configure mock_open to raise PermissionError
    mock_open.side_effect = PermissionError("Cannot write there")

    # Configure other checks before open
    resolved_path_mock.exists.return_value = False
    resolved_path_mock.is_dir.return_value = False

    result = await FileManagerSkill.write_file.__wrapped__(
        file_manager_instance,
        resolved_path=resolved_path_mock,
        original_path_str=original_path,
        content=content_to_write,
        overwrite=False,
        path=original_path
    )

    assert result["status"] == "error"
    assert result["action"] == "write_file_failed"
    assert "Permission denied" in result["data"]["message"]

    # Verify checks, mkdir and open were called (open raised the error)
    resolved_path_mock.exists.assert_called_once() # Initial check
    # resolved_path_mock.is_dir.assert_called_once() # This is NOT called if exists() is False
    resolved_path_mock.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    # Assert on mock_open
    mock_open.assert_called_once_with(resolved_path_mock, "w", encoding="utf-8")
