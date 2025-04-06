import pytest
from pathlib import Path  # Import real Path
from unittest.mock import MagicMock

# Import the skill function
from a3x.skills.file_manager import FileManagerSkill

# Remove MOCK_WORKSPACE_ROOT and mock_resolve helper if not needed by other tests


# Test the wrapped function directly
@pytest.mark.asyncio
async def test_append_to_file_success(tmp_path):
    """Test successfully appending content via wrapped function."""
    original_path = "test_dir/output.txt"
    content_to_append = "New line of text.\n"
    file_manager_instance = FileManagerSkill()

    # --- Mock Configuration ---
    resolved_path_mock = MagicMock(spec=Path, name="resolved_path_mock")
    resolved_path_mock.parent = MagicMock(spec=Path, name="resolved_path_mock.parent")
    mock_file_handle = MagicMock(name="mock_file_handle")
    resolved_path_mock.open.return_value.__enter__.return_value = mock_file_handle

    # No specific path checks needed for append logic itself

    # --- Execution ---
    result = await FileManagerSkill.append_to_file.__wrapped__(
        file_manager_instance,  # self
        resolved_path=resolved_path_mock,
        original_path_str=original_path,
        content=content_to_append,
        path=original_path,  # Pass original path too
    )

    # --- Assertions ---
    assert result["status"] == "success"
    assert result["action"] == "file_appended"
    assert original_path in result["data"]["message"]
    assert result["data"]["filepath"] == original_path

    # Verify open and write calls on the resolved_path_mock
    resolved_path_mock.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    resolved_path_mock.open.assert_called_once_with(
        "a", encoding="utf-8"
    )  # Use resolved_path_mock.open
    mock_file_handle.write.assert_called_once_with(content_to_append)


@pytest.mark.asyncio
async def test_append_to_file_no_newline(tmp_path):
    """Test appending content without newline via wrapped function."""
    original_path = "another/file.log"
    content_to_append = "Log entry without newline"
    expected_content = content_to_append + "\n"  # Logic adds a newline
    file_manager_instance = FileManagerSkill()

    resolved_path_mock = MagicMock(spec=Path, name="resolved_path_mock")
    resolved_path_mock.parent = MagicMock(spec=Path, name="resolved_path_mock.parent")
    mock_file_handle = MagicMock(name="mock_file_handle")
    resolved_path_mock.open.return_value.__enter__.return_value = mock_file_handle

    result = await FileManagerSkill.append_to_file.__wrapped__(
        file_manager_instance,
        resolved_path=resolved_path_mock,
        original_path_str=original_path,
        content=content_to_append,
        path=original_path,
    )

    assert result["status"] == "success"
    resolved_path_mock.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    resolved_path_mock.open.assert_called_once_with(
        "a", encoding="utf-8"
    )  # Use resolved_path_mock.open
    mock_file_handle.write.assert_called_once_with(expected_content)


@pytest.mark.asyncio
async def test_append_to_file_permission_error(tmp_path):
    """Test PermissionError during append via wrapped function."""
    original_path = "locked/file.dat"
    content_to_append = "data"
    file_manager_instance = FileManagerSkill()

    resolved_path_mock = MagicMock(spec=Path, name="resolved_path_mock")
    resolved_path_mock.parent = MagicMock(spec=Path, name="resolved_path_mock.parent")
    # Configure open ON the resolved_path_mock to raise PermissionError
    resolved_path_mock.open.side_effect = PermissionError("Permission denied")

    result = await FileManagerSkill.append_to_file.__wrapped__(
        file_manager_instance,
        resolved_path=resolved_path_mock,
        original_path_str=original_path,
        content=content_to_append,
        path=original_path,
    )

    # Check results for error
    assert result["status"] == "error"  # Check status is error
    assert result["action"] == "append_failed"
    assert "Permission denied" in result["data"]["message"]
    assert original_path in result["data"]["message"]

    # Verify mkdir and open were called
    resolved_path_mock.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    resolved_path_mock.open.assert_called_once_with(
        "a", encoding="utf-8"
    )  # Use resolved_path_mock.open


# Add test for IsADirectoryError if necessary (though should be caught by decorator in real use)
# Add test for generic Exception if necessary
