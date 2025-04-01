import pytest
from unittest.mock import MagicMock

# Import the skill class and relevant exceptions/constants
from skills.file_manager import FileManagerSkill

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio


# Helper fixture to provide an instance of the skill with a temporary workspace
@pytest.fixture
def file_manager(mocker, temp_workspace_files):
    """Provides an instance of FileManagerSkill with a mocked workspace path."""
    # # Patch the WORKSPACE_PATH used by the validator (REMOVED)
    # mocker.patch("core.validators.WORKSPACE_PATH", temp_workspace_files)
    # # Patch the PROJECT_ROOT used elsewhere (REMOVED)
    # mocker.patch("core.config.PROJECT_ROOT", temp_workspace_files)
    # # Patch the WORKSPACE_PATH potentially used by the backup module (REMOVED)
    # mocker.patch("core.backup.WORKSPACE_PATH", temp_workspace_files, create=True)

    # Instantiate the skill, passing the temporary workspace path
    return FileManagerSkill(workspace_root=temp_workspace_files)


# --- Test Cases for create_file ---


async def test_create_file_success(file_manager, temp_workspace_files):
    """Test successfully creating a new file in the temp workspace."""
    relative_path = "new_folder/new_file.txt"
    absolute_path = temp_workspace_files / relative_path
    content = "Hello from test!\\nLine 2."

    # Action: Call the skill method (Explicitly pass overwrite=False)
    result = await file_manager.write_file(
        filepath=relative_path, content=content, overwrite=False
    )

    # Assertions
    assert result["status"] == "success"
    assert absolute_path.exists()
    assert absolute_path.read_text(encoding="utf-8") == content
    assert result["action"] == "file_created"  # Assuming this is the action name
    assert (
        f"File '{relative_path}' was successfully file created."
        == result["data"]["message"]
    )


async def test_create_file_no_content(file_manager, temp_workspace_files):
    """Test creating an empty file when content is None or empty."""
    relative_path = "empty_file.txt"
    absolute_path = temp_workspace_files / relative_path

    # Test with content="" (Explicitly pass overwrite=False)
    result_empty = await file_manager.write_file(
        filepath=relative_path, content="", overwrite=False
    )
    assert result_empty["status"] == "success"
    assert absolute_path.exists()
    assert absolute_path.read_text(encoding="utf-8") == ""
    assert result_empty["action"] == "file_created"  # Check action name
    assert (
        f"File '{relative_path}' was successfully file created."
        == result_empty["data"]["message"]
    )


async def test_create_file_overwrite_success(file_manager, temp_workspace_files):
    """Test successfully overwriting an existing file."""
    relative_path = "overwrite_me.txt"
    absolute_path = temp_workspace_files / relative_path
    initial_content = "Old content."
    new_content = "New content!"

    # Create the initial file
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_text(initial_content, encoding="utf-8")

    # Action: Call write_file with overwrite=True
    result = await file_manager.write_file(
        filepath=relative_path, content=new_content, overwrite=True
    )

    # Assertions
    assert result["status"] == "success"
    assert result["action"] == "file_overwritten"
    assert (
        f"File '{relative_path}' was successfully file overwritten."
        == result["data"]["message"]
    )
    assert absolute_path.exists()
    assert (
        absolute_path.read_text(encoding="utf-8") == new_content
    )  # Verify content was overwritten


async def test_create_file_exists_no_overwrite(file_manager, temp_workspace_files):
    """Test failing to create when file exists and overwrite is False."""
    relative_path = "no_overwrite.log"
    absolute_path = temp_workspace_files / relative_path
    initial_content = "Original log entry."

    # Create the initial file
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_text(initial_content, encoding="utf-8")

    # Action: Call write_file with overwrite=False (explicitly)
    result = await file_manager.write_file(
        filepath=relative_path, content="New content", overwrite=False
    )

    # Assertions
    assert result["status"] == "error"
    assert (
        result["action"] == "write_file_failed"
    )  # Check actual action name from skill
    assert (
        f"File '{relative_path}' already exists. Use overwrite=True to replace it."
        == result["data"]["message"]
    )
    assert (
        absolute_path.read_text(encoding="utf-8") == initial_content
    )  # File unchanged


async def test_create_file_target_is_directory(file_manager, temp_workspace_files):
    """Test failing to create a file where a directory with the same name exists."""
    relative_path = "folder_as_file"
    absolute_path = temp_workspace_files / relative_path

    # Create a directory at the target path
    absolute_path.mkdir(parents=True, exist_ok=True)

    # Action: Attempt to create a file at the directory path (Explicitly pass overwrite=False)
    result = await file_manager.write_file(
        filepath=relative_path, content="Some content", overwrite=False
    )

    # Assertions
    assert result["status"] == "error"
    assert result["action"] == "write_file_failed"  # Correct expected action
    assert (
        f"Cannot create file, a directory already exists at '{relative_path}'"
        in result["data"]["message"]
    )


async def test_create_file_permission_error(file_manager, temp_workspace_files, mocker):
    """Test handling PermissionError during file creation using write_file."""
    relative_path = "protected/new_file.txt"
    absolute_path = temp_workspace_files / relative_path
    content = "Secret data"

    absolute_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Mock open WITHIN the skill module scope ---
    mock_skill_open = mocker.patch(
        "skills.file_manager.open",
        side_effect=PermissionError("Mock Write Permission Denied"),
    )

    result = await file_manager.write_file(
        filepath=relative_path, content=content, overwrite=False
    )

    assert result["status"] == "error"
    assert result["action"] == "write_file_failed"
    # Check the specific error message from the skill's except block (with colon)
    assert (
        f"Permission denied to write file: '{relative_path}'"
        in result["data"]["message"]
    )
    # Ensure the mock was called (verify it tried to open)
    mock_skill_open.assert_called_once()
    # Check that the actual file wasn't created on the filesystem
    assert not absolute_path.exists()


# --- Test Cases for append_to_file ---


async def test_append_to_file_success(file_manager, temp_workspace_files):
    """Test successfully appending content to an existing file."""
    relative_path = "my_log.txt"
    absolute_path = temp_workspace_files / relative_path
    initial_content = "Line 1\n"
    content_to_append = "Line 2\n"
    # Corrected expected content (no extra newline)
    expected_content = initial_content + content_to_append

    # Create the initial file
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_text(initial_content, encoding="utf-8")

    # Action: Call the skill method with await
    result = await file_manager.append_to_file(
        filepath=relative_path, content=content_to_append
    )

    # Assertions
    assert result["status"] == "success"
    assert result["action"] == "file_appended"
    assert (
        f"Content successfully appended to file '{relative_path}'."
        == result["data"]["message"]
    )
    assert result["data"]["filepath"] == relative_path

    # Verify file system state
    assert absolute_path.exists()
    assert absolute_path.read_text(encoding="utf-8") == expected_content


async def test_append_to_file_creates_if_not_exists(file_manager, temp_workspace_files):
    """Test that append_to_file creates the file if it doesn't exist."""
    relative_path = "newly_created_log.txt"
    absolute_path = temp_workspace_files / relative_path
    content_to_append = "First line in new file.\\n"

    # Ensure file does not exist initially
    assert not absolute_path.exists()

    # Action: Call the skill method with await
    result = await file_manager.append_to_file(
        filepath=relative_path, content=content_to_append
    )

    # Assertions (Expecting success after validator fix)
    assert result["status"] == "success"
    assert absolute_path.exists()
    assert absolute_path.read_text(encoding="utf-8") == content_to_append + "\n"
    assert (
        result["action"] == "file_appended"
    )  # Or maybe "file_created_and_appended"? Check skill.
    assert (
        f"Content successfully appended to file '{relative_path}'."
        == result["data"]["message"]
    )


async def test_append_to_file_adds_newline(file_manager, temp_workspace_files):
    """Test that append_to_file automatically adds a newline if missing."""
    relative_path = "auto_newline.txt"
    absolute_path = temp_workspace_files / relative_path
    initial_content = "Existing line."  # No newline
    content_to_append = "Appended line"  # No newline
    # Expected: Initial content + appended content + ONE newline added by the skill
    expected_content = initial_content + content_to_append + "\n"

    # Create the initial file without a trailing newline
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_text(initial_content, encoding="utf-8")

    # Action: Call the skill method with await
    result = await file_manager.append_to_file(
        filepath=relative_path, content=content_to_append
    )

    # Assertions
    assert result["status"] == "success"
    assert absolute_path.read_text(encoding="utf-8") == expected_content


async def test_append_to_file_permission_error(
    file_manager, temp_workspace_files, mocker
):
    """Test handling PermissionError during file append."""
    relative_path = "locked_dir/append_fail.log"
    original_path_str = relative_path  # Store for assertion message
    absolute_path = temp_workspace_files / relative_path
    content_to_append = "Important data"

    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.touch()

    # --- Mock Path.open WITHIN the skill module scope AGAIN ---
    def mock_path_open_no_fallback(self, mode="r", *args, **kwargs):
        # Check if 'self' (the Path object) matches our target path AND mode is append
        if self.resolve() == absolute_path.resolve() and "a" in mode:
            raise PermissionError("Mock Append Permission Denied")
        # --- NO Fallback ---
        # If the target/mode doesn't match, do nothing or return a basic mock
        # This prevents the weird FileNotFoundError: 'a' or AttributeError from fallback
        return MagicMock()  # Return a dummy mock object to satisfy the 'with' statement

    # Add autospec=True for potentially better signature matching
    mocker.patch(
        "skills.file_manager.Path.open",
        side_effect=mock_path_open_no_fallback,
        autospec=True,
    )

    result = await file_manager.append_to_file(
        filepath=relative_path, content=content_to_append
    )

    assert result["status"] == "error"
    assert result["action"] == "append_failed"
    # Check the specific error message from the skill's except block
    assert (
        f"Permission denied to append to file: '{original_path_str}'"
        in result["data"]["message"]
    )  # Corrected message format


async def test_append_to_file_target_is_directory(file_manager, temp_workspace_files):
    """Test failing to append to a path that is a directory."""
    relative_path = "folder_to_append"
    absolute_path = temp_workspace_files / relative_path

    # Create a directory at the target path
    absolute_path.mkdir(parents=True, exist_ok=True)

    # Action: Attempt to append to the directory path with await
    result = await file_manager.append_to_file(
        filepath=relative_path, content="Some content"
    )

    # Assertions
    assert result["status"] == "error"
    # The specific error might vary (IsADirectoryError during open, or caught earlier)
    # Check the FileManagerSkill implementation for the exact action/message
    assert result["action"] == "path_validation_failed"
    assert "Path is not a file" in result["data"]["message"]  # Or similar error message
    assert absolute_path.is_dir()  # Verify it's still a directory


# --- Test Cases for read_file ---
# TODO: Migrate tests from test_read.py and test_read_file_skill.py


async def test_read_file_success(file_manager, temp_workspace_files):
    """Test successfully reading an existing file."""
    relative_path = "read_folder/readable.txt"
    absolute_path = temp_workspace_files / relative_path
    content = "Line 1\\nLine 2\\nTest content."

    # Create the file
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_text(content, encoding="utf-8")

    # Action: Call the skill method with await
    result = await file_manager.read_file(filepath=relative_path)

    # Assertions
    assert result["status"] == "success"
    assert result["action"] == "file_read"
    assert result["data"]["filepath"] == relative_path
    assert result["data"]["content"] == content
    assert f"File '{relative_path}' read successfully" in result["data"]["message"]


async def test_read_file_not_found(file_manager):
    """Test reading a file that does not exist."""
    relative_path = "non_existent_folder/missing_file.dat"

    # Action: Call the skill method with await
    result = await file_manager.read_file(filepath=relative_path)

    # Assertions
    assert result["status"] == "error"
    assert result["action"] == "path_validation_failed"
    assert "Path not found" in result["data"]["message"]
    assert relative_path in result["data"]["message"]


async def test_read_file_target_is_directory(file_manager, temp_workspace_files):
    """Test failing to read a path that is a directory."""
    relative_path = "this_is_a_folder"
    absolute_path = temp_workspace_files / relative_path

    # Create a directory at the target path
    absolute_path.mkdir(parents=True, exist_ok=True)

    # Action: Attempt to read the directory path with await
    result = await file_manager.read_file(filepath=relative_path)

    # Assertions
    assert result["status"] == "error"
    assert result["action"] == "path_validation_failed"
    assert "Path is not a file" in result["data"]["message"]
    assert relative_path in result["data"]["message"]
    assert absolute_path.is_dir()  # Verify it's still a directory


async def test_read_file_permission_error(file_manager, temp_workspace_files, mocker):
    """Test handling PermissionError during file read."""
    relative_path = "forbidden_zone/secret.txt"
    original_path_str = relative_path  # Store for assertion message
    absolute_path = temp_workspace_files / relative_path
    content = "You cannot see this!"

    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_text(content, encoding="utf-8")

    # --- Mock open WITHIN the skill module scope for read mode 'r' ---
    original_open = open  # Keep reference to original builtin

    def mock_skill_open_read_permission(*args, **kwargs):
        path_arg = args[0]
        # Default mode for open is 'r'
        mode_arg = args[1] if len(args) > 1 else kwargs.get("mode", "r")
        # Check if path matches and it's any kind of read mode
        if str(path_arg) == str(absolute_path) and (
            "r" in mode_arg or mode_arg == "rt" or mode_arg == "rb"
        ):
            raise PermissionError("Mock Read Permission Denied")
        # Fallback to original builtin open if not our target
        return original_open(*args, **kwargs)

    mock_skill_open = mocker.patch(
        "skills.file_manager.open", side_effect=mock_skill_open_read_permission
    )

    result = await file_manager.read_file(filepath=relative_path)

    assert result["status"] == "error"
    assert result["action"] == "read_file_failed"
    # Check the specific error message from the skill's except block
    assert (
        f"Permission denied to read file: '{original_path_str}'"
        in result["data"]["message"]
    )  # Corrected message format
    # Ensure the mock was called
    mock_skill_open.assert_called_once()


# Use the standard file manager fixture
async def test_read_file_invalid_path_relative_outside(file_manager):
    """Test failing to read a file using relative path outside workspace."""
    relative_path = "../confidential.cfg"  # Attempt to go up one level

    # Action: Attempt to read the file outside the allowed workspace with await
    result = await file_manager.read_file(filepath=relative_path)

    # Assertions
    assert result["status"] == "error"
    assert result["action"] == "path_validation_failed"  # Validator should catch this
    # Check for the specific error message from the validator
    assert (
        f"Access denied: Path '{relative_path}' resolves outside the designated workspace"
        in result["data"]["message"]
    )


async def test_read_file_invalid_path_absolute(file_manager):
    """Test failing to read a file using an absolute path."""
    absolute_path = "/tmp/some_other_file.log"  # Changed path for more realism

    # Action: Attempt to read the file using an absolute path with await
    result = await file_manager.read_file(filepath=absolute_path)

    # Assertions
    assert result["status"] == "error"
    assert result["action"] == "path_validation_failed"  # Validator should catch this
    # Check for the specific error message
    assert (
        f"Access denied: Path '{absolute_path}' resolves outside the designated workspace"
        in result["data"]["message"]
    )


# --- Test Cases for list_files ---
# Renamed list_files -> list_directory, adjusted params/assertions


async def test_list_files_success(file_manager, temp_workspace_files):
    """Test successfully listing files and directories."""
    base_dir = temp_workspace_files
    # Create structure
    (base_dir / "folder1").mkdir()
    (base_dir / "folder1" / "file1.txt").write_text("content1")
    (base_dir / "folder2").mkdir()
    (base_dir / "file_top.log").write_text("log")
    # Hidden file/folder (should be ignored by default)
    (base_dir / ".hidden_folder").mkdir()
    (base_dir / ".hidden_file").write_text("secret")

    # List root with await
    result_root = await file_manager.list_directory(directory=".")  # Renamed call
    assert result_root["status"] == "success"
    assert result_root["action"] == "directory_listed"
    expected_root = sorted(["folder1/", "folder2/", "file_top.log"])
    assert sorted(result_root["data"]["items"]) == expected_root

    # List subfolder with await
    result_folder1 = await file_manager.list_directory(
        directory="folder1"
    )  # Renamed call
    assert result_folder1["status"] == "success"
    # The paths in items should be relative to the workspace root
    expected_folder1 = sorted(["folder1/file1.txt"])
    assert sorted(result_folder1["data"]["items"]) == expected_folder1


async def test_list_files_empty_directory(file_manager, temp_workspace_files):
    """Test listing an empty directory."""
    relative_path = "empty_dir"
    absolute_path = temp_workspace_files / relative_path
    absolute_path.mkdir(parents=True, exist_ok=True)

    # Action with await
    result = await file_manager.list_directory(directory=relative_path)  # Renamed call

    # Assertions
    assert result["status"] == "success"
    assert result["action"] == "directory_listed"
    assert result["data"]["items"] == []
    assert "0 non-hidden item(s) found" in result["data"]["message"]


async def test_list_files_excludes_hidden_by_default(
    file_manager, temp_workspace_files
):  # Renamed test
    """Test that listing files excludes hidden ones by default."""  # Updated docstring
    base_dir = temp_workspace_files
    (base_dir / "visible.txt").write_text("visible")
    (base_dir / ".hidden_file").write_text("hidden")
    (base_dir / ".hidden_dir").mkdir()
    (base_dir / ".hidden_dir" / "inside.txt").write_text("inside hidden")

    # Action with await
    result = await file_manager.list_directory(
        directory="."
    )  # Renamed call, no include_hidden

    # Assertions (expecting hidden to be excluded)
    assert result["status"] == "success"
    expected_items = sorted(["visible.txt"])  # Only visible
    assert sorted(result["data"]["items"]) == expected_items
    assert "1 non-hidden item(s) found" in result["data"]["message"]  # Updated count


async def test_list_files_is_not_recursive_by_default(
    file_manager, temp_workspace_files
):  # Renamed test
    """Test that listing files is not recursive by default."""  # Updated docstring
    base_dir = temp_workspace_files
    (base_dir / "level1").mkdir()
    (base_dir / "level1" / "file1.txt").write_text("1")
    (base_dir / "level1" / "level2").mkdir()
    (base_dir / "level1" / "level2" / "file2.txt").write_text("2")
    (base_dir / "top_file.md").write_text("top")

    # Action with await
    result = await file_manager.list_directory(
        directory="."
    )  # Renamed call, no recursive

    # Assertions (expecting only top level)
    assert result["status"] == "success"
    expected_items = sorted(["level1/", "top_file.md"])  # Only top level items
    assert sorted(result["data"]["items"]) == expected_items
    assert "2 non-hidden item(s) found" in result["data"]["message"]  # Updated count


async def test_list_files_directory_not_found(file_manager):
    """Test listing a non-existent directory."""
    relative_path = "no_such_folder"

    # Action with await
    result = await file_manager.list_directory(directory=relative_path)  # Renamed call

    # Assertions
    assert result["status"] == "error"
    assert result["action"] == "path_validation_failed"  # Validator
    assert "Path not found" in result["data"]["message"]
    assert relative_path in result["data"]["message"]


async def test_list_files_target_is_file(file_manager, temp_workspace_files):
    """Test listing a path that is a file, not a directory."""
    relative_path = "just_a_file.ini"
    absolute_path = temp_workspace_files / relative_path
    absolute_path.write_text("[config]")

    # Action with await
    result = await file_manager.list_directory(directory=relative_path)  # Renamed call

    # Assertions
    assert result["status"] == "error"
    assert result["action"] == "path_validation_failed"  # Validator
    assert "is not a directory" in result["data"]["message"]
    assert relative_path in result["data"]["message"]


# Permission error test for list_files skipped as noted before


# Use the standard file manager fixture
async def test_list_files_invalid_path_relative_outside(
    file_manager,
):  # Changed fixture
    """Test failing to list files using relative path outside workspace."""
    relative_path = "../../some_other_project"

    # Action with await
    result = await file_manager.list_directory(directory=relative_path)  # Renamed call

    # Assertions
    assert result["status"] == "error"
    assert result["action"] == "path_validation_failed"  # Validator
    # Check for the specific error message
    assert (
        f"Access denied: Path '{relative_path}' resolves outside the designated workspace"
        in result["data"]["message"]
    )


async def test_list_files_invalid_path_absolute(file_manager):  # Changed fixture
    """Test failing to list files using an absolute path."""
    absolute_path = "/var/log"

    # Action with await
    result = await file_manager.list_directory(directory=absolute_path)  # Renamed call

    # Assertions
    assert result["status"] == "error"
    assert result["action"] == "path_validation_failed"  # Validator
    # Check for the specific error message
    assert (
        f"Access denied: Path '{absolute_path}' resolves outside the designated workspace"
        in result["data"]["message"]
    )


# --- Test Cases for delete_path ---
# Renamed delete_file -> delete_path


async def test_delete_file_success(file_manager, temp_workspace_files):
    """Test successfully deleting an existing file."""
    relative_path = "to_be_deleted.tmp"
    absolute_path = temp_workspace_files / relative_path

    # Create the file
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_text("delete me")
    assert absolute_path.exists()

    # Action: Call the skill method with backup=True with await
    result = await file_manager.delete_path(
        filepath=relative_path, backup=True
    )  # Pass backup=True

    # Assertions (Backup and delete SHOULD SUCCEED)
    assert result["status"] == "success"
    assert result["action"] == "file_deleted"
    assert f"File '{relative_path}' successfully deleted." in result["data"]["message"]
    assert not absolute_path.exists()  # File should be deleted
    assert "backup_path" in result["data"] and result["data"]["backup_path"] is not None


async def test_delete_file_not_found(file_manager):
    """Test attempting to delete a file that does not exist."""
    relative_path = "already_gone.txt"

    # Action: Call the skill method with await
    result = await file_manager.delete_path(
        filepath=relative_path, backup=False
    )  # Renamed call

    # Assertions
    assert result["status"] == "error"
    assert (
        result["action"] == "path_validation_failed"
    )  # Validator should catch non-existence
    assert "Path not found" in result["data"]["message"]
    assert relative_path in result["data"]["message"]


async def test_delete_directory_success(
    file_manager, temp_workspace_files
):  # Renamed test slightly
    """Test successfully deleting an existing directory using delete_path."""  # Updated docstring
    relative_path = "delete_this_folder"  # Renamed for clarity
    absolute_path = temp_workspace_files / relative_path
    absolute_path.mkdir(parents=True, exist_ok=True)
    (absolute_path / "some_file.txt").write_text("data")  # Make it non-empty

    # Action: Attempt to delete the directory with backup=True with await
    result = await file_manager.delete_path(
        filepath=relative_path, backup=True
    )  # Pass backup=True

    # Assertions (Backup fails for directory, so operation fails)
    assert result["status"] == "error"
    assert result["action"] == "delete_failed_backup"
    assert (
        "Failed to create backup for" in result["data"]["message"]
    )  # Expect specific backup failure message from skill
    assert absolute_path.exists()  # Directory should NOT be deleted


async def test_delete_file_permission_error(file_manager, temp_workspace_files, mocker):
    """Test handling PermissionError during file deletion."""
    relative_path = "protected_stuff/cant_delete.dat"
    absolute_path = temp_workspace_files / relative_path

    # Create the file
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_text("important data")

    # Mock Path.unlink to raise PermissionError
    mocker.patch("pathlib.Path.unlink", side_effect=PermissionError("OS says no"))
    # Mock shutil.rmtree as well if directories might be involved
    mocker.patch("shutil.rmtree", side_effect=PermissionError("OS says no to dir"))

    # Action: Attempt to delete the file with backup=True with await
    result = await file_manager.delete_path(
        filepath=relative_path, backup=True
    )  # Pass backup=True

    # Assertions (Backup succeeds, delete fails)
    assert result["status"] == "error"
    assert (
        result["action"] == "delete_failed_permission"
    )  # Error is permission on delete
    assert "Permission denied to delete" in result["data"]["message"]
    assert absolute_path.exists()  # File should still exist


# Use the standard file manager fixture
async def test_delete_file_invalid_path_relative_outside(
    file_manager,
):  # Changed fixture
    """Test failing to delete a file using relative path outside workspace."""
    relative_path = "../dangerous_delete.sh"

    # Action with await
    result = await file_manager.delete_path(
        filepath=relative_path, backup=False
    )  # Renamed call

    # Assertions
    assert result["status"] == "error"
    assert result["action"] == "path_validation_failed"  # Validator
    # Check specific message
    assert (
        f"Access denied: Path '{relative_path}' resolves outside the designated workspace"
        in result["data"]["message"]
    )


async def test_delete_file_invalid_path_absolute(file_manager):  # Changed fixture
    """Test failing to delete a file using an absolute path."""
    absolute_path = "/bin/bash"

    # Action with await
    result = await file_manager.delete_path(
        filepath=absolute_path, backup=False
    )  # Renamed call

    # Assertions
    assert result["status"] == "error"
    assert result["action"] == "path_validation_failed"  # Validator
    # Check specific message
    assert (
        f"Access denied: Path '{absolute_path}' resolves outside the designated workspace"
        in result["data"]["message"]
    )


# --- Test Cases for delete_path ---
# Renamed delete_file -> delete_path
