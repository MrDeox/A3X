import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

# Import the skill function and WORKSPACE_ROOT
from skills.manage_files import create_file

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
    return mocker.patch("pathlib.Path.exists")


@pytest.fixture
def mock_path_is_file(mocker):
    return mocker.patch("pathlib.Path.is_file")


@pytest.fixture
def mock_path_is_dir(mocker):
    return mocker.patch("pathlib.Path.is_dir")


@pytest.fixture
def mock_path_mkdir(mocker):
    return mocker.patch("pathlib.Path.mkdir")


@pytest.fixture
def mock_open_func(mocker):
    return mocker.patch("builtins.open", mock_open())


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

    mocker.patch(
        "skills.manage_files._is_path_within_workspace", side_effect=_mock_check
    )


# Mock Path object for controlled testing
@pytest.fixture
def mock_path_object(tmp_path):
    # Creates a mock that mimics Path behavior
    mock_p = MagicMock(spec=Path)
    mock_p.exists.return_value = False  # Default: file doesn't exist
    mock_p.is_file.return_value = False
    mock_p.is_dir.return_value = False
    mock_p.parent = MagicMock(spec=Path)
    mock_p.parent.mkdir.return_value = None
    # Use a real temporary file path for the open operation
    real_file_path = tmp_path / "test_create.txt"
    mock_p.__str__.return_value = str(real_file_path)
    mock_p._real_path = real_file_path
    return mock_p


@pytest.mark.asyncio
@patch("skills.manage_files.validate_workspace_path")  # Mock the decorator
async def test_create_file_success(mock_validate, tmp_path, mock_path_object):
    """Test successfully creating a new file."""
    original_path = "new_folder/new_file.txt"
    content_to_write = "This is the first line.\n"
    resolved_path_mock = mock_path_object
    resolved_path_mock.exists.return_value = (
        False  # Ensure exists is False for creation
    )

    # Configure the mock decorator
    def decorator_side_effect(*args, **kwargs):
        original_func = args[0] if callable(args[0]) else None
        if original_func:
            func_kwargs = kwargs.copy()
            # Remove the decorator arg name if present, pass the actual args
            func_kwargs.pop("filepath", None)
            func_kwargs["resolved_path"] = resolved_path_mock
            func_kwargs["original_path_str"] = original_path
            return original_func(**func_kwargs)
        else:
            raise ValueError("Could not find original function")

    mock_validate.side_effect = decorator_side_effect

    # Patch open to use the real temp file
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        # Call the skill function directly with its expected args
        result = create_file(
            content=content_to_write,
            overwrite=False,
            resolved_path=resolved_path_mock,
            original_path_str=original_path,
        )

    assert result["status"] == "success"
    assert result["action"] == "file_created"
    assert original_path in result["data"]["message"]
    assert result["data"]["filepath"] == original_path

    # Verify open was called correctly for writing
    mock_open.assert_called_once_with(
        str(resolved_path_mock._real_path), "w", encoding="utf-8"
    )
    # Verify write was called with the content
    mock_open().__enter__().write.assert_called_once_with(content_to_write)
    # Verify parent directory creation was attempted
    resolved_path_mock.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    # Check that the real file was written to
    assert resolved_path_mock._real_path.exists()
    assert resolved_path_mock._real_path.read_text() == content_to_write


@pytest.mark.asyncio
@patch("skills.manage_files.validate_workspace_path")
async def test_create_file_overwrite_success(mock_validate, tmp_path, mock_path_object):
    """Test successfully overwriting an existing file."""
    original_path = "existing_file.md"
    content_to_write = "# New Markdown Content\n"
    resolved_path_mock = mock_path_object
    resolved_path_mock.exists.return_value = True  # File exists
    resolved_path_mock.is_file.return_value = True  # It is a file
    # Simulate the file having old content
    resolved_path_mock._real_path.write_text("Old content")

    # Configure mock decorator
    def decorator_side_effect(*args, **kwargs):
        original_func = args[0] if callable(args[0]) else None
        if original_func:
            func_kwargs = kwargs.copy()
            func_kwargs.pop("filepath", None)
            func_kwargs["resolved_path"] = resolved_path_mock
            func_kwargs["original_path_str"] = original_path
            return original_func(**func_kwargs)
        else:
            raise ValueError("Could not find original function")

    mock_validate.side_effect = decorator_side_effect

    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        # Call the skill function directly
        result = create_file(
            content=content_to_write,
            overwrite=True,
            resolved_path=resolved_path_mock,
            original_path_str=original_path,
        )

    assert result["status"] == "success"
    assert result["action"] == "file_overwritten"  # Action indicates overwrite
    assert original_path in result["data"]["message"]
    mock_open.assert_called_once_with(
        str(resolved_path_mock._real_path), "w", encoding="utf-8"
    )
    mock_open().__enter__().write.assert_called_once_with(content_to_write)
    assert (
        resolved_path_mock._real_path.read_text() == content_to_write
    )  # Verify new content


@pytest.mark.asyncio
@patch("skills.manage_files.validate_workspace_path")
async def test_create_file_exists_no_overwrite(
    mock_validate, tmp_path, mock_path_object
):
    """Test failing to create a file when it exists and overwrite is False."""
    original_path = "dont_overwrite.txt"
    content_to_write = "some data"
    resolved_path_mock = mock_path_object
    resolved_path_mock.exists.return_value = True  # File exists
    resolved_path_mock.is_file.return_value = True

    # Configure mock decorator
    def decorator_side_effect(*args, **kwargs):
        original_func = args[0] if callable(args[0]) else None
        if original_func:
            func_kwargs = kwargs.copy()
            func_kwargs.pop("filepath", None)
            func_kwargs["resolved_path"] = resolved_path_mock
            func_kwargs["original_path_str"] = original_path
            return original_func(**func_kwargs)
        else:
            raise ValueError("Could not find original function")

    mock_validate.side_effect = decorator_side_effect

    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        # Call the skill function directly
        result = create_file(
            content=content_to_write,
            overwrite=False,
            resolved_path=resolved_path_mock,
            original_path_str=original_path,
        )

    assert result["status"] == "error"
    assert result["action"] == "create_file_failed"
    assert "already exists" in result["data"]["message"]
    assert "overwrite=True" in result["data"]["message"]
    mock_open.assert_not_called()  # open should not be called


@pytest.mark.asyncio
@patch("skills.manage_files.validate_workspace_path")
async def test_create_file_target_is_directory(
    mock_validate, tmp_path, mock_path_object
):
    """Test failing to create a file where a directory exists."""
    original_path = "existing_dir"  # Path exists as a directory
    content_to_write = "some data"
    resolved_path_mock = mock_path_object
    resolved_path_mock.exists.return_value = True
    resolved_path_mock.is_dir.return_value = True  # It IS a directory
    resolved_path_mock.is_file.return_value = False

    # Configure mock decorator
    def decorator_side_effect(*args, **kwargs):
        original_func = args[0] if callable(args[0]) else None
        if original_func:
            func_kwargs = kwargs.copy()
            func_kwargs.pop("filepath", None)
            func_kwargs["resolved_path"] = resolved_path_mock
            func_kwargs["original_path_str"] = original_path
            return original_func(**func_kwargs)
        else:
            raise ValueError("Could not find original function")

    mock_validate.side_effect = decorator_side_effect

    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        # Call the skill function directly
        result = create_file(
            content=content_to_write,
            overwrite=False,
            resolved_path=resolved_path_mock,
            original_path_str=original_path,
        )

    assert result["status"] == "error"
    assert result["action"] == "create_file_failed"
    assert "directory already exists" in result["data"]["message"]
    mock_open.assert_not_called()


@pytest.mark.asyncio
@patch("skills.manage_files.validate_workspace_path")
async def test_create_file_permission_error(mock_validate, tmp_path, mock_path_object):
    """Test handling PermissionError during file creation."""
    original_path = "protected/new_file.txt"
    content_to_write = "data"
    resolved_path_mock = mock_path_object
    resolved_path_mock.exists.return_value = False

    # Configure mock decorator
    def decorator_side_effect(*args, **kwargs):
        original_func = args[0] if callable(args[0]) else None
        if original_func:
            func_kwargs = kwargs.copy()
            func_kwargs.pop("filepath", None)
            func_kwargs["resolved_path"] = resolved_path_mock
            func_kwargs["original_path_str"] = original_path
            return original_func(**func_kwargs)
        else:
            raise ValueError("Could not find original function")

    mock_validate.side_effect = decorator_side_effect

    # Patch open to raise PermissionError
    with patch(
        "builtins.open", side_effect=PermissionError("Access denied")
    ): # as mock_open: # F841
        # Call the skill function directly
        result = create_file(
            content=content_to_write,
            overwrite=False,
            resolved_path=resolved_path_mock,
            original_path_str=original_path,
        )

    assert result["status"] == "error"
    assert result["action"] == "create_file_failed"
    assert "Permission denied" in result["data"]["message"]
    assert original_path in result["data"]["message"]
