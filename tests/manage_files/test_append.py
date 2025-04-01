import pytest
import asyncio
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the skill function and WORKSPACE_ROOT
from skills.manage_files import append_to_file, WORKSPACE_ROOT

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

# Mock Path object for controlled testing
@pytest.fixture
def mock_path_object(tmp_path):
    # Creates a mock that mimics Path behavior for existence, type checks etc.
    # Simplistic version: assumes path exists and is a file after validation
    mock_p = MagicMock(spec=Path)
    mock_p.exists.return_value = True
    mock_p.is_file.return_value = True
    mock_p.is_dir.return_value = False
    # Setup parent mock if needed
    mock_p.parent = MagicMock(spec=Path)
    mock_p.parent.mkdir.return_value = None # Mock mkdir
    # Use a real temporary file path for the open operation
    real_file_path = tmp_path / "test_append.txt"
    real_file_path.touch() # Ensure the file exists for append
    mock_p.__str__.return_value = str(real_file_path)

    # Configure the mock to behave correctly with open
    # We'll patch 'builtins.open' within the test using this path
    mock_p._real_path = real_file_path

    return mock_p

@pytest.mark.asyncio
@patch('skills.manage_files.validate_workspace_path') # Mock the decorator
async def test_append_to_file_success(mock_validate, tmp_path, mock_path_object):
    """Test successfully appending content to a file."""
    original_path = "test_dir/output.txt"
    content_to_append = "New line of text.\n"
    resolved_path_mock = mock_path_object

    # Configure the mock decorator to directly call the function with mocked paths
    def decorator_side_effect(*args, **kwargs):
        # This function simulates the decorator calling the decorated function
        # It needs to get the original function from the args or context
        # For simplicity, assume the decorated function is the first arg if it's callable
        original_func = args[0] if callable(args[0]) else None
        if original_func:
             # Call the original function with the necessary injected args
             func_kwargs = kwargs.copy()
             func_kwargs['resolved_path'] = resolved_path_mock
             func_kwargs['original_path_str'] = original_path
             return original_func(**func_kwargs)
        else:
             # Fallback or raise error if function not found
             raise ValueError("Could not find original function in mock decorator")
    mock_validate.side_effect = decorator_side_effect

    # Patch open to use the real temp file associated with the mock path
    with patch('builtins.open', new_callable=MagicMock) as mock_open:
         # Call the skill function - decorator is mocked
         # The actual parameters passed here are those defined in @skill decorator
         result = append_to_file(content=content_to_append)

    assert result["status"] == "success"
    assert result["action"] == "file_appended"
    assert original_path in result["data"]["message"]
    assert result["data"]["filepath"] == original_path

    # Verify open was called correctly
    mock_open.assert_called_once_with(resolved_path_mock._real_path, "a", encoding="utf-8")
    # Verify write was called with the content
    mock_open().__enter__().write.assert_called_once_with(content_to_append)

    # Check the actual file content
    assert resolved_path_mock._real_path.read_text() == content_to_append

@pytest.mark.asyncio
@patch('skills.manage_files.validate_workspace_path')
async def test_append_to_file_no_newline(mock_validate, tmp_path, mock_path_object):
    """Test appending content without a trailing newline."""
    original_path = "another/file.log"
    content_to_append = "Log entry without newline"
    expected_content = content_to_append + "\n" # Logic adds a newline
    resolved_path_mock = mock_path_object

    # Configure mock decorator (same as above)
    def decorator_side_effect(*args, **kwargs):
        original_func = args[0] if callable(args[0]) else None
        if original_func:
             func_kwargs = kwargs.copy()
             func_kwargs['resolved_path'] = resolved_path_mock
             func_kwargs['original_path_str'] = original_path
             return original_func(**func_kwargs)
        else: raise ValueError("Could not find original function in mock decorator")
    mock_validate.side_effect = decorator_side_effect

    with patch('builtins.open', new_callable=MagicMock) as mock_open:
        result = append_to_file(content=content_to_append)

    assert result["status"] == "success"
    mock_open.assert_called_once_with(resolved_path_mock._real_path, "a", encoding="utf-8")
    mock_open().__enter__().write.assert_called_once_with(expected_content)
    assert resolved_path_mock._real_path.read_text() == expected_content

@pytest.mark.asyncio
@patch('skills.manage_files.validate_workspace_path')
async def test_append_to_file_permission_error(mock_validate, tmp_path, mock_path_object):
    """Test handling of PermissionError during file append."""
    original_path = "locked/file.dat"
    content_to_append = "data"
    resolved_path_mock = mock_path_object

    # Configure mock decorator
    def decorator_side_effect(*args, **kwargs):
        original_func = args[0] if callable(args[0]) else None
        if original_func:
            func_kwargs = kwargs.copy()
            func_kwargs['resolved_path'] = resolved_path_mock
            func_kwargs['original_path_str'] = original_path
            return original_func(**func_kwargs)
        else: raise ValueError("Could not find original function in mock decorator")
    mock_validate.side_effect = decorator_side_effect

    # Patch open to raise PermissionError
    with patch('builtins.open', side_effect=PermissionError("Permission denied")) as mock_open:
        result = append_to_file(content=content_to_append)

    assert result["status"] == "error"
    assert result["action"] == "append_failed"
    assert "Permission denied" in result["data"]["message"]
    assert original_path in result["data"]["message"]

# Add test for IsADirectoryError if necessary, although decorator should prevent
# Add test for generic Exception if necessary
