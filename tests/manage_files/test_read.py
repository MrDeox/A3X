import pytest
from unittest.mock import mock_open, MagicMock, patch
from pathlib import Path

# Import the skill function
from a3x.skills.file_manager import FileManagerSkill, MAX_READ_SIZE

# Import the actual WORKSPACE_ROOT
from a3x.core.config import PROJECT_ROOT as WORKSPACE_ROOT  # Use the real one

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
    return mocker.patch.object(Path, "exists")


@pytest.fixture
def mock_path_is_file(mocker):
    return mocker.patch.object(Path, "is_file")


@pytest.fixture
def mock_path_is_dir(mocker):
    return mocker.patch.object(Path, "is_dir")


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
        mock_resolved.stat.return_value = MagicMock(st_size=100)  # Default size
        return mock_resolved

    return mocker.patch.object(Path, "resolve", _mock_resolve)


@pytest.fixture
def mock_open_func(mocker):
    m = mock_open()
    return mocker.patch("builtins.open", m)


# == Read Action Tests (Testing wrapped function) ==


@pytest.mark.asyncio
@patch("builtins.open", new_callable=mock_open)  # Correct syntax
async def test_read_success(mock_open_func, tmp_path):
    """Test successful file reading via wrapped function."""
    filepath = "data/my_file.txt"
    expected_content = "Hello, world!"
    file_manager_instance = FileManagerSkill()

    # --- Mock Configuration ---
    resolved_path_mock = MagicMock(spec=Path, name="resolved_path_mock")
    resolved_path_mock.suffix = ".txt"  # Needs suffix for extension check
    resolved_path_mock.stat.return_value = MagicMock(
        st_size=len(expected_content)
    )  # Mock stat for size check

    # Configure builtins.open mock
    mock_open_func.return_value.read.return_value = expected_content

    # --- Execution ---
    result = await FileManagerSkill.read_file.__wrapped__(
        file_manager_instance,  # self
        resolved_path=resolved_path_mock,
        original_path_str=filepath,
        path=filepath,
    )

    # --- Assertions ---
    assert result["status"] == "success"
    assert result["action"] == "file_read"
    assert result["data"]["filepath"] == filepath
    assert result["data"]["content"] == expected_content
    assert "read successfully" in result["data"]["message"]

    # Verify mock calls
    resolved_path_mock.stat.assert_called_once()
    mock_open_func.assert_called_once_with(resolved_path_mock, "r", encoding="utf-8")
    mock_open_func().read.assert_called_once()


@pytest.mark.asyncio
async def test_read_unsupported_extension(tmp_path):
    """Test reading file with unsupported extension via wrapped function."""
    filepath = "archive.zip"
    file_manager_instance = FileManagerSkill()

    resolved_path_mock = MagicMock(spec=Path, name="resolved_path_mock")
    resolved_path_mock.suffix = ".zip"  # Unsupported extension

    result = await FileManagerSkill.read_file.__wrapped__(
        file_manager_instance,
        resolved_path=resolved_path_mock,
        original_path_str=filepath,
        path=filepath,
    )

    assert result["status"] == "error"
    assert result["action"] == "read_file_failed_unsupported_ext"
    assert "Extension '.zip' not supported" in result["data"]["message"]
    resolved_path_mock.stat.assert_not_called()  # Should fail before stat


@pytest.mark.asyncio
async def test_read_file_too_large(tmp_path):
    """Test reading file larger than MAX_READ_SIZE via wrapped function."""
    filepath = "large_log.txt"
    file_manager_instance = FileManagerSkill()

    resolved_path_mock = MagicMock(spec=Path, name="resolved_path_mock")
    resolved_path_mock.suffix = ".txt"  # Supported extension
    resolved_path_mock.stat.return_value = MagicMock(
        st_size=MAX_READ_SIZE + 1
    )  # Mock size check

    result = await FileManagerSkill.read_file.__wrapped__(
        file_manager_instance,
        resolved_path=resolved_path_mock,
        original_path_str=filepath,
        path=filepath,
    )

    assert result["status"] == "error"
    assert result["action"] == "read_file_failed_too_large"
    assert "File too large" in result["data"]["message"]
    resolved_path_mock.stat.assert_called_once()  # Stat is called


@pytest.mark.asyncio
@patch("builtins.open", new_callable=mock_open)  # Correct syntax
async def test_read_permission_error(mock_open_func, tmp_path):
    """Test PermissionError during file read via wrapped function."""
    filepath = "restricted.txt"
    file_manager_instance = FileManagerSkill()

    resolved_path_mock = MagicMock(spec=Path, name="resolved_path_mock")
    resolved_path_mock.suffix = ".txt"  # Supported
    resolved_path_mock.stat.return_value = MagicMock(st_size=50)  # Size is okay

    # Configure builtins.open to raise PermissionError
    mock_open_func.side_effect = PermissionError("Permission denied by OS")

    result = await FileManagerSkill.read_file.__wrapped__(
        file_manager_instance,
        resolved_path=resolved_path_mock,
        original_path_str=filepath,
        path=filepath,
    )

    assert result["status"] == "error"
    assert result["action"] == "read_file_failed"
    assert "Permission denied" in result["data"]["message"]

    # Verify calls up to the point of error
    resolved_path_mock.stat.assert_called_once()
    mock_open_func.assert_called_once_with(resolved_path_mock, "r", encoding="utf-8")


# Removed tests that only tested decorator validation logic.
