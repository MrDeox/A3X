import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path to allow importing a3x modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from a3x.skills.system.validate_fragment import skill_validate_fragment, ValidateFragmentParams
from a3x.fragments.base import BaseFragment

# Test data directory
TEST_DATA_DIR = os.path.join(project_root, "tests", "data", "temp", "validate_fragment_tests")

@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """Create necessary test files and directories."""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    # Create a valid fragment file
    valid_fragment_path = os.path.join(TEST_DATA_DIR, "valid_fragment.py")
    with open(valid_fragment_path, "w") as f:
        f.write("""
from a3x.fragments.base import BaseFragment
from typing import Any

class ValidTestFragment(BaseFragment):
    async def execute(self, ctx: Any):
        return {"status": "success", "message": "Executed valid fragment"}
""")

    # Create a fragment file with syntax error
    syntax_error_fragment_path = os.path.join(TEST_DATA_DIR, "syntax_error_fragment.py")
    with open(syntax_error_fragment_path, "w") as f:
        f.write("""
from a3x.fragments.base import BaseFragment

class SyntaxErrorFragment(BaseFragment)
    def execute(self, ctx):
        pass
""") # Missing colon

    # Create a fragment file with no fragment class
    no_class_fragment_path = os.path.join(TEST_DATA_DIR, "no_class_fragment.py")
    with open(no_class_fragment_path, "w") as f:
        f.write("def some_function(): pass")

    # Create a fragment file with multiple fragment classes
    multiple_classes_fragment_path = os.path.join(TEST_DATA_DIR, "multiple_classes_fragment.py")
    with open(multiple_classes_fragment_path, "w") as f:
        f.write("""
from a3x.fragments.base import BaseFragment

class FirstFragment(BaseFragment):
    def execute(self, ctx):
        pass

class SecondFragment(BaseFragment):
    def execute(self, ctx):
        pass
""")

    # Create a fragment file with class not inheriting from BaseFragment
    wrong_base_fragment_path = os.path.join(TEST_DATA_DIR, "wrong_base_fragment.py")
    with open(wrong_base_fragment_path, "w") as f:
        f.write("""
class NotAFragment:
    def execute(self, ctx):
        pass
""")

    # Create a fragment file with no execute method
    no_execute_fragment_path = os.path.join(TEST_DATA_DIR, "no_execute_fragment.py")
    with open(no_execute_fragment_path, "w") as f:
        f.write("""
from a3x.fragments.base import BaseFragment

class NoExecuteFragment(BaseFragment):
    pass
""")

    yield # Run tests

    # Teardown: Remove test files and directory (optional, depending on policy)
    # import shutil
    # shutil.rmtree(TEST_DATA_DIR)

import asyncio

@pytest.mark.asyncio
async def test_validate_fragment_success():
    """Test successful validation of a correctly formatted fragment."""
    params = ValidateFragmentParams(fragment_path=os.path.join(TEST_DATA_DIR, "valid_fragment.py"))
    mock_context = MagicMock()
    mock_context.workspace_root = TEST_DATA_DIR
    result = await skill_validate_fragment(mock_context, {"fragment_path": params.fragment_path})
    assert result["status"] == "success"
    assert "Validation successful" in result["message"]

@pytest.mark.asyncio
async def test_validate_fragment_file_not_found():
    """Test validation failure when the fragment file does not exist."""
    params = ValidateFragmentParams(fragment_path=os.path.join(TEST_DATA_DIR, "non_existent_fragment.py"))
    mock_context = MagicMock()
    mock_context.workspace_root = TEST_DATA_DIR
    result = await skill_validate_fragment(mock_context, {"fragment_path": params.fragment_path})
    assert result["status"] == "error"
    assert "not found" in result["message"]

@pytest.mark.asyncio
async def test_validate_fragment_syntax_error():
    """Test validation failure due to syntax error in the fragment file."""
    params = ValidateFragmentParams(fragment_path=os.path.join(TEST_DATA_DIR, "syntax_error_fragment.py"))
    mock_context = MagicMock()
    mock_context.workspace_root = TEST_DATA_DIR
    result = await skill_validate_fragment(mock_context, {"fragment_path": params.fragment_path})
    assert result["status"] == "error"
    assert "syntax error" in result["message"].lower()

@pytest.mark.asyncio
async def test_validate_fragment_import_error():
    """Test validation failure due to other import errors (e.g., missing dependency)."""
    params = ValidateFragmentParams(fragment_path=os.path.join(TEST_DATA_DIR, "valid_fragment.py"))
    mock_context = MagicMock()
    mock_context.workspace_root = TEST_DATA_DIR
    # Patch spec.loader.exec_module to raise ImportError
    import importlib.util
    real_spec_from_file_location = importlib.util.spec_from_file_location
    def fake_spec_from_file_location(name, path):
        spec = real_spec_from_file_location(name, path)
        class FakeLoader:
            def create_module(self, spec):
                return None
            def exec_module(self, module):
                raise ImportError("Simulated import error")
        spec.loader = FakeLoader()
        return spec
    with patch('importlib.util.spec_from_file_location', side_effect=fake_spec_from_file_location):
        result = await skill_validate_fragment(mock_context, {"fragment_path": params.fragment_path})
        assert result["status"] == "error"
        assert "import" in result["message"].lower()

@pytest.mark.asyncio
async def test_validate_fragment_no_class():
    """Test validation failure when no class is found in the file."""
    params = ValidateFragmentParams(fragment_path=os.path.join(TEST_DATA_DIR, "no_class_fragment.py"))
    mock_context = MagicMock()
    mock_context.workspace_root = TEST_DATA_DIR
    result = await skill_validate_fragment(mock_context, {"fragment_path": params.fragment_path})
    assert result["status"] == "error"
    assert "no class" in result["message"].lower()

@pytest.mark.asyncio
async def test_validate_fragment_no_base_fragment_class():
    """Test validation failure when no class inherits from BaseFragment."""
    params = ValidateFragmentParams(fragment_path=os.path.join(TEST_DATA_DIR, "wrong_base_fragment.py"))
    mock_context = MagicMock()
    mock_context.workspace_root = TEST_DATA_DIR
    result = await skill_validate_fragment(mock_context, {"fragment_path": params.fragment_path})
    assert result["status"] == "error"
    assert "no class inheriting" in result["message"].lower()

@pytest.mark.asyncio
async def test_validate_fragment_multiple_classes():
    """Test validation failure when multiple classes inherit from BaseFragment."""
    params = ValidateFragmentParams(fragment_path=os.path.join(TEST_DATA_DIR, "multiple_classes_fragment.py"))
    mock_context = MagicMock()
    mock_context.workspace_root = TEST_DATA_DIR
    result = await skill_validate_fragment(mock_context, {"fragment_path": params.fragment_path})
    assert result["status"] == "error"
    assert "multiple" in result["message"].lower()

@pytest.mark.asyncio
async def test_validate_fragment_no_execute_method():
    """Test validation failure when the fragment class lacks an execute method."""
    params = ValidateFragmentParams(fragment_path=os.path.join(TEST_DATA_DIR, "no_execute_fragment.py"))
    mock_context = MagicMock()
    mock_context.workspace_root = TEST_DATA_DIR
    result = await skill_validate_fragment(mock_context, {"fragment_path": params.fragment_path})
    assert result["status"] == "error"
    assert "execute" in result["message"].lower()

# Optional: Add a test for the path conversion logic if needed
# def test_path_conversion():
#    ...

# Add more tests as needed, e.g., edge cases for paths 