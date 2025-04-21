import asyncio
import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from typing import Dict, Any, Optional, List

# --- A³X Core Imports (needed for fixtures) ---
try:
    from a3x.core.tool_registry import ToolRegistry
    from a3x.core.context import FragmentContext, SharedTaskContext # Base context class
    from a3x.core.llm_interface import LLMInterface # For type hinting
    from a3x.fragments.base import FragmentDef, BaseFragment # For type hinting
    from a3x.fragments.registry import FragmentRegistry # For type hinting or mocking
    # Import specific fragments if needed for type hints in fixtures, though usually not necessary here
    # Assume PROJECT_ROOT is available via config or determined path
    from a3x.core.config import PROJECT_ROOT
except ImportError as e:
    # If core components can't be imported, skip all tests in this directory
    pytest.skip(f"Skipping emergence tests due to import errors: {e}", allow_module_level=True)

# --- Shared Fixtures for Emergence Tests --- #

@pytest.fixture
def mock_llm_interface() -> MagicMock:
    """Provides a mock LLMInterface."""
    mock = MagicMock(spec=LLMInterface)
    mock.call_llm = AsyncMock(return_value=iter(["mock llm response"]))
    return mock

@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Provides a ToolRegistry instance for each test function."""
    # Use function scope so each test gets a fresh registry before mock_skills populates it
    return ToolRegistry()

@pytest.fixture
def fragment_registry() -> MagicMock:
    """Provides a mock FragmentRegistry."""
    return MagicMock(spec=FragmentRegistry)

@pytest.fixture
def shared_task_context() -> SharedTaskContext:
    """Provides a SharedTaskContext."""
    return SharedTaskContext(
        task_id='emergence_test_001',
        initial_objective="Test isolated fragment behavior."
    )

@pytest.fixture
def workspace_root(tmp_path) -> Path:
    """Provides a temporary workspace root directory and patches PROJECT_ROOT."""
    # Create a realistic subpath if fragments expect it
    (tmp_path / "a3x" / "modules" / "temp").mkdir(parents=True, exist_ok=True)
    
    # Patch PROJECT_ROOT used within fragments if they import it directly
    # Note: Need to patch for each relevant module. Adjust paths as needed.
    patches = [
        patch('a3x.fragments.structure_auto_refactor.PROJECT_ROOT', str(tmp_path), create=True),
        patch('a3x.fragments.mutator.PROJECT_ROOT', str(tmp_path), create=True),
        patch('a3x.fragments.anomaly_detector.PROJECT_ROOT', str(tmp_path), create=True),
        # Add other fragment modules here if they also import PROJECT_ROOT
    ]
    try:
        for p in patches:
            p.start()
        yield tmp_path
    finally:
        for p in patches:
            # Use try-except for stop() in case patch wasn't started (e.g., module import failed)
            try:
                p.stop()
            except RuntimeError: 
                pass # Ignore if patching wasn't successful

@pytest.fixture
def mock_fragment_context(tool_registry) -> MagicMock:
    """Provides a MagicMock simulating FragmentContext."""
    mock_context = MagicMock(spec=FragmentContext)
    mock_context.tool_registry = tool_registry
    
    # --- Mocking message posting via SharedTaskContext --- #
    # Mock the shared context and its add_chat_message method
    mock_shared_context = MagicMock() # No spec
    # Explicitly add the method we need
    mock_shared_context.add_chat_message = MagicMock(name="shared_context.add_chat_message")

    # List to store arguments passed to add_chat_message
    captured_messages: List[Dict[str, Any]] = []

    def side_effect_add_chat_message(*args, **kwargs):
        # Capture the message content (assuming it's the first positional argument)
        if args:
            captured_messages.append(args[0])
        elif 'message' in kwargs:
            captured_messages.append(kwargs['message'])
        # Return a dummy TaskMessage or whatever the original method returns
        return MagicMock() # Or None, or a simulated TaskMessage instance

    mock_shared_context.add_chat_message.side_effect = side_effect_add_chat_message
    mock_shared_context._captured_messages = captured_messages # Attach for easy access

    # Ensure the mock evaluates as True in checks like `if context.shared_task_context:`
    # Since spec is removed, we can freely assign __bool__
    mock_shared_context.__bool__ = lambda: True

    # Assign the mocked shared context to the main context mock
    mock_context.shared_task_context = mock_shared_context

    # --- Mocking Agent ID ---
    mock_context.agent_id = "test_agent"

    # Mock logger
    mock_context.logger = MagicMock()

    return mock_context

@pytest.fixture
def mock_skills(tool_registry): # Depends on tool_registry fixture
    """Creates and registers AsyncMock skills in the provided tool registry."""
    mocks = {
        "generate_module_from_directive": AsyncMock(name="generate_module_from_directive"),
        "write_file": AsyncMock(name="write_file"),
        "execute_python_in_sandbox": AsyncMock(name="execute_python_in_sandbox"),
        "read_file": AsyncMock(name="read_file"),
        "modify_code": AsyncMock(name="modify_code"),
        "learn_from_correction_result": AsyncMock(name="learn_from_correction_result"),
        # Add other skills if needed by fragments under test
    }

    # Define basic schemas (can be simplified for mocks)
    schemas = {
        name: {"name": name, "description": "mock skill", "parameters": {}}
        for name in mocks
    }

    for name, mock_func in mocks.items():
        # Configure default success return values (can be overridden in tests)
        if name == "generate_module_from_directive":
            mock_func.return_value = {"status": "success", "content": "# Default mock code", "path": "default/mock/path.py"}
        elif name == "learn_from_correction_result":
            mock_func.return_value = {"status": "skipped"}
        elif name == "read_file":
             mock_func.return_value = {"status": "success", "data": {"content": "# Default read content"}}
        elif name == "modify_code":
            mock_func.return_value = {"status": "success", "data": {"modified_code": "# Default modified code"}}
        else:
            # Default success for write_file, execute_python_in_sandbox
            mock_func.return_value = {"status": "success", "data": {}, "stdout": "", "stderr": "", "exit_code": 0}
        
        # Register mock with the tool registry provided by the fixture
        tool_registry.register_tool(name=name, instance=None, tool=mock_func, schema=schemas[name])

    logging.getLogger("conftest").debug(f"Registered mock skills: {list(mocks.keys())}")
    # Return the dict of mocks so tests can configure return values/side effects
    return mocks

# --- Test Helper Function --- #

def find_message(messages: List[Dict], msg_type: str, condition: Optional[callable] = None) -> Optional[Dict]:
    """Helper to find the first message of a specific type matching an optional condition."""
    for msg in messages:
        if msg.get("type") == msg_type:
            if condition is None or condition(msg):
                return msg
    return None 