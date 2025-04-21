import asyncio
import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- A³X Core Imports ---
try:
    from a3x.core.tool_registry import ToolRegistry
    from a3x.core.context import FragmentContext, SharedTaskContext # Using FragmentContext directly here
    from a3x.core.llm_interface import LLMInterface # For type hinting
    from a3x.fragments.base import FragmentDef
    from a3x.fragments.registry import FragmentRegistry # For type hinting or mocking
    # Import Fragment to test
    from a3x.fragments.structure_auto_refactor import StructureAutoRefactorFragment
    # Assume PROJECT_ROOT is available via config or determined path
    from a3x.core.config import PROJECT_ROOT
except ImportError as e:
    logger.error(f"Failed to import A³X components: {e}. Ensure PYTHONPATH is set or run from root.")
    pytest.skip("Skipping tests due to import errors.", allow_module_level=True)

# --- Test Context Fixture ---
# Using FragmentContext directly and mocking its methods/attributes as needed
# OR define a simpler test-specific context if preferred.
# Let's try using a mocked FragmentContext for now.

@pytest.fixture
def mock_fragment_context(tmp_path) -> MagicMock:
    """Provides a mocked FragmentContext with essential attributes/methods."""
    # Create mocks for dependencies of FragmentContext if they are accessed
    mock_logger = logging.getLogger("MockFragmentContext")
    mock_llm = MagicMock(spec=LLMInterface)
    mock_tool_registry = ToolRegistry() # Use a real one to register mocks
    mock_fragment_registry = MagicMock(spec=FragmentRegistry)
    mock_shared_context = SharedTaskContext(task_id='frag_test_01', initial_objective='Test fragment')
    # tmp_path is provided by pytest
    mock_workspace_root = tmp_path
    mock_memory_manager = MagicMock() # Mock if used

    # Create the mock FragmentContext instance
    mock_context = MagicMock(spec=FragmentContext)

    # Assign mocked attributes
    mock_context.logger = mock_logger
    mock_context.llm_interface = mock_llm
    mock_context.tool_registry = mock_tool_registry # Store the registry for skill mocks
    mock_context.fragment_registry = mock_fragment_registry
    mock_context.shared_task_context = mock_shared_context
    mock_context.workspace_root = mock_workspace_root
    mock_context.memory_manager = mock_memory_manager

    # Mock the post_chat_message method
    mock_context.post_chat_message = AsyncMock(name="post_chat_message")

    # Store messages posted via the mock
    mock_context.posted_messages = []
    async def capture_message(*args, **kwargs):
        # Extract relevant parts from args/kwargs based on actual post_chat_message signature
        # Assuming positional: context, message_type, content, sender, target_fragment
        # Or keyword based: message_type=, content=, sender=, target_fragment=
        message_content = kwargs.get('content', args[2] if len(args) > 2 else None)
        message_type = kwargs.get('message_type', args[1] if len(args) > 1 else 'unknown')
        sender = kwargs.get('sender', 'unknown_sender')
        target = kwargs.get('target_fragment', None)
        mock_context.posted_messages.append({
            "type": message_type,
            "sender": sender,
            "content": message_content,
            "target": target
        })
        mock_logger.debug(f"Mock context captured message: Type='{message_type}', Sender='{sender}'")

    # Make the mock method call our capture function
    # Note: post_chat_message is called like: await self.post_chat_message(context=context, ...)
    # The fragment passes the *real* context instance to its own method.
    # So we might need to patch BaseFragment.post_chat_message instead, or mock the context passed.
    # Let's stick with mocking the context method for now.
    # mock_context.post_chat_message.side_effect = capture_message
    # UPDATE: Since the fragment calls `await self.post_chat_message(...)`, we need to mock THAT method
    # on the fragment instance itself later, or pass this mock_context correctly.
    # For isolated fragment tests, we'll call fragment methods directly, passing this mock context.

    # Patch PROJECT_ROOT used within the fragment's scope for this test
    with patch('a3x.fragments.structure_auto_refactor.PROJECT_ROOT', str(mock_workspace_root)):
        yield mock_context # Yield the configured mock

# --- Mock Skills Fixture ---
@pytest.fixture
def mock_skills(mock_fragment_context): # Depends on the context fixture
    """Creates and registers mock skills in the context's tool registry."""
    tool_registry = mock_fragment_context.tool_registry # Get the registry from the mock context
    mocks = {
        "generate_module_from_directive": AsyncMock(name="generate_module_from_directive"),
        "write_file": AsyncMock(name="write_file"),
        "execute_python_in_sandbox": AsyncMock(name="execute_python_in_sandbox"),
        "read_file": AsyncMock(name="read_file"),
        "modify_code": AsyncMock(name="modify_code"),
        "learn_from_correction_result": AsyncMock(name="learn_from_correction_result"),
    }

    # Define basic schemas for registration
    schemas = {
        "generate_module_from_directive": {"name": "generate_module_from_directive", "description": "mock", "parameters": {}},
        "write_file": {"name": "write_file", "description": "mock", "parameters": {}},
        "execute_python_in_sandbox": {"name": "execute_python_in_sandbox", "description": "mock", "parameters": {}},
        "read_file": {"name": "read_file", "description": "mock", "parameters": {}},
        "modify_code": {"name": "modify_code", "description": "mock", "parameters": {}},
        "learn_from_correction_result": {"name": "learn_from_correction_result", "description": "mock", "parameters": {}},
    }

    for name, mock_func in mocks.items():
        # Configure default success return values initially
        if name == "generate_module_from_directive":
            mock_func.return_value = {"status": "success", "content": "# Mock generated code", "path": "a3x/modules/temp/generated_mock.py"}
        elif name == "learn_from_correction_result":
            mock_func.return_value = {"status": "skipped", "data": {"message": "Learning not implemented in mock"}}
        else:
            mock_func.return_value = {"status": "success", "data": {}, "stdout": "", "stderr": "", "exit_code": 0}

        # Register mock with the tool registry inside the mock context
        tool_registry.register_tool(name=name, instance=None, tool=mock_func, schema=schemas[name])

    logger.debug(f"Registered mock skills in mock context: {list(mocks.keys())}")
    return mocks # Return the dict of mocks for configuration in tests

# --- Fragment Fixture ---
@pytest.fixture
def structure_fragment(mock_fragment_context) -> StructureAutoRefactorFragment:
    """Provides an instance of StructureAutoRefactorFragment with mocked tool registry."""
    tool_registry = mock_fragment_context.tool_registry # Get the registry with mocks
    metadata = {"name": "StructureAutoRefactor", "description": "Test", "category": "Test",
                "skills": list(tool_registry.list_tools().keys())}
    frag_def = FragmentDef(name=metadata["name"], description=metadata["description"],
                           category=metadata["category"], skills=metadata["skills"],
                           fragment_class=StructureAutoRefactorFragment)

    fragment = StructureAutoRefactorFragment(fragment_def=frag_def, tool_registry=tool_registry)
    # Ensure logger level is appropriate
    fragment._logger.setLevel(logging.DEBUG)
    # Patch the fragment's post_chat_message to use the mock context's capture list
    # This is crucial because the fragment calls self.post_chat_message
    fragment.post_chat_message = mock_fragment_context.post_chat_message

    return fragment

# --- Test Helper --- #
def find_message(messages: List[Dict], msg_type: str) -> Optional[Dict]:
    """Helper to find the first message of a specific type in a list."""
    for msg in messages:
        if msg.get("type") == msg_type:
            return msg
    return None 

# --- Test Cases ---

@pytest.mark.asyncio
async def test_handle_create_module_directive_success(
    structure_fragment: StructureAutoRefactorFragment,
    mock_fragment_context: MagicMock,
    mock_skills: Dict[str, AsyncMock]
):
    """Test successful handling of a create_helper_module directive."""
    target_path_str = "a3x/modules/temp/new_helper.py"
    directive_message = "Create a simple functional helper module."
    initial_directive = {
        "type": "directive", # This is the *content* of the architecture_suggestion
        "action": "create_helper_module",
        "target": target_path_str,
        "message": directive_message
    }

    # Configure mock skills (default setup in fixture is success, but be explicit if needed)
    generated_code = "def helper_func():\n    print('Success!')\nhelper_func()"
    # Ensure generate returns the correct target path relative to mock workspace
    abs_target_path = mock_fragment_context.workspace_root / target_path_str
    mock_skills["generate_module_from_directive"].return_value = {
        "status": "success",
        "content": generated_code,
        "path": str(abs_target_path) # Skill should return resolved path
    }
    mock_skills["write_file"].return_value = {"status": "success", "data": {}}
    mock_skills["execute_python_in_sandbox"].return_value = {"status": "success", "exit_code": 0, "stdout": "Success!", "stderr": ""}

    # --- Act ---
    # Call the handler directly, passing the directive content and the mock context
    await structure_fragment.handle_directive(initial_directive, mock_fragment_context)

    # --- Assert ---
    # 1. Check skill calls
    mock_skills["generate_module_from_directive"].assert_called_once_with(
        "generate_module_from_directive", # Tool name
        {"message": directive_message, "target_path": str(abs_target_path)} # Args
    )
    # The fragment converts the absolute path from generate back to relative for write/execute
    expected_relative_path = target_path_str
    mock_skills["write_file"].assert_called_once_with(
        "write_file",
        {"file_path": expected_relative_path, "content": generated_code}
    )
    mock_skills["execute_python_in_sandbox"].assert_called_once_with(
        "execute_python_in_sandbox",
        {"script_path": expected_relative_path}
    )

    # 2. Check posted messages
    posted = mock_fragment_context.posted_messages
    assert len(posted) == 1, f"Expected 1 message, found {len(posted)}"

    result_msg = find_message(posted, "refactor_result")
    assert result_msg is not None, "refactor_result message was not posted"
    assert result_msg["content"]["status"] == "success"
    assert result_msg["content"]["original_action"] == "create_helper_module"
    assert result_msg["content"]["target"] == target_path_str
    assert "Successfully generated, created, and sandbox-tested" in result_msg["content"]["summary"]

    # Check details (optional, but good practice)
    details = json.loads(result_msg["content"]["details"])
    assert details["generation_status"] == "success"
    assert details["write_status"] == "success"
    assert details["sandbox_status"] == "success"
    assert details["sandbox_exit_code"] == 0

@pytest.mark.asyncio
async def test_handle_refactor_directive_correction_success(
    structure_fragment: StructureAutoRefactorFragment,
    mock_fragment_context: MagicMock,
    mock_skills: Dict[str, AsyncMock]
):
    """Test successful handling of a refactor_module directive for correction."""
    target_path_str = "a3x/modules/temp/failing_helper.py"
    initial_error = "Traceback...\nNameError: name 'undefined_variable' is not defined"
    # Simulate the correction directive sent after a failure
    correction_directive = {
        "type": "directive",
        "action": "refactor_module",
        "target": target_path_str,
        "message": f"The previously generated code in '{target_path_str}' failed during sandbox execution. Please correct.\nError Output (stderr):\n---\n{initial_error}\n---"
    }

    # Configure mock skills for the correction flow
    faulty_code = "def fail_func():\n    print(undefined_variable)\nfail_func()"
    corrected_code = "def fail_func():\n    defined_variable = 1\n    print(defined_variable)\nfail_func()"
    
    mock_skills["read_file"].return_value = {"status": "success", "data": {"content": faulty_code}}
    mock_skills["modify_code"].return_value = {"status": "success", "data": {"modified_code": corrected_code}}
    mock_skills["write_file"].return_value = {"status": "success", "data": {}}
    # Sandbox should succeed after correction
    mock_skills["execute_python_in_sandbox"].return_value = {"status": "success", "exit_code": 0, "stdout": "1", "stderr": ""}
    # Learning skill can be skipped or success
    mock_skills["learn_from_correction_result"].return_value = {"status": "skipped", "data": {"message": "Learning mock skipped"}}

    # --- Act ---
    await structure_fragment.handle_directive(correction_directive, mock_fragment_context)

    # --- Assert ---
    # 1. Check skill calls (read, modify, write, sandbox, learn)
    mock_skills["read_file"].assert_called_once_with("read_file", {"path": target_path_str})
    # Check that modify_code received the error message and original code
    modify_call_args = mock_skills["modify_code"].call_args[0][1] # Get the dict arg
    assert initial_error in modify_call_args["modification"]
    assert faulty_code in modify_call_args["modification"]
    assert modify_call_args["code_to_modify"] == faulty_code
    mock_skills["write_file"].assert_called_once_with("write_file", {"file_path": target_path_str, "content": corrected_code, "overwrite": True})
    mock_skills["execute_python_in_sandbox"].assert_called_once_with("execute_python_in_sandbox", {"script_path": target_path_str})
    mock_skills["learn_from_correction_result"].assert_called_once()
    # Check learn skill args (optional but good)
    learn_call_args = mock_skills["learn_from_correction_result"].call_args[0][1]
    assert learn_call_args["stderr"] == correction_directive["message"] # Should pass the original error message that triggered the refactor
    assert learn_call_args["original_code"] == faulty_code
    assert learn_call_args["corrected_code"] == corrected_code

    # 2. Check posted messages
    posted = mock_fragment_context.posted_messages
    assert len(posted) == 2, f"Expected 2 messages (reward + result), found {len(posted)}"

    # Check reward message
    reward_msg = find_message(posted, "reward")
    assert reward_msg is not None, "Reward message was not posted"
    assert reward_msg["content"]["target"] == structure_fragment.get_name()
    assert reward_msg["content"]["amount"] > 0
    assert f"Successfully corrected and validated file: {target_path_str}" in reward_msg["content"]["reason"]

    # Check success result message
    success_result_msg = find_message(posted, "refactor_result")
    assert success_result_msg is not None, "Successful refactor_result message was not posted"
    assert success_result_msg["content"]["status"] == "success"
    assert success_result_msg["content"]["original_action"] == "refactor_module"
    assert success_result_msg["content"]["target"] == target_path_str
    assert "Successfully corrected and sandbox-tested" in success_result_msg["content"]["summary"]
    assert "Learning recorded" in success_result_msg["content"]["summary"]
    # Check details
    success_details = json.loads(success_result_msg["content"]["details"])
    assert success_details["attempts_needed"] == 1
    assert success_details["final_sandbox_status"] == "success"
    assert success_details["final_sandbox_exit_code"] == 0 