import pytest
import logging
import json
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, List, Any
from pathlib import Path

# --- A³X Core Imports ---
try:
    from a3x.fragments.structure_auto_refactor import StructureAutoRefactorFragment
    from a3x.fragments.base import FragmentDef
    from a3x.core.tool_registry import ToolRegistry
    # Import fixtures implicitly from conftest.py
    from a3x.tests.emergence.conftest import find_message
except ImportError as e:
    pytest.skip(f"Skipping StructureAutoRefactor tests due to import errors: {e}", allow_module_level=True)

logger = logging.getLogger(__name__)

# --- Fixture for the Fragment Under Test --- #

@pytest.fixture
def structure_fragment(
    mock_fragment_context: MagicMock, # From conftest.py
    tool_registry: MagicMock,          # From conftest.py, used to get skills
    mock_skills: Dict[str, AsyncMock] # From conftest.py, ensures skills are mocked
) -> StructureAutoRefactorFragment:
    """Provides an instance of StructureAutoRefactorFragment with mocked context/tools."""
    # Use skills registered by mock_skills fixture
    registered_skills = list(tool_registry.list_tools().keys())
    metadata = {"name": "StructureAutoRefactor", "description": "Test", "category": "Test",
                "skills": registered_skills} # Use mocked skills
    frag_def = FragmentDef(name=metadata["name"], description=metadata["description"],
                           category=metadata["category"], skills=metadata["skills"],
                           fragment_class=StructureAutoRefactorFragment)

    fragment = StructureAutoRefactorFragment(fragment_def=frag_def, tool_registry=tool_registry)
    fragment._logger.setLevel(logging.DEBUG)
    
    # --- Crucial Patching for message capture --- 
    # Since the fragment calls `self.post_chat_message`, we need to patch *this instance's* method
    # to use the capture mechanism from our mock_fragment_context fixture.
    async def patched_post_chat_message(*args, **kwargs):
        # The fragment's internal call might pass itself or specific args.
        # We delegate to the mock context's capture method, ensuring sender is correct.
        kwargs['sender'] = fragment.get_name() # Set the correct sender
        await mock_fragment_context.post_chat_message(*args, **kwargs)

    fragment.post_chat_message = patched_post_chat_message 
    # ---------------------------------------------

    return fragment 

# --- Test Cases ---

@pytest.mark.asyncio
async def test_create_module_success(
    structure_fragment: StructureAutoRefactorFragment,
    mock_fragment_context: MagicMock,
    mock_skills: Dict[str, AsyncMock],
    workspace_root: Path # Fixture from conftest.py
):
    """Test successful handling of a create_helper_module directive."""
    target_rel_path = "a3x/modules/temp/success_helper.py"
    target_abs_path = workspace_root / target_rel_path
    directive_message = "Create a simple working module."
    initial_directive_content = {
        "type": "directive",
        "action": "create_helper_module",
        "target": target_rel_path, # Use relative path in directive
        "message": directive_message
    }
    # Simulate the incoming message that the fragment's handle_realtime_chat would receive
    incoming_message = {
        "type": "architecture_suggestion",
        "sender": "TestRunner",
        "content": initial_directive_content
    }

    # Configure mock skills for success
    generated_code = "print('OK')"
    mock_skills["generate_module_from_directive"].return_value = {
        "status": "success",
        "content": generated_code,
        "path": str(target_abs_path) # Skill returns absolute path
    }
    mock_skills["write_file"].return_value = {"status": "success", "data": {}}
    mock_skills["execute_python_in_sandbox"].return_value = {
        "status": "success", "exit_code": 0, "stdout": "OK", "stderr": ""
    }

    # --- Act ---
    # Call the fragment's message handler
    await structure_fragment.handle_realtime_chat(incoming_message, mock_fragment_context)

    # --- Assert ---
    # 1. Check skill calls (using the registry from the fragment)
    tool_registry = structure_fragment._tool_registry
    tool_registry.get_tool("generate_module_from_directive").assert_called_once_with(
        "generate_module_from_directive",
        {"message": directive_message, "target_path": str(target_abs_path)}
    )
    tool_registry.get_tool("write_file").assert_called_once_with(
        "write_file",
        {"file_path": target_rel_path, "content": generated_code}
    )
    tool_registry.get_tool("execute_python_in_sandbox").assert_called_once_with(
        "execute_python_in_sandbox",
        {"script_path": target_rel_path}
    )

    # 2. Check posted messages (using the helper on the mock context)
    posted = mock_fragment_context.get_posted_messages()
    assert len(posted) == 1, f"Expected 1 message, found {len(posted)}"

    result_msg = find_message(posted, "refactor_result") # find_message is in conftest.py
    assert result_msg is not None, "refactor_result message was not posted"
    assert result_msg["sender"] == structure_fragment.get_name() # Check sender is correct
    assert result_msg["content"]["status"] == "success"
    assert result_msg["content"]["original_action"] == "create_helper_module"
    assert result_msg["content"]["target"] == target_rel_path
    assert "Successfully generated, created, and sandbox-tested" in result_msg["content"]["summary"]

@pytest.mark.asyncio
async def test_handle_create_module_directive_sandbox_failure(
    structure_fragment: StructureAutoRefactorFragment,
    mock_fragment_context: MagicMock,
    mock_skills: Dict[str, AsyncMock]
):
    """Test handling of create_helper_module when sandbox execution fails."""
    target_path_str = "a3x/modules/temp/failing_helper.py"
    directive_message = "Create a helper module designed to fail execution."
    # Need the *content* dictionary for handle_directive
    directive_content = {
        "type": "directive",
        "action": "create_helper_module",
        "target": target_path_str,
        "message": directive_message
    }
    # Simulate the incoming message that the fragment's handle_realtime_chat would receive
    incoming_message = {
        "type": "architecture_suggestion", 
        "sender": "TestRunner",
        "content": directive_content
    }

    # Configure mock skills
    failing_code = "def fail_func():\n    print(undefined_variable)\nfail_func()" # Code with NameError
    abs_target_path = mock_fragment_context.workspace_root / target_path_str
    mock_skills["generate_module_from_directive"].return_value = {
        "status": "success",
        "content": failing_code,
        "path": str(abs_target_path)
    }
    mock_skills["write_file"].return_value = {"status": "success", "data": {}}
    # Simulate sandbox failure
    sandbox_error_message = "Traceback (most recent call last):\n  File \"<string>\", line 3, in <module>\nNameError: name 'undefined_variable' is not defined"
    mock_skills["execute_python_in_sandbox"].return_value = {
        "status": "error", # Indicate failure
        "exit_code": 1, 
        "stdout": "",
        "stderr": sandbox_error_message
    }

    # --- Act ---
    # Call the fragment's message handler
    await structure_fragment.handle_realtime_chat(incoming_message, mock_fragment_context)

    # --- Assert ---
    # 1. Check skill calls (generation, write, sandbox attempt)
    tool_registry = structure_fragment._tool_registry
    tool_registry.get_tool("generate_module_from_directive").assert_called_once()
    tool_registry.get_tool("write_file").assert_called_once()
    tool_registry.get_tool("execute_python_in_sandbox").assert_called_once()

    # 2. Check posted messages
    posted = mock_fragment_context.get_posted_messages()
    assert len(posted) == 1, f"Expected 1 message (refactor_result), found {len(posted)}"
    # The fragment *should* post the refactor_result, and *within* handle_directive, 
    # it calls broadcast_result_via_chat which uses the patched post_chat_message.
    # It *then* initiates correction internally by calling handle_directive again, 
    # OR by posting a new architecture_suggestion. Checking the code, it posts a new suggestion.
    # Let's re-check the fragment code... Ah, it seems I made a mistake in the previous analysis.
    # _handle_create_helper_module *does* post an architecture_suggestion on failure.

    # Re-assertion: Expect 2 messages: result(error) + suggestion(correction)
    assert len(posted) == 2, f"Expected 2 messages (result + correction), found {len(posted)}"

    # Check the error result message
    error_result_msg = find_message(posted, "refactor_result")
    assert error_result_msg is not None, "refactor_result message was not posted"
    assert error_result_msg["sender"] == structure_fragment.get_name()
    assert error_result_msg["content"]["status"] == "error"
    assert error_result_msg["content"]["original_action"] == "create_helper_module"
    assert error_result_msg["content"]["target"] == target_path_str
    assert "failed sandbox execution" in error_result_msg["content"]["summary"]
    assert "Correction cycle initiated" in error_result_msg["content"]["summary"]
    error_details = json.loads(error_result_msg["content"]["details"])
    assert error_details["sandbox_stderr"] == sandbox_error_message
    assert error_details["correction_initiated"] is True

    # Check the correction directive message
    correction_suggestion_msg = find_message(posted, "architecture_suggestion")
    assert correction_suggestion_msg is not None, "architecture_suggestion (for correction) was not posted"
    assert correction_suggestion_msg["sender"] == structure_fragment.get_name()
    correction_content = correction_suggestion_msg["content"]
    assert correction_content["type"] == "directive"
    assert correction_content["action"] == "refactor_module"
    assert correction_content["target"] == target_path_str
    assert "failed during sandbox execution" in correction_content["message"]
    assert sandbox_error_message in correction_content["message"]

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
    # Note: We call handle_directive directly here, assuming the routing logic
    # (handle_realtime_chat) correctly extracted the content.
    correction_directive_content = {
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
    # We are testing the handler for the correction directive directly
    await structure_fragment.handle_directive(correction_directive_content, mock_fragment_context)

    # --- Assert ---
    # 1. Check skill calls (read, modify, write, sandbox, learn)
    tool_registry = structure_fragment._tool_registry
    tool_registry.get_tool("read_file").assert_called_once_with("read_file", {"path": target_path_str})
    
    # Check modify_code args precisely
    tool_registry.get_tool("modify_code").assert_called_once() 
    modify_call_args = tool_registry.get_tool("modify_code").call_args[0][1] # Get the dict arg
    # The fragment constructs a specific instruction including the stderr message
    assert f"Stderr:\n{initial_error}" in modify_call_args["modification"]
    assert f"Original Code (potentially already modified):\n```python\n{faulty_code}\n```" in modify_call_args["modification"]
    assert modify_call_args["code_to_modify"] == faulty_code
    
    tool_registry.get_tool("write_file").assert_called_once_with("write_file", {"file_path": target_path_str, "content": corrected_code, "overwrite": True})
    tool_registry.get_tool("execute_python_in_sandbox").assert_called_once_with("execute_python_in_sandbox", {"script_path": target_path_str})
    tool_registry.get_tool("learn_from_correction_result").assert_called_once()
    
    # Check learn skill args 
    learn_call_args = tool_registry.get_tool("learn_from_correction_result").call_args[0][1]
    # The fragment passes the *full* incoming message to the learn skill's stderr parameter
    assert learn_call_args["stderr"] == correction_directive_content["message"]
    assert learn_call_args["original_code"] == faulty_code # Should be code before this correction attempt
    assert learn_call_args["corrected_code"] == corrected_code

    # 2. Check posted messages
    posted = mock_fragment_context.get_posted_messages()
    # Should post a reward and a success result
    assert len(posted) == 2, f"Expected 2 messages (reward + result), found {len(posted)}"

    # Check reward message
    reward_msg = find_message(posted, "reward") # find_message from conftest.py
    assert reward_msg is not None, "Reward message was not posted"
    assert reward_msg["sender"] == structure_fragment.get_name()
    assert reward_msg["content"]["target"] == structure_fragment.get_name()
    assert reward_msg["content"]["amount"] > 0
    assert f"Successfully corrected and validated file: {target_path_str}" in reward_msg["content"]["reason"]

    # Check success result message
    success_result_msg = find_message(posted, "refactor_result")
    assert success_result_msg is not None, "Successful refactor_result message was not posted"
    assert success_result_msg["sender"] == structure_fragment.get_name()
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

@pytest.mark.asyncio
async def test_structure_auto_refactor_generates_module_from_directive(
    structure_fragment: StructureAutoRefactorFragment,
    mock_fragment_context: MagicMock,
    mock_skills: Dict[str, AsyncMock],
    workspace_root: Path # To construct paths
):
    """Test successful generation and writing of a module from a directive."""
    target_path_relative = "a3x/utils/new_helper.py"
    target_path_absolute = workspace_root / target_path_relative
    directive = "Create a Python module with a function 'add(a, b)' that returns the sum of two numbers."
    generated_code = "def add(a, b):\n    return a + b"

    # Simulate incoming directive message (Nested as expected by the handler)
    directive_content = {
        "type": "directive", # Inner type check
        "action": "create_helper_module",
        "target": target_path_relative, # Changed key to 'target' as expected by handler
        "message": directive # Changed key to 'message' as expected by handler
    }
    incoming_message = {
        "type": "architecture_suggestion", # Outer type expected by handler
        "sender": "Planner", # Example sender
        "content": directive_content
    }

    # Configure mock skill responses for success
    mock_skills["generate_module_from_directive"] = AsyncMock(return_value={
        "status": "success", 
        "data": {"code": generated_code}
    })
    mock_skills["write_file"] = AsyncMock(return_value={
        "status": "success", 
        "data": {"bytes_written": len(generated_code)}
    })
    mock_skills["execute_python_in_sandbox"] = AsyncMock(return_value={
        "status": "success",
        "exit_code": 0,
        "stdout": "",
        "stderr": ""
    })

    # --- Mock ToolRegistry locally and assign --- # 
    # Bypassing the potentially problematic check in the handler
    mock_registry = MagicMock(spec=ToolRegistry)
    # Configure get_tool on the mock registry to return our test mocks
    def get_tool_side_effect(skill_name):
        if skill_name in mock_skills:
            return mock_skills[skill_name]
        else:
            raise KeyError(f"Mock skill {skill_name} not found")
    mock_registry.get_tool = MagicMock(side_effect=get_tool_side_effect)
    structure_fragment._tool_registry = mock_registry
    # -------------------------------------------- #

    # --- Patch post_chat_message directly on the instance --- #
    mock_post_chat = AsyncMock(name="test_generate_module_patched_post_chat")
    structure_fragment.post_chat_message = mock_post_chat
    # ---------------------------------------------------------- #

    # --- Act --- 
    # Call the handler that processes directives (likely handle_realtime_chat or similar)
    await structure_fragment.handle_realtime_chat(incoming_message, mock_fragment_context)

    # --- Assert ---
    # 1. Check skill calls
    mock_skills["generate_module_from_directive"].assert_called_once_with(
        target_path=str(target_path_absolute), # <<< CHANGED: Expect absolute path
        directive=directive
    )
    mock_skills["write_file"].assert_called_once_with(
        file_path=target_path_relative, # Write file uses relative path calculated later
        content=generated_code
    )
    mock_skills["execute_python_in_sandbox"].assert_called_once_with(script_path=target_path_relative)

    # 2. Check posted success message
    structure_fragment.post_chat_message.assert_called_once()
    call_args, call_kwargs = structure_fragment.post_chat_message.call_args
    
    assert call_kwargs['context'] == mock_fragment_context
    assert call_kwargs['message_type'] == "refactor_result" # Fragment usually posts results
    posted_content = call_kwargs['content']
    
    assert posted_content.get("status") == "success"
    assert posted_content.get("original_action") == "create_helper_module"
    assert posted_content.get("target") == target_path_relative
    assert "summary" in posted_content
    assert "sandbox-tested module" in posted_content["summary"].lower()
    assert "details" in posted_content # Should include details like path
    # Details might be JSON string or dict depending on fragment implementation
    # Parse details if it's a string
    details_dict = {}
    if isinstance(posted_content["details"], str):
        try:
            details_dict = json.loads(posted_content["details"])
        except json.JSONDecodeError:
            pytest.fail(f"Failed to parse details JSON: {posted_content['details']}")
    elif isinstance(posted_content["details"], dict):
        details_dict = posted_content["details"]
    else:
        pytest.fail(f"Unexpected details format: {type(posted_content['details'])}")

    assert details_dict.get("generated_path") == str(target_path_absolute) # Check the absolute path in details

    # Check additional details
    # success_msg_content = json.loads(posted_content["details"])
    # assert success_msg_content["corrected_path"] == target_path_absolute # Incorrect key for creation
    # assert success_msg_content["learned_status"] == "skipped" # Incorrect assertion for creation

    # Removed duplicated/incorrect assertions from previous step

@pytest.mark.asyncio
async def test_structure_auto_refactor_fails_on_generation_error(
    structure_fragment: StructureAutoRefactorFragment,
    mock_fragment_context: MagicMock,
    mock_skills: Dict[str, AsyncMock],
    workspace_root: Path # To construct paths
):
    """Test error handling when generate_module_from_directive skill fails."""
    target_path_relative = "a3x/utils/gen_fail_helper.py"
    target_path_absolute = workspace_root / target_path_relative
    directive = "Create a module that will cause a generation error."
    error_message = "LLM generation failed spectacularly"

    # Simulate incoming directive message
    directive_content = {
        "type": "directive", 
        "action": "create_helper_module",
        "target": target_path_relative, 
        "message": directive
    }
    incoming_message = {
        "type": "architecture_suggestion",
        "sender": "Planner", 
        "content": directive_content
    }

    # Configure mock skill responses
    # *** Key: generate_module_from_directive raises an Exception ***
    mock_skills["generate_module_from_directive"] = AsyncMock(side_effect=Exception(error_message))
    # Other skills should not be called

    # --- Mock ToolRegistry locally and assign --- # 
    mock_registry = MagicMock(spec=ToolRegistry)
    def get_tool_side_effect(skill_name):
        if skill_name in mock_skills:
            return mock_skills[skill_name]
        else:
            raise KeyError(f"Mock skill {skill_name} not found")
    mock_registry.get_tool = MagicMock(side_effect=get_tool_side_effect)
    structure_fragment._tool_registry = mock_registry
    # -------------------------------------------- #

    # --- Patch post_chat_message directly on the instance --- #
    mock_post_chat = AsyncMock(name="test_gen_fail_patched_post_chat")
    structure_fragment.post_chat_message = mock_post_chat
    # ---------------------------------------------------------- #

    # --- Act ---
    await structure_fragment.handle_realtime_chat(incoming_message, mock_fragment_context)

    # --- Assert ---
    # 1. Check skill calls
    mock_skills["generate_module_from_directive"].assert_called_once_with(
        target_path=str(target_path_absolute), 
        directive=directive
    )
    mock_skills["write_file"].assert_not_called() # *** Should NOT be called ***
    mock_skills["execute_python_in_sandbox"].assert_not_called() # *** Should NOT be called ***

    # 2. Check posted error message
    structure_fragment.post_chat_message.assert_called_once()
    call_args, call_kwargs = structure_fragment.post_chat_message.call_args
    
    assert call_kwargs['context'] == mock_fragment_context
    assert call_kwargs['message_type'] == "refactor_result" 
    posted_content = call_kwargs['content']
    
    assert posted_content.get("status") == "error"
    assert posted_content.get("original_action") == "create_helper_module"
    assert posted_content.get("target") == target_path_relative
    assert "unexpected error creating/testing module" in posted_content["summary"].lower()
    assert "details" in posted_content
    assert error_message in posted_content["details"] # Check exception message is in details