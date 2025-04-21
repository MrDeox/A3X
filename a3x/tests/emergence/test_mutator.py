import pytest
import logging
import json
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict

# --- A³X Core Imports ---
try:
    from a3x.fragments.mutator import MutatorFragment
    from a3x.fragments.base import FragmentDef
    # Import find_message implicitly from conftest.py
    from a3x.tests.emergence.conftest import find_message 
except ImportError as e:
    pytest.skip(f"Skipping Mutator tests due to import errors: {e}", allow_module_level=True)

logger = logging.getLogger(__name__)

# --- Fixture for the Fragment Under Test --- #

@pytest.fixture
def mutator_fragment(
    mock_fragment_context: MagicMock, # From conftest.py
    tool_registry: MagicMock,          # From conftest.py, used to get skills
    mock_skills: Dict[str, AsyncMock] # From conftest.py, ensures skills are mocked
) -> MutatorFragment:
    """Provides an instance of MutatorFragment with mocked context/tools."""
    mutator_skills = ["read_file", "modify_code", "write_file"]
    metadata = {"name": "Mutator", "description": "Test", "category": "Test",
                "skills": mutator_skills}
    frag_def = FragmentDef(name=metadata["name"], description=metadata["description"],
                           category=metadata["category"], skills=metadata["skills"],
                           fragment_class=MutatorFragment)

    fragment = MutatorFragment(fragment_def=frag_def, tool_registry=tool_registry)
    fragment._logger.setLevel(logging.DEBUG)
    
    # --- Patching for message capture --- 
    async def patched_post_chat_message(*args, **kwargs):
        kwargs['sender'] = fragment.get_name() 
        await mock_fragment_context.post_chat_message(*args, **kwargs)
    fragment.post_chat_message = patched_post_chat_message
    # -------------------------------------

    return fragment

# --- Test Cases --- #

@pytest.mark.asyncio
async def test_mutator_handles_error_result(
    mutator_fragment: MutatorFragment,
    mock_fragment_context: MagicMock,
    mock_skills: Dict[str, AsyncMock]
):
    """Test MutatorFragment reacts to a refactor error result, reads, modifies, writes, and proposes a mutation."""
    target_path_str = "a3x/modules/temp/failed_module.py"
    error_stderr = "Traceback...\nNameError: something is wrong"
    error_details_str = json.dumps({
        "sandbox_status": "error",
        "sandbox_exit_code": 1,
        "sandbox_stderr": error_stderr,
        # Mutator specifically looks for generated_path or target
        "generated_path": target_path_str 
    })
    # Simulate the incoming error message
    incoming_message = {
        "type": "refactor_result", # Message type Mutator listens for
        "sender": "StructureAutoRefactor", # Or another source
        "content": {
            "status": "error",
            "target": target_path_str, # Can also be here
            "original_action": "create_helper_module",
            "summary": "Module created but failed sandbox execution.",
            "details": error_details_str
        }
    }

    # Configure mock skills for the mutation flow
    original_code = "print(some_undefined_variable)"
    mutated_code = "print('mutated code!')"
    # Ensure mocks are AsyncMocks if not already set by fixture
    # mock_skills["read_file"] = AsyncMock(return_value={"status": "success", "data": {"content": original_code}})
    # mock_skills["modify_code"] = AsyncMock(return_value={"status": "success", "data": {"modified_code": mutated_code}})
    # mock_skills["write_file"] = AsyncMock(return_value={"status": "success", "data": {"bytes_written": len(mutated_code)}})
    
    # *** Configure the return values of the EXISTING mocks from the fixture ***
    mock_skills["read_file"].return_value = {"status": "success", "data": {"content": original_code}}
    mock_skills["modify_code"].return_value = {"status": "success", "data": {"modified_code": mutated_code}}
    mock_skills["write_file"].return_value = {"status": "success", "data": {"bytes_written": len(mutated_code)}}

    # --- Patch post_chat_message directly on the instance --- #
    # (Ensuring it's patched even if fixture patching failed/wasn't done)
    mock_post_chat = AsyncMock(name="test_handler_patched_post_chat")
    mutator_fragment.post_chat_message = mock_post_chat
    # ---------------------------------------------------------- #

    # --- Act ---
    # Call the fragment's message handler
    await mutator_fragment.handle_realtime_chat(incoming_message, mock_fragment_context)

    # --- Assert ---
    # 1. Check skill calls (read, modify, write)
    # Use the mocks directly from the mock_skills fixture
    mock_skills["read_file"].assert_called_once_with(file_path=target_path_str)
    
    # Check modify_code args - ensure it includes error context
    mock_skills["modify_code"].assert_called_once()
    modify_call_kwargs = mock_skills["modify_code"].call_args.kwargs # Get kwargs
    assert modify_call_kwargs['file_path'] == target_path_str
    assert modify_call_kwargs['original_code'] == original_code
    assert error_stderr in modify_call_kwargs['instructions'] # Check error context is included

    mock_skills["write_file"].assert_called_once_with(file_path=target_path_str, content=mutated_code)

    # 2. Check posted message (mutation_attempt)
    mutator_fragment.post_chat_message.assert_called_once()
    call_args, call_kwargs = mutator_fragment.post_chat_message.call_args
    
    assert call_kwargs['context'] == mock_fragment_context
    assert call_kwargs['message_type'] == "mutation_attempt"
    posted_content = call_kwargs['content']
    
    assert posted_content.get("status") == "proposed" # Changed from success
    assert posted_content.get("file_path") == target_path_str
    assert posted_content.get("from") == "Mutator" # Added check for origin
    assert "modification_details" in posted_content
    # Optionally check details like snippets if needed
    assert posted_content["modification_details"]['original_code_snippet'] == original_code[:500]
    assert posted_content["modification_details"]['modified_code_snippet'] == mutated_code[:500]
    assert posted_content["modification_details"]['instructions'] == modify_call_kwargs['instructions']

@pytest.mark.asyncio
async def test_mutator_skips_when_modify_code_returns_no_change(
    mutator_fragment: MutatorFragment,
    mock_fragment_context: MagicMock,
    mock_skills: Dict[str, AsyncMock]
):
    """Test Mutator skips write and post when modify_code reports no change."""
    target_path_str = "a3x/modules/temp/no_change_module.py"
    error_stderr = "Traceback...\nSomeError: A minor issue"
    error_details_str = json.dumps({
        "sandbox_status": "error",
        "sandbox_exit_code": 1,
        "sandbox_stderr": error_stderr,
        "generated_path": target_path_str 
    })
    incoming_message = {
        "type": "refactor_result",
        "sender": "StructureAutoRefactor",
        "content": {
            "status": "error",
            "target": target_path_str,
            "details": error_details_str
        }
    }

    # Configure mock skills
    original_code = "def func(): pass"
    mock_skills["read_file"].return_value = {"status": "success", "data": {"content": original_code}}
    # *** Key part: modify_code returns 'no_change' ***
    mock_skills["modify_code"].return_value = {"status": "no_change", "data": {"modified_code": original_code}} # Content might be same or None
    # write_file mock doesn't need specific config as it shouldn't be called

    # --- Patch post_chat_message directly on the instance --- #
    mock_post_chat = AsyncMock(name="test_no_change_patched_post_chat")
    mutator_fragment.post_chat_message = mock_post_chat
    # ---------------------------------------------------------- #

    # --- Act ---
    await mutator_fragment.handle_realtime_chat(incoming_message, mock_fragment_context)

    # --- Assert ---
    # 1. Check skill calls
    mock_skills["read_file"].assert_called_once_with(file_path=target_path_str)
    mock_skills["modify_code"].assert_called_once() # Should still be called
    mock_skills["write_file"].assert_not_called() # *** Should NOT be called ***

    # 2. Check no message was posted
    mutator_fragment.post_chat_message.assert_not_called() # *** Should NOT be called ***

@pytest.mark.asyncio
async def test_mutator_aborts_on_modify_code_error(
    mutator_fragment: MutatorFragment,
    mock_fragment_context: MagicMock,
    mock_skills: Dict[str, AsyncMock]
):
    """Test Mutator aborts if modify_code skill returns an error status."""
    target_path_str = "a3x/modules/temp/modify_fail_module.py"
    error_stderr = "Traceback...\nTypeError: bad type"
    error_details_str = json.dumps({
        "sandbox_status": "error",
        "sandbox_exit_code": 1,
        "sandbox_stderr": error_stderr,
        "generated_path": target_path_str 
    })
    incoming_message = {
        "type": "refactor_result",
        "sender": "StructureAutoRefactor",
        "content": {
            "status": "error",
            "target": target_path_str,
            "details": error_details_str
        }
    }

    # Configure mock skills
    original_code = "def func_with_type_error(): return 1 + '1'"
    mock_skills["read_file"].return_value = {"status": "success", "data": {"content": original_code}}
    # *** Key part: modify_code returns 'error' ***
    mock_skills["modify_code"].return_value = {"status": "error", "data": {"message": "LLM modification failed"}}
    # write_file mock doesn't need specific config as it shouldn't be called

    # --- Patch post_chat_message directly on the instance --- #
    mock_post_chat = AsyncMock(name="test_modify_error_patched_post_chat")
    mutator_fragment.post_chat_message = mock_post_chat
    # ---------------------------------------------------------- #

    # --- Act ---
    await mutator_fragment.handle_realtime_chat(incoming_message, mock_fragment_context)

    # --- Assert ---
    # 1. Check skill calls
    mock_skills["read_file"].assert_called_once_with(file_path=target_path_str)
    mock_skills["modify_code"].assert_called_once() # Should still be called
    mock_skills["write_file"].assert_not_called() # *** Should NOT be called ***

    # 2. Check no message was posted
    mutator_fragment.post_chat_message.assert_not_called() # *** Should NOT be called ***

# Removed duplicated assertions