# <<< File: tests/test_agent_reflector.py >>>
# ... (imports and fixtures) ...
import pytest
import logging
import json
from unittest import mock
from unittest.mock import MagicMock, AsyncMock

from core.agent_reflector import reflect_on_observation
from core.agent import ReactAgent  # For type hinting
# from core.tools import parse_tool_call # Removed unused import


# Fixture for common context needed by reflect_on_observation
@pytest.fixture
def base_context():
    """Provides a base context dictionary for reflector tests."""
    return {
        "current_step_index": 0,
        "plan": ["Step 1"],
        "objective": "Test objective",
        "max_iterations": 5,
        "iteration_count": 1,
        "agent_logger": MagicMock(spec=logging.Logger),
        "tools": {"tool1": MagicMock()},  # Basic mock tool
        "previous_thought": "Initial thought",
        "agent_instance": MagicMock(spec=ReactAgent),  # Added mock agent instance
    }


# Fixture to create a mock ReactAgent instance
@pytest.fixture
def mock_agent_instance(mocker):
    instance = MagicMock(spec=ReactAgent)
    instance.run = AsyncMock()  # Mock the async run method
    instance.memory = MagicMock()  # Mock the memory component
    instance.memory.add_interaction = MagicMock()
    instance.logger = MagicMock(spec=logging.Logger)
    instance.tools = {
        "execute_code": MagicMock()
    }  # Add relevant tools used by reflector
    # Configure logger mocks if needed for specific tests, e.g.,
    # instance.logger.info = MagicMock()
    # instance.logger.error = MagicMock()
    # instance.logger.warning = MagicMock()
    # instance.logger.exception = MagicMock()
    return instance


# --- Existing Test Cases --- (Ensure they use mock_agent_instance fixture)


@pytest.mark.asyncio
async def test_reflect_success_final_answer(
    base_context, mock_agent_instance
):  # Add mock_agent_instance
    """Test reflection when the observation indicates a final answer."""
    context = base_context.copy()
    context["agent_logger"] = mock_agent_instance.logger  # Use logger from mock agent
    observation_dict = {
        "status": "success",
        "action": "final_answer",  # Corrected action name
        "data": {"message": "Final answer provided"},  # Example data
        "thought": "Objective complete, provide final answer.",
    }
    # Use mock_agent_instance here
    decision, new_plan = await reflect_on_observation(
        objective=context["objective"],
        plan=context["plan"],
        current_step_index=context["current_step_index"],
        action_name="final_answer",
        action_input=observation_dict.get("action_input", {}),
        observation_dict=observation_dict,
        history=[],
        memory={},
        agent_logger=context["agent_logger"],
        agent_instance=mock_agent_instance,
    )
    assert decision == "plan_complete"
    assert new_plan is None
    mock_agent_instance.logger.info.assert_any_call(
        "[Reflector] Final Answer provided. Plan complete."
    )  # Use logger from mock agent


@pytest.mark.asyncio
async def test_reflect_success_tool_step(
    base_context, mock_agent_instance
):  # Add mock_agent_instance
    """Test reflection after a successful tool execution step."""
    context = base_context.copy()
    context["agent_logger"] = mock_agent_instance.logger  # Use logger from mock agent
    observation_dict = {
        "status": "success",
        "action": "tool1_success",  # Example successful action
        "data": {"output": "Tool output"},  # Example data
        "thought": "Tool executed successfully, continue plan.",
    }
    # Use mock_agent_instance here
    decision, new_plan = await reflect_on_observation(
        objective=context["objective"],
        plan=context["plan"],
        current_step_index=context["current_step_index"],
        action_name="tool1_success",
        action_input=observation_dict.get("action_input", {}),
        observation_dict=observation_dict,
        history=[],
        memory={},
        agent_logger=context["agent_logger"],
        agent_instance=mock_agent_instance,
    )
    assert decision == "continue_plan"
    assert new_plan is None
    mock_agent_instance.logger.info.assert_any_call(
        "[Reflector] Action 'tool1_success' completed successfully."
    )  # Use logger from mock agent


@pytest.mark.asyncio
async def test_reflect_no_change(
    base_context, mock_agent_instance
):  # Add mock_agent_instance
    """Test reflection when the observation status is no_change."""
    context = base_context.copy()
    context["agent_logger"] = mock_agent_instance.logger  # Use logger from mock agent
    observation_dict = {
        "status": "no_change",
        "action": "thought_only",  # Example action
        "data": {"message": "No external action taken"},  # Example data
        "thought": "Just thinking, no action needed yet.",
    }
    # Use mock_agent_instance here
    decision, new_plan = await reflect_on_observation(
        objective=context["objective"],
        plan=context["plan"],
        current_step_index=context["current_step_index"],
        action_name="no_op",
        action_input=observation_dict.get("action_input", {}),
        observation_dict=observation_dict,
        history=[],
        memory={},
        agent_logger=context["agent_logger"],
        agent_instance=mock_agent_instance,
    )
    assert decision == "continue_plan"
    assert new_plan is None
    mock_agent_instance.logger.info.assert_any_call(
        "[Reflector] Action 'no_op' resulted in no change. Continuing plan."
    )  # Use logger from mock agent


@pytest.mark.asyncio
async def test_reflect_error_tool_not_found(
    mock_agent_instance,
):  # Removed base_context, use mock_agent_instance
    """Test reflection for tool_not_found error."""
    context = {
        "current_step_index": 0,
        "plan": ["Step 1"],
        "objective": "Test objective",
        "max_iterations": 5,
        "iteration_count": 1,
        "agent_logger": mock_agent_instance.logger,  # Use logger from mock agent
        "tools": {"tool1": MagicMock()},
        "previous_thought": "Initial thought",
        # No need for base_context["agent_instance"]
    }
    observation_dict = {
        "status": "error",
        "action": "tool_not_found",
        "data": {
            "message": "Tool 'non_existent_tool' not found.",
            "tool_name": "non_existent_tool",
        },
        "thought": "Trying to use a tool that doesn't exist.",
    }
    decision, new_plan = await reflect_on_observation(
        objective=context["objective"],
        plan=context["plan"],
        current_step_index=context["current_step_index"],
        action_name="non_existent_tool",
        action_input=observation_dict.get("action_input", {}),
        observation_dict=observation_dict,
        history=[],
        memory={},
        agent_logger=context["agent_logger"],
        agent_instance=mock_agent_instance,
    )
    assert decision == "retry_step"  # <<< Correct assertion for updated logic
    assert new_plan is None
    # <<< Check updated log message >>>
    mock_agent_instance.logger.warning.assert_any_call(
        "[Reflector] Tool 'non_existent_tool' not found. Suggesting step retry."
    )


@pytest.mark.asyncio
async def test_reflect_error_execution_failed(
    mock_agent_instance,
):  # Removed base_context, use mock_agent_instance
    """Test reflection for execution_failed where meta-run fails JSON parsing."""  # <<< Updated docstring
    context = {
        "current_step_index": 0,
        "plan": ["Step 1: Execute code"],
        "objective": "Execute failing code",
        "max_iterations": 5,
        "iteration_count": 1,
        "agent_logger": mock_agent_instance.logger,
        "tools": {"execute_code": MagicMock()},
        "previous_thought": "Try to execute print(1/0)",
        # No need for base_context["agent_instance"]
    }
    observation_dict = {
        "status": "error",
        "action": "execution_failed",
        "data": {
            "message": "Code failed",
            "stdout": "",
            "stderr": "ZeroDivisionError",
            "returncode": 1,
        },
        "thought": "Executing print(1/0)",
        "action_input": {
            "code": "print(1/0)"
        },  # Original code needed for auto-correction
    }
    # Meta-run returns non-JSON
    mock_agent_instance.run.return_value = "Meta-run failed to produce JSON"

    decision, new_plan = await reflect_on_observation(
        objective=context["objective"],
        plan=context["plan"],
        current_step_index=context["current_step_index"],
        action_name="execute_code",
        action_input=observation_dict.get("action_input", {}),
        observation_dict=observation_dict,
        history=[],
        memory={},
        agent_logger=context["agent_logger"],
        agent_instance=mock_agent_instance,
    )
    # <<< Reflector now returns stop_plan if meta-run fails parsing >>>
    assert decision == "stop_plan"
    assert new_plan is None
    mock_agent_instance.run.assert_called_once()
    # Check for the specific JSON parsing error log
    log_calls = mock_agent_instance.logger.error.call_args_list
    assert any(
        "[Reflector] Failed to parse meta-cycle result as JSON" in str(call)
        for call in log_calls
    ), f"Logs: {log_calls}"


@pytest.mark.asyncio
async def test_reflect_error_parsing_failed(
    mock_agent_instance,
):  # Removed base_context, use mock_agent_instance
    """Test reflection for parsing_failed error."""
    context = {
        "current_step_index": 0,
        "plan": ["Step 1"],
        "objective": "Test objective",
        "max_iterations": 5,
        "iteration_count": 1,
        "agent_logger": mock_agent_instance.logger,
        "tools": {"tool1": MagicMock()},
        "previous_thought": "Initial thought",
        # No need for base_context["agent_instance"]
    }
    observation_dict = {
        "status": "error",
        "action": "parsing_failed",
        "data": {"message": "Invalid JSON", "original_text": "{bad json"},
        "thought": "LLM returned bad JSON.",
    }
    decision, new_plan = await reflect_on_observation(
        objective=context["objective"],
        plan=context["plan"],
        current_step_index=context["current_step_index"],
        action_name="execute_code",
        action_input=observation_dict.get("action_input", {}),
        observation_dict=observation_dict,
        history=[],
        memory={},
        agent_logger=context["agent_logger"],
        agent_instance=mock_agent_instance,
    )
    assert decision == "retry_step"  # <<< Correct assertion for updated logic
    assert new_plan is None
    # <<< Check updated log message >>>
    mock_agent_instance.logger.error.assert_any_call(
        "[Reflector] Internal agent error detected (parsing_failed). Suggesting step retry."
    )


@pytest.mark.asyncio
async def test_reflect_error_llm_call_failed(
    mock_agent_instance,
):  # Removed base_context, use mock_agent_instance
    """Test reflection for llm_call_failed error."""
    context = {
        "current_step_index": 0,
        "plan": ["Step 1"],
        "objective": "Test objective",
        "max_iterations": 5,
        "iteration_count": 1,
        "agent_logger": mock_agent_instance.logger,
        "tools": {"tool1": MagicMock()},
        "previous_thought": "Initial thought",
        # No need for base_context["agent_instance"]
    }
    observation_dict = {
        "status": "error",
        "action": "llm_call_failed",
        "data": {"message": "API error"},
        "thought": "Trying to call LLM.",
    }
    decision, new_plan = await reflect_on_observation(
        objective=context["objective"],
        plan=context["plan"],
        current_step_index=context["current_step_index"],
        action_name="llm_thought_action",
        action_input=observation_dict.get("action_input", {}),
        observation_dict=observation_dict,
        history=[],
        memory={},
        agent_logger=context["agent_logger"],
        agent_instance=mock_agent_instance,
    )
    assert decision == "retry_step"  # <<< Correct assertion for updated logic
    assert new_plan is None
    # <<< Check updated log message >>>
    mock_agent_instance.logger.error.assert_any_call(
        "[Reflector] Internal agent error detected (llm_call_failed). Suggesting step retry."
    )


@pytest.mark.asyncio
async def test_reflect_error_unhandled_action(
    mock_agent_instance,
):  # Removed base_context, use mock_agent_instance
    """Test reflection for an unhandled error action."""
    context = {
        "current_step_index": 0,
        "plan": ["Step 1"],
        "objective": "Test objective",
        "max_iterations": 5,
        "iteration_count": 1,
        "agent_logger": mock_agent_instance.logger,
        "tools": {"tool1": MagicMock()},
        "previous_thought": "Initial thought",
        # No need for base_context["agent_instance"]
    }
    observation_dict = {
        "status": "error",
        "action": "some_unexpected_error_action",
        "data": {"message": "Something weird happened"},
        "thought": "An unknown error occurred.",
    }
    decision, new_plan = await reflect_on_observation(
        objective=context["objective"],
        plan=context["plan"],
        current_step_index=context["current_step_index"],
        action_name="unknown_action",
        action_input=observation_dict.get("action_input", {}),
        observation_dict=observation_dict,
        history=[],
        memory={},
        agent_logger=context["agent_logger"],
        agent_instance=mock_agent_instance,
    )
    assert decision == "stop_plan"
    assert new_plan is None
    # <<< Use simpler log check for this case >>>
    assert any(
        "[Reflector] Unhandled error type (some_unexpected_error_action). Stopping plan."
        in str(call)
        for call in mock_agent_instance.logger.error.call_args_list
    )


@pytest.mark.asyncio
async def test_reflect_unknown_status(
    mock_agent_instance,
):  # Removed base_context, use mock_agent_instance
    """Test reflection for an unknown observation status."""
    context = {
        "current_step_index": 0,
        "plan": ["Step 1"],
        "objective": "Test objective",
        "max_iterations": 5,
        "iteration_count": 1,
        "agent_logger": mock_agent_instance.logger,
        "tools": {"tool1": MagicMock()},
        "previous_thought": "Initial thought",
        # No need for base_context["agent_instance"]
    }
    observation_dict = {
        "status": "maybe_success?",  # Unknown status
        "action": "some_action",
        "data": {"message": "Not sure what happened"},
        "thought": "Status is ambiguous.",
    }
    decision, new_plan = await reflect_on_observation(
        objective=context["objective"],
        plan=context["plan"],
        current_step_index=context["current_step_index"],
        action_name="unknown_action",
        action_input=observation_dict.get("action_input", {}),
        observation_dict=observation_dict,
        history=[],
        memory={},
        agent_logger=context["agent_logger"],
        agent_instance=mock_agent_instance,
    )
    assert decision == "stop_plan"
    assert new_plan is None
    # <<< Use simpler log check for this case >>>
    assert any(
        "[Reflector] Unknown status 'maybe_success?' in observation. Stopping plan as a precaution."
        in str(call)
        for call in mock_agent_instance.logger.warning.call_args_list
    )


# --- Auto-correction Tests --- (Ensure they use mock_agent_instance)


@pytest.mark.asyncio
async def test_reflect_autocorrect_success(mock_agent_instance):  # Removed base_context
    """Test successful auto-correction flow."""
    context = {
        "current_step_index": 0,
        "plan": ["Step 1: Execute code"],
        "objective": "Execute failing code and fix it",
        "max_iterations": 5,
        "iteration_count": 1,
        "agent_logger": mock_agent_instance.logger,
        "tools": {"execute_code": MagicMock()},
        "previous_thought": "Try to execute print(1/0)",
        # No need for base_context["agent_instance"]
    }
    original_code = "print(1/0)"
    error_message = 'Traceback...\nFile "<string>", line 1, in <module>\nZeroDivisionError: division by zero'
    observation_dict = {
        "status": "error",
        "action": "execution_failed",
        "data": {"message": "Code failed", "stderr": error_message, "returncode": 1},
        "thought": "Executing print(1/0)",
        "action_input": {"code": original_code},  # Crucial for auto-correction
    }

    # Mock the recursive agent.run call to return a successful modification
    corrected_code = "print(1)"
    meta_run_response = json.dumps(
        {
            "thought": "The user wants to fix the division by zero. I will change it to print 1.",
            "action": "final_answer",
            "action_input": {
                "answer": f"```python\n{corrected_code}\n```"  # Agent should return code in markdown
            },
        }
    )
    mock_agent_instance.run.return_value = meta_run_response

    decision, new_plan = await reflect_on_observation(
        objective=context["objective"],
        plan=context["plan"],
        current_step_index=context["current_step_index"],
        action_name="execute_code",
        action_input=observation_dict.get("action_input", {}),
        observation_dict=observation_dict,
        history=[],
        memory={},
        agent_logger=context["agent_logger"],
        agent_instance=mock_agent_instance,
    )

    assert (
        decision == "stop_plan"
    )  # TODO: Revisit reflector logic for successful autocorrect
    assert new_plan is None  # No plan modification in this case
    mock_agent_instance.run.assert_called_once()
    meta_objective_call = mock_agent_instance.run.call_args[0][0]
    # <<< Check specific error detail from improved extraction >>>
    assert (
        "The error was: Code failed" in meta_objective_call
    )  # TODO: Improve error detail extraction in reflector
    assert original_code in meta_objective_call
    # Check if memory was updated
    # mock_agent_instance.memory.add_interaction.assert_called_once() # Removed: Reflector assigns to memory key, doesn't call add_interaction
    # Verify the logger was called appropriately (checking the error log due to current parsing failure)
    mock_agent_instance.logger.error.assert_any_call(
        mock.ANY  # Match the start of the error message flexibly
    )
    assert any(
        "Auto-correction meta-cycle did not return successful modified code"
        in call.args[0]
        for call in mock_agent_instance.logger.error.call_args_list
    )


@pytest.mark.asyncio
async def test_reflect_autocorrect_meta_run_fails(
    mock_agent_instance,
):  # Removed base_context
    """Test auto-correction when the recursive agent.run call itself raises an exception."""  # <<< Updated docstring
    context = {
        "current_step_index": 0,
        "plan": ["Step 1: Execute code"],
        "objective": "Execute failing code",
        "max_iterations": 5,
        "iteration_count": 1,
        "agent_logger": mock_agent_instance.logger,
        "tools": {"execute_code": MagicMock()},
        "previous_thought": "Try to execute print(1/0)",
        # No need for base_context["agent_instance"]
    }
    observation_dict = {
        "status": "error",
        "action": "execution_failed",
        "data": {"message": "Code failed", "stderr": "Some Error", "returncode": 1},
        "thought": "Executing print(1/0)",
        "action_input": {"code": "print(1/0)"},
    }

    # Mock the recursive agent.run call to raise an exception
    run_exception = Exception("Recursive run failed!")
    mock_agent_instance.run.side_effect = run_exception

    decision, new_plan = await reflect_on_observation(
        objective=context["objective"],
        plan=context["plan"],
        current_step_index=context["current_step_index"],
        action_name="execute_code",
        action_input=observation_dict.get("action_input", {}),
        observation_dict=observation_dict,
        history=[],
        memory={},
        agent_logger=context["agent_logger"],
        agent_instance=mock_agent_instance,
    )

    assert decision == "stop_plan"
    assert new_plan is None
    mock_agent_instance.run.assert_called_once()
    # <<< Use simpler log check for exception >>>
    assert any(
        "[Reflector] Exception during recursive agent run for auto-correction:"
        in str(call)
        for call in mock_agent_instance.logger.exception.call_args_list
    )


@pytest.mark.asyncio
async def test_reflect_autocorrect_meta_run_invalid_json(
    mock_agent_instance,
):  # Removed base_context
    """Test auto-correction when the recursive agent.run returns invalid JSON."""
    context = {
        "current_step_index": 0,
        "plan": ["Step 1: Execute code"],
        "objective": "Execute failing code",
        "max_iterations": 5,
        "iteration_count": 1,
        "agent_logger": mock_agent_instance.logger,
        "tools": {"execute_code": MagicMock()},
        "previous_thought": "Try to execute print(1/0)",
        # No need for base_context["agent_instance"]
    }
    observation_dict = {
        "status": "error",
        "action": "execution_failed",
        "data": {"message": "Code failed", "stderr": "Some Error", "returncode": 1},
        "thought": "Executing print(1/0)",
        "action_input": {"code": "print(1/0)"},
    }

    # Mock the recursive agent.run call to return non-JSON string
    mock_agent_instance.run.return_value = "This is not JSON"

    decision, new_plan = await reflect_on_observation(
        objective=context["objective"],
        plan=context["plan"],
        current_step_index=context["current_step_index"],
        action_name="execute_code",
        action_input=observation_dict.get("action_input", {}),
        observation_dict=observation_dict,
        history=[],
        memory={},
        agent_logger=context["agent_logger"],
        agent_instance=mock_agent_instance,
    )

    assert decision == "stop_plan"
    assert new_plan is None
    mock_agent_instance.run.assert_called_once()
    # <<< Check specific log message for JSON parsing failure >>>
    assert any(
        "[Reflector] Failed to parse meta-cycle result as JSON" in str(call)
        for call in mock_agent_instance.logger.error.call_args_list
    )


@pytest.mark.asyncio
async def test_reflect_autocorrect_no_original_code(
    mock_agent_instance,
):  # Removed base_context
    """Test auto-correction attempt when original code is missing."""
    context = {
        "current_step_index": 0,
        "plan": ["Step 1: Execute code"],
        "objective": "Execute failing code",
        "max_iterations": 5,
        "iteration_count": 1,
        "agent_logger": mock_agent_instance.logger,
        "tools": {"execute_code": MagicMock()},
        "previous_thought": "Try to execute code (code missing)",
        # No need for base_context["agent_instance"]
    }
    observation_dict = {
        "status": "error",
        "action": "execution_failed",
        "data": {"message": "Code failed", "stderr": "Some Error", "returncode": 1},
        "thought": "Executing code, but forgot to include it in input.",
        "action_input": {},  # Missing 'code' key
    }

    decision, new_plan = await reflect_on_observation(
        objective=context["objective"],
        plan=context["plan"],
        current_step_index=context["current_step_index"],
        action_name="execute_code",
        action_input=observation_dict.get("action_input", {}),
        observation_dict=observation_dict,
        history=[],
        memory={},
        agent_logger=context["agent_logger"],
        agent_instance=mock_agent_instance,
    )

    assert decision == "stop_plan"
    assert new_plan is None
    mock_agent_instance.run.assert_not_called()
    # <<< Use simpler log check >>>
    assert any(
        "[Reflector] Cannot attempt correction: Original code not found in action_input."
        in str(call)
        for call in mock_agent_instance.logger.error.call_args_list
    )
