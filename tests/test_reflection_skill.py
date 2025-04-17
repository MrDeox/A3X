# tests/test_reflection_skill.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from a3x.skills.reflection import (
    reflect_plan_step,
    _parse_reflection_output,
)
# Import context types for mocking
from a3x.core.context import Context, _ToolExecutionContext
from a3x.core.llm_interface import LLMInterface

# Add project root to sys.path
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)


# <<< ADDED: Helper for async generator >>>
async def async_generator_for(item):
    yield item


# --- Tests for _parse_reflection_output (Synchronous Helper) ---


def test_parse_reflection_output_execute():
    response = "Decision: execute\nJustification: The step seems safe and necessary."
    expected = {
        "decision": "execute",
        "justification": "The step seems safe and necessary.",
    }
    assert _parse_reflection_output(response) == expected


def test_parse_reflection_output_modify():
    response = "Decision: modify\nJustification: Need to specify the filename."
    expected = {"decision": "modify", "justification": "Need to specify the filename."}
    assert _parse_reflection_output(response) == expected


def test_parse_reflection_output_skip():
    response = "Decision: skip\nJustification: This step is redundant."
    expected = {"decision": "skip", "justification": "This step is redundant."}
    assert _parse_reflection_output(response) == expected


def test_parse_reflection_output_case_insensitive():
    response = "decision: EXECUTE\njustification: Looks good."
    expected = {"decision": "execute", "justification": "Looks good."}
    assert _parse_reflection_output(response) == expected


def test_parse_reflection_output_extra_whitespace():
    response = " Decision :  modify \n Justification :\tNeeds more detail. "
    expected = {"decision": "modify", "justification": "Needs more detail."}
    assert _parse_reflection_output(response) == expected


def test_parse_reflection_output_multiline_justification():
    response = "Decision: skip\nJustification: Step is too risky.\nIt might delete important files."
    expected = {
        "decision": "skip",
        "justification": "Step is too risky.\nIt might delete important files.",
    }
    assert _parse_reflection_output(response) == expected


def test_parse_reflection_output_missing_justification():
    response = "Decision: execute"
    expected = {"decision": "execute", "justification": "No justification provided."}
    assert _parse_reflection_output(response) == expected


def test_parse_reflection_output_missing_decision():
    response = "Justification: Seems okay."
    expected = {"decision": "unknown", "justification": "Seems okay."}
    assert _parse_reflection_output(response) == expected


def test_parse_reflection_output_malformed():
    response = "Let's execute this. Justification: Go!"
    expected = {"decision": "execute", "justification": "Go!"}  # Fallback guess
    assert _parse_reflection_output(response) == expected


def test_parse_reflection_output_empty():
    response = ""
    expected = {"decision": "unknown", "justification": "No justification provided."}
    assert _parse_reflection_output(response) == expected


# --- Tests for reflect_plan_step (Async Skill) ---

# Helper function to create a mock context
def create_mock_context(llm_response=None, llm_side_effect=None):
    mock_llm = AsyncMock(spec=LLMInterface)
    if llm_side_effect:
        # Use side_effect for exceptions
        mock_llm.call_llm.side_effect = llm_side_effect
    elif llm_response is not None:
        # Use return_value for normal responses (async generator)
        async def mock_llm_stream(*args, **kwargs):
            yield llm_response
        mock_llm.call_llm = mock_llm_stream
    else: # Default empty response if neither specified
         async def mock_llm_stream_empty(*args, **kwargs):
             if False: # Ensure it's an async generator
                 yield
         mock_llm.call_llm = mock_llm_stream_empty

    # Mock the context object passed to the skill
    # Use _ToolExecutionContext as it's often the type passed by execute_tool
    mock_ctx = MagicMock(spec=_ToolExecutionContext)
    mock_ctx.logger = MagicMock() # Mock logger methods if needed
    mock_ctx.llm_interface = mock_llm # Assign the mocked LLM interface
    mock_ctx.memory = None # Mock memory if needed by the skill
    # Add other attributes if the skill uses them from context

    return mock_ctx, mock_llm # Return both context and llm mock for assertions

@pytest.mark.asyncio
async def test_reflect_plan_step_success_execute():
    step_index = 1
    plan = ["Step 1", "Step 2"]
    action = "do_something"
    observation = "Simulation looks good."
    success = True
    mock_llm_response = "Decision: execute\nJustification: Seems fine."
    mock_ctx, mock_llm = create_mock_context(llm_response=mock_llm_response)

    result = await reflect_plan_step(
        context=mock_ctx,
        step_index=step_index,
        plan=plan,
        action_taken=action,
        observation=observation,
        success=success
    )

    assert result["status"] == "success"
    assert result["decision"] == "execute"
    assert result["justification"] == "Seems fine."
    # mock_llm.call_llm.assert_called_once() # Difficult to assert on the generator directly

@pytest.mark.asyncio
async def test_reflect_plan_step_success_modify():
    step_index = 2
    plan = ["Step 1", "Step 2"]
    action = "do_something_else"
    observation = "Might fail."
    success = False
    mock_llm_response = "Decision: modify\nJustification: Needs adjustment."
    mock_ctx, mock_llm = create_mock_context(llm_response=mock_llm_response)

    result = await reflect_plan_step(
        context=mock_ctx,
        step_index=step_index,
        plan=plan,
        action_taken=action,
        observation=observation,
        success=success
    )

    assert result["status"] == "success"
    assert result["decision"] == "modify"
    assert result["justification"] == "Needs adjustment."

@pytest.mark.asyncio
async def test_reflect_plan_step_success_skip():
    step_index = 3
    plan = ["Step 1", "Step 2", "Step 3"]
    action = "do_nothing"
    observation = "Redundant."
    success = True # Assume simulation ran successfully but outcome is redundant
    mock_llm_response = "Decision: skip\nJustification: Not needed."
    mock_ctx, mock_llm = create_mock_context(llm_response=mock_llm_response)

    result = await reflect_plan_step(
        context=mock_ctx,
        step_index=step_index,
        plan=plan,
        action_taken=action,
        observation=observation,
        success=success
    )

    assert result["status"] == "success"
    assert result["decision"] == "skip"
    assert result["justification"] == "Not needed."

@pytest.mark.asyncio
async def test_reflect_plan_step_llm_error():
    step_index = 1
    plan = ["Step 1"]
    action = "causes_error"
    observation = "Failed in simulation."
    success = False
    mock_exception = Exception("LLM API Error")
    mock_ctx, mock_llm = create_mock_context(llm_side_effect=mock_exception)

    result = await reflect_plan_step(
        context=mock_ctx,
        step_index=step_index,
        plan=plan,
        action_taken=action,
        observation=observation,
        success=success
    )

    assert result["status"] == "error"
    assert result["decision"] == "unknown"
    assert "LLM error" in result["justification"]
    assert f"{mock_exception}" in result["error_message"]
    # mock_llm.call_llm.assert_called_once() # Assert call even on exception

@pytest.mark.asyncio
async def test_reflect_plan_step_empty_llm_response():
    step_index = 1
    plan = ["Step 1"]
    action = "empty_response_action"
    observation = "Simulation outcome."
    success = True
    mock_llm_response = "" # Empty string response
    mock_ctx, mock_llm = create_mock_context(llm_response=mock_llm_response)

    result = await reflect_plan_step(
        context=mock_ctx,
        step_index=step_index,
        plan=plan,
        action_taken=action,
        observation=observation,
        success=success
    )

    assert result["status"] == "error"
    assert result["decision"] == "unknown"
    # Check the specific error message for empty response
    assert "LLM response was empty" in result["justification"]
    assert "LLM response empty" in result["error_message"]

@pytest.mark.asyncio
async def test_reflect_plan_step_missing_llm_interface():
    """Tests the case where LLMInterface is missing in context."""
    step_index = 1
    plan = ["Step 1"]
    action = "some_action"
    observation = "Some outcome."
    success = True

    # Create a mock context but explicitly set llm_interface to None
    mock_ctx = MagicMock(spec=_ToolExecutionContext)
    mock_ctx.logger = MagicMock()
    mock_ctx.llm_interface = None # Simulate missing interface
    mock_ctx.memory = None

    result = await reflect_plan_step(
        context=mock_ctx,
        step_index=step_index,
        plan=plan,
        action_taken=action,
        observation=observation,
        success=success
    )

    assert result["status"] == "error"
    assert result["decision"] == "unknown"
    assert "LLMInterface missing" in result["justification"]
    assert "LLMInterface missing" in result["error_message"]
