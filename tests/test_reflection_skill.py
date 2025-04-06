# tests/test_reflection_skill.py
import pytest
from unittest.mock import patch
from a3x.skills.reflection import (
    reflect_plan_step,
    _parse_reflection_output,
)

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


@pytest.mark.asyncio
async def test_reflect_plan_step_success_execute():
    step = "Test Step 1"
    outcome = "Simulation looks good."
    mock_llm_response = "Decision: execute\nJustification: Seems fine."

    # <<< MODIFIED: Patch target, use MagicMock, return async_generator >>>
    with patch("skills.reflection.call_llm") as mock_call_llm:
        mock_call_llm.return_value = async_generator_for(mock_llm_response)
        result = await reflect_plan_step(step, outcome)

        assert result["status"] == "success"
        assert result["decision"] == "execute"
        assert result["justification"] == "Seems fine."
        # <<< MODIFIED: Check call args, ensure stream=False >>>
        mock_call_llm.assert_called_once()
        call_args, call_kwargs = mock_call_llm.call_args
        assert isinstance(call_args[0], list)  # Check messages format
        assert not call_kwargs.get("stream")


@pytest.mark.asyncio
async def test_reflect_plan_step_success_modify():
    step = "Test Step 2"
    outcome = "Might fail."
    mock_llm_response = "Decision: modify\nJustification: Needs adjustment."

    # <<< MODIFIED: Patch target, use MagicMock, return async_generator >>>
    with patch("skills.reflection.call_llm") as mock_call_llm:
        mock_call_llm.return_value = async_generator_for(mock_llm_response)
        result = await reflect_plan_step(step, outcome)

        assert result["status"] == "success"
        assert result["decision"] == "modify"
        assert result["justification"] == "Needs adjustment."
        # <<< MODIFIED: Check call args, ensure stream=False >>>
        mock_call_llm.assert_called_once()
        call_args, call_kwargs = mock_call_llm.call_args
        assert isinstance(call_args[0], list)
        assert not call_kwargs.get("stream")


@pytest.mark.asyncio
async def test_reflect_plan_step_success_skip():
    step = "Test Step 3"
    outcome = "Redundant."
    mock_llm_response = "Decision: skip\nJustification: Not needed."

    # <<< MODIFIED: Patch target, use MagicMock, return async_generator >>>
    with patch("skills.reflection.call_llm") as mock_call_llm:
        mock_call_llm.return_value = async_generator_for(mock_llm_response)
        result = await reflect_plan_step(step, outcome)

        assert result["status"] == "success"
        assert result["decision"] == "skip"
        assert result["justification"] == "Not needed."
        # <<< MODIFIED: Check call args, ensure stream=False >>>
        mock_call_llm.assert_called_once()
        call_args, call_kwargs = mock_call_llm.call_args
        assert isinstance(call_args[0], list)
        assert not call_kwargs.get("stream")


@pytest.mark.asyncio
async def test_reflect_plan_step_llm_error():
    step = "Test Step Error"
    outcome = "Causes error."
    mock_exception = Exception("LLM API Error")

    # <<< MODIFIED: Patch target, use MagicMock, use side_effect for exception >>>
    with patch("skills.reflection.call_llm") as mock_call_llm:
        mock_call_llm.side_effect = mock_exception
        result = await reflect_plan_step(step, outcome)

        assert result["status"] == "error"
        assert result["decision"] == "unknown"
        assert "LLM error" in result["justification"]
        assert f"{mock_exception}" in result["error_message"]
        # <<< MODIFIED: Check call args, ensure stream=False >>>
        mock_call_llm.assert_called_once()
        call_args, call_kwargs = mock_call_llm.call_args
        assert isinstance(call_args[0], list)
        assert not call_kwargs.get("stream")


@pytest.mark.asyncio
async def test_reflect_plan_step_empty_llm_response():
    step = "Test Step Empty"
    outcome = "Outcome Empty"
    mock_llm_response = ""

    # <<< MODIFIED: Patch target, use MagicMock, return async_generator >>>
    with patch("skills.reflection.call_llm") as mock_call_llm:
        mock_call_llm.return_value = async_generator_for(mock_llm_response)
        result = await reflect_plan_step(step, outcome)

        assert result["status"] == "error"
        assert result["decision"] == "unknown"
        assert "empty or not a string" in result["justification"]
        # <<< MODIFIED: Check call args, ensure stream=False >>>
        mock_call_llm.assert_called_once()
        call_args, call_kwargs = mock_call_llm.call_args
        assert isinstance(call_args[0], list)
        assert not call_kwargs.get("stream")
