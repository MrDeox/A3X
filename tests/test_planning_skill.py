# tests/test_planning_skill.py
import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import json
from a3x.skills.planning import hierarchical_planner

# --- Add project root to sys.path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio

# Removed async_generator_for helper


# <<< ADDED: Helper for async generator >>>
async def async_generator_for(item):
    """Creates a simple async generator that yields a single item."""
    yield item


# --- Fixtures (if any, keep as is) ---
@pytest.fixture
def mock_planner_logger():
    return MagicMock()


# --- Test Cases ---


@pytest.mark.asyncio
async def test_hierarchical_planner_success(mock_planner_logger):
    objective = "Test objective"
    tools = "Tool A, Tool B"
    expected_plan = [
        "Step 1: Use Tool A with relevant info.",
        "Step 2: Use Tool B based on Tool A output.",
        "Step 3: Final answer summarizing results.",
    ]
    mock_llm_response = json.dumps(expected_plan)

    with patch("skills.planning.call_llm") as mock_call_llm:
        mock_call_llm.return_value = async_generator_for(mock_llm_response)

        plan = await hierarchical_planner(objective, tools)

        assert plan == expected_plan
        mock_call_llm.assert_called_once()
        call_args, call_kwargs = mock_call_llm.call_args
        assert isinstance(call_args[0], list)  # Check prompt is list
        assert not call_kwargs.get("stream")
        # mock_planner_logger.info.assert_called()


@pytest.mark.asyncio
async def test_hierarchical_planner_empty_plan(mock_planner_logger):
    objective = "Empty plan objective"
    tools = "Tool C"
    mock_llm_response = "[]"  # Empty list JSON

    with patch("skills.planning.call_llm") as mock_call_llm:
        mock_call_llm.return_value = async_generator_for(mock_llm_response)

        plan = await hierarchical_planner(objective, tools)

        assert plan == []
        mock_call_llm.assert_called_once()
        call_args, call_kwargs = mock_call_llm.call_args
        assert isinstance(call_args[0], list)
        assert not call_kwargs.get("stream")
        # mock_planner_logger.warning.assert_called_with(
        #     "[PlanningSkill] LLM returned an empty plan list []."
        # )


@pytest.mark.asyncio
async def test_hierarchical_planner_invalid_json(mock_planner_logger):
    objective = "Invalid JSON objective"
    tools = "Tool D"
    mock_llm_response = "this is not json"

    with patch("skills.planning.call_llm") as mock_call_llm:
        mock_call_llm.return_value = async_generator_for(mock_llm_response)

        plan = await hierarchical_planner(objective, tools)

        assert plan is None
        mock_call_llm.assert_called_once()
        call_args, call_kwargs = mock_call_llm.call_args
        assert isinstance(call_args[0], list)
        assert not call_kwargs.get("stream")
        # mock_planner_logger.error.assert_called()


@pytest.mark.asyncio
async def test_hierarchical_planner_not_list(mock_planner_logger):
    objective = "Not list objective"
    tools = "Tool E"
    mock_llm_response = json.dumps({"plan": "not a list"})

    with patch("skills.planning.call_llm") as mock_call_llm:
        mock_call_llm.return_value = async_generator_for(mock_llm_response)

        plan = await hierarchical_planner(objective, tools)

        assert plan is None
        # <<< MODIFIED: Check call args, ensure stream=False >>>
        mock_call_llm.assert_called_once()
        call_args, call_kwargs = mock_call_llm.call_args
        assert isinstance(call_args[0], list)
        assert not call_kwargs.get("stream")
        # mock_planner_logger.error.assert_called()


@pytest.mark.asyncio
async def test_hierarchical_planner_llm_exception(mock_planner_logger):
    objective = "LLM exception objective"
    tools = "Tool F"
    mock_exception = ValueError("LLM Error")

    # <<< MODIFIED: Patch skills.planning.call_llm, use MagicMock, side_effect=exception >>>
    # For exceptions, we still use side_effect with the standard MagicMock
    with patch("skills.planning.call_llm") as mock_call_llm:
        mock_call_llm.side_effect = mock_exception

        plan = await hierarchical_planner(objective, tools)

        assert plan is None
        # <<< MODIFIED: Check call args, ensure stream=False >>>
        mock_call_llm.assert_called_once()
        call_args, call_kwargs = mock_call_llm.call_args
        assert isinstance(call_args[0], list)
        assert not call_kwargs.get("stream")
        # mock_planner_logger.exception.assert_called_once_with(
        #     f"[PlanningSkill] Error calling LLM: {mock_exception}"
        # )


# To run: pytest tests/test_planning_skill.py -v -s
