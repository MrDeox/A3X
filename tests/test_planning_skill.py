# tests/test_planning_skill.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os
import json
from a3x.skills.planning import hierarchical_planner
from a3x.core.context import _ToolExecutionContext
from a3x.core.llm_interface import LLMInterface
from a3x.core.tool_registry import ToolRegistry
from a3x.fragments.registry import FragmentRegistry

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


# <<< ADDED: Helper to create mock context >>>
def create_mock_plan_context(llm_response=None, llm_side_effect=None, available_tools=None):
    """Creates a mock _ToolExecutionContext for the planner skill."""
    mock_llm = AsyncMock(spec=LLMInterface)
    # Ensure call_llm itself is a mock we can check calls on
    mock_llm.call_llm = AsyncMock()

    if llm_side_effect:
        mock_llm.call_llm.side_effect = llm_side_effect
    elif llm_response is not None:
        # Configure the mock to return an async generator
        mock_llm.call_llm.return_value = async_generator_for(llm_response)
    else:
        # Configure the mock to return an empty async generator
        async def async_gen_empty():
            if False: yield
        mock_llm.call_llm.return_value = async_gen_empty()

    mock_tool_reg = MagicMock(spec=ToolRegistry)
    # Mock get_tool_details to return a basic schema for listed tools
    def mock_get_details(tool_name):
        if available_tools and tool_name in available_tools:
            return {
                "name": tool_name,
                "description": f"Description for {tool_name}",
                "parameters": {"arg1": {"type": "str", "description": "Some arg"}}
            }
        return None
    mock_tool_reg.get_tool_details.side_effect = mock_get_details

    mock_frag_reg = MagicMock(spec=FragmentRegistry)
    mock_frag_reg.get_fragment_definition.return_value = None # Assume no fragments for now

    # Mock the context object
    mock_ctx = MagicMock(spec=_ToolExecutionContext)
    mock_ctx.logger = MagicMock()
    mock_ctx.llm_interface = mock_llm
    mock_ctx.tool_registry = mock_tool_reg
    mock_ctx.fragment_registry = mock_frag_reg
    mock_ctx.shared_task_context = MagicMock() # Mock shared context if needed
    mock_ctx.shared_task_context.task_id = "test-task-123" # Example task ID

    return mock_ctx, mock_llm, mock_tool_reg, mock_frag_reg


# --- Test Cases (Modified to use context) ---

@pytest.mark.asyncio
async def test_hierarchical_planner_success():
    task_description = "Test objective"
    available_tools = ["Tool_A", "Tool_B"] # Use list of strings
    expected_plan = [
        {"step_id": 1, "description": "Do A", "action_type": "skill", "target_name": "Tool_A", "arguments": {"arg1": "valueA"}},
        {"step_id": 2, "description": "Do B", "action_type": "skill", "target_name": "Tool_B", "arguments": {"arg1": "valueB"}}
    ]
    mock_llm_response = json.dumps(expected_plan)

    # Create mock context with the LLM response
    mock_ctx, mock_llm, _, _ = create_mock_plan_context(
        llm_response=mock_llm_response,
        available_tools=available_tools
    )

    # Call the skill with context
    result = await hierarchical_planner(
        context=mock_ctx,
        task_description=task_description,
        available_tools=available_tools
    )

    assert result["status"] == "success"
    assert result["data"]["plan"] == expected_plan
    # Check if LLM was called (hard to assert specific args due to complex prompt)
    assert mock_llm.call_llm.called
    # Check if logger was used
    assert mock_ctx.logger.info.called


@pytest.mark.asyncio
async def test_hierarchical_planner_empty_plan():
    task_description = "Empty plan objective"
    available_tools = ["Tool_C"] # Use list of strings
    mock_llm_response = "[]"  # Empty list JSON

    mock_ctx, mock_llm, _, _ = create_mock_plan_context(
        llm_response=mock_llm_response,
        available_tools=available_tools
    )

    result = await hierarchical_planner(
        context=mock_ctx,
        task_description=task_description,
        available_tools=available_tools
    )

    assert result["status"] == "success" # Still success, just empty plan
    assert result["data"]["plan"] == []
    assert mock_llm.call_llm.called
    assert mock_ctx.logger.info.called # Check planner started
    assert mock_ctx.logger.info.call_count >= 2 # Start + Success message


@pytest.mark.asyncio
async def test_hierarchical_planner_invalid_json():
    task_description = "Invalid JSON objective"
    available_tools = ["Tool_D"]
    mock_llm_response = "this is not json"

    mock_ctx, mock_llm, _, _ = create_mock_plan_context(
        llm_response=mock_llm_response,
        available_tools=available_tools
    )

    result = await hierarchical_planner(
        context=mock_ctx,
        task_description=task_description,
        available_tools=available_tools
    )

    assert result["status"] == "failure"
    assert "Failed to parse LLM plan response" in result["error"]
    assert mock_llm.call_llm.called
    assert mock_ctx.logger.error.called # Check for error log


@pytest.mark.asyncio
async def test_hierarchical_planner_not_list():
    task_description = "Not list objective"
    available_tools = ["Tool_E"]
    mock_llm_response = json.dumps({"plan": "not a list"}) # Valid JSON, but not a list

    mock_ctx, mock_llm, _, _ = create_mock_plan_context(
        llm_response=mock_llm_response,
        available_tools=available_tools
    )

    result = await hierarchical_planner(
        context=mock_ctx,
        task_description=task_description,
        available_tools=available_tools
    )

    assert result["status"] == "failure"
    assert "Parsed JSON is not a list" in result["error"]
    assert mock_llm.call_llm.called
    assert mock_ctx.logger.error.called


@pytest.mark.asyncio
async def test_hierarchical_planner_llm_exception():
    task_description = "LLM exception objective"
    available_tools = ["Tool_F"]
    mock_exception = ValueError("LLM API Error")

    mock_ctx, mock_llm, _, _ = create_mock_plan_context(
        llm_side_effect=mock_exception, # Use side_effect for exceptions
        available_tools=available_tools
    )

    result = await hierarchical_planner(
        context=mock_ctx,
        task_description=task_description,
        available_tools=available_tools
    )

    assert result["status"] == "failure"
    assert "Exception during LLM call" in result["error"]
    assert f"{mock_exception}" in result["error"]
    assert mock_llm.call_llm.called
    assert mock_ctx.logger.exception.called # Check for exception log


@pytest.mark.asyncio
async def test_hierarchical_planner_missing_llm_interface():
    task_description = "Missing LLM"
    available_tools = ["Tool_G"]

    mock_ctx, _, _, _ = create_mock_plan_context(available_tools=available_tools)
    mock_ctx.llm_interface = None # Explicitly remove LLM interface

    result = await hierarchical_planner(
        context=mock_ctx,
        task_description=task_description,
        available_tools=available_tools
    )

    assert result["status"] == "failure"
    assert "LLMInterface missing" in result["error"]
    assert mock_ctx.logger.error.called


# To run: pytest tests/test_planning_skill.py -v -s
