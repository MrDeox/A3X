# tests/test_planning_skill.py
import pytest
from unittest.mock import patch, AsyncMock
import sys
import os
import json
from skills.planning import hierarchical_planner, DEFAULT_PLANNER_SYSTEM_PROMPT
from core.prompt_builder import build_planning_prompt  # Need this to verify prompt args

# --- Add project root to sys.path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio

# --- Test Cases ---


@patch("skills.planning.call_llm", new_callable=AsyncMock)
async def test_hierarchical_planner_success(mock_call_llm):
    """Tests successful plan generation when LLM returns a valid JSON list."""
    # Arrange: Mock the LLM response (direct string return for stream=False)
    mock_llm_response_json = json.dumps(
        ["Step 1: Do this", "Step 2: Do that", "Step 3: Finish up"]
    )
    mock_call_llm.return_value = (
        mock_llm_response_json  # <<< MODIFIED: Use return_value
    )

    objective = "Test objective for planning"
    tools_desc = "- tool1: Desc 1\n- tool2: Desc 2"

    # Act: Call the planner skill
    result = await hierarchical_planner(objective=objective, available_tools=tools_desc)

    # Assert: Check if LLM was called correctly and result is successful
    expected_prompt = build_planning_prompt(
        objective, tools_desc, DEFAULT_PLANNER_SYSTEM_PROMPT
    )
    mock_call_llm.assert_called_once_with(expected_prompt, stream=False)

    assert result["status"] == "success"
    assert result["action"] == "plan_generated"
    assert "data" in result
    assert isinstance(result["data"].get("plan"), list)
    assert len(result["data"]["plan"]) == 3
    assert result["data"]["plan"][0] == "Step 1: Do this"
    print(f"Success Test Result: {result}")


@patch("skills.planning.call_llm", new_callable=AsyncMock)
async def test_hierarchical_planner_empty_plan(mock_call_llm):
    """Tests the case where the LLM returns an empty list."""
    # Arrange: Mock LLM response with empty list string
    mock_llm_response_json = json.dumps([])
    mock_call_llm.return_value = (
        mock_llm_response_json  # <<< MODIFIED: Use return_value
    )

    objective = "Empty plan test"
    tools_desc = "Tool descriptions"

    # Act
    result = await hierarchical_planner(objective=objective, available_tools=tools_desc)

    # Assert: Should return a warning status
    assert result["status"] == "warning"
    assert result["action"] == "plan_generated_empty"
    assert result["data"]["plan"] == []
    print(f"Empty Plan Test Result: {result}")


@patch("skills.planning.call_llm", new_callable=AsyncMock)
async def test_hierarchical_planner_invalid_json(mock_call_llm):
    """Tests the case where the LLM returns invalid JSON."""
    # Arrange: Mock LLM response with broken JSON string
    mock_llm_response_text = "This is not JSON [Step 1, Step 2] oops"
    mock_call_llm.return_value = (
        mock_llm_response_text  # <<< MODIFIED: Use return_value
    )

    objective = "Invalid JSON test"
    tools_desc = "Tool descriptions"

    # Act
    result = await hierarchical_planner(objective=objective, available_tools=tools_desc)

    # Assert: Should return a parsing error
    assert result["status"] == "error"
    assert result["action"] == "plan_generation_failed_parsing"
    assert "Failed to parse plan JSON" in result["data"]["message"]
    print(f"Invalid JSON Test Result: {result}")


@patch("skills.planning.call_llm", new_callable=AsyncMock)
async def test_hierarchical_planner_not_list(mock_call_llm):
    """Tests the case where the LLM returns valid JSON, but not a list of strings."""
    # Arrange: Mock LLM response with a dictionary string
    mock_llm_response_json = json.dumps({"plan": ["Step 1"]})
    mock_call_llm.return_value = (
        mock_llm_response_json  # <<< MODIFIED: Use return_value
    )

    objective = "Not list test"
    tools_desc = "Tool descriptions"

    # Act
    result = await hierarchical_planner(objective=objective, available_tools=tools_desc)

    # Assert: Should return a structure error
    assert result["status"] == "error"
    assert result["action"] == "plan_generation_failed_structure"
    assert (
        "Invalid plan structure: Parsed JSON is type 'dict', not list."
        in result["data"]["message"]
    )
    print(f"Not List Test Result: {result}")


@patch("skills.planning.call_llm", new_callable=AsyncMock)
async def test_hierarchical_planner_llm_exception(mock_call_llm):
    """Tests the case where the call_llm function raises an exception."""
    # Arrange: Mock call_llm to raise an exception
    # side_effect is still correct for raising exceptions
    mock_call_llm.side_effect = Exception("Simulated LLM API error")

    objective = "LLM exception test"
    tools_desc = "Tool descriptions"

    # Act
    result = await hierarchical_planner(objective=objective, available_tools=tools_desc)

    # Assert: Should return an unknown error
    assert result["status"] == "error"
    assert result["action"] == "plan_generation_failed_unknown"
    assert "Simulated LLM API error" in result["data"]["message"]
    print(f"LLM Exception Test Result: {result}")


# To run: pytest tests/test_planning_skill.py -v -s
