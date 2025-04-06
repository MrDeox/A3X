# tests/test_planner.py
import pytest
import json
import logging
import requests  # Import requests *only* for exception types
from unittest.mock import MagicMock, patch, AsyncMock  # Import AsyncMock

# Import the functions/constants to be tested
from core.planner import generate_plan
from core.prompt_builder import build_planning_prompt


# Mock logger fixture
@pytest.fixture
def mock_agent_logger():
    return MagicMock(spec=logging.Logger)


# Mock tool descriptions fixture
@pytest.fixture
def mock_tool_descriptions():
    return "Tool A: Does A.\nTool B: Does B."


# Mock LLM URL fixture
@pytest.fixture
def mock_llm_url():
    return "http://mock-llm-url-planner/v1/chat/completions"  # Use a distinct URL if needed


# --- Test Cases for generate_plan (now async) ---


@pytest.mark.asyncio
async def test_generate_plan_success(
    mock_agent_logger, mock_tool_descriptions, mock_llm_url
):
    """Test successful plan generation."""
    objective = "Test objective"
    expected_plan = ["Step 1", "Step 2"]
    # Ensure the mock content is *exactly* what the parser expects (raw JSON)
    mock_llm_content = json.dumps(expected_plan)

    # Define an async generator function for the mock
    async def mock_llm_return():
        yield mock_llm_content

    with patch("core.planner.call_llm") as mock_call_llm:
        mock_call_llm.return_value = mock_llm_return()

        plan = await generate_plan(
            objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url
        )

        # Add assertion to check the mock was called correctly
        mock_call_llm.assert_called_once()

        # Explicitly check the plan value before assertion
        print(f"\n[Test Log] generate_plan returned: {plan}\n")

    assert plan == expected_plan
    # mock_call_llm.assert_called_once() # Check that the mock was called
    # Redundant - checked above
    call_args, call_kwargs = mock_call_llm.call_args # Use call_args for sync call
    messages = call_args[0]
    assert any(objective in msg["content"] for msg in messages if msg["role"] == "user")
    mock_agent_logger.info.assert_any_call(
        f"[Planner] Plan generated successfully with {len(expected_plan)} steps."
    )


@pytest.mark.asyncio
async def test_generate_plan_llm_http_error(
    mock_agent_logger, mock_tool_descriptions, mock_llm_url
):
    """Test plan generation when call_llm raises an HTTPError."""
    objective = "Test objective for HTTP error"
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 500
    mock_response.reason = "Internal Server Error"
    mock_response.url = mock_llm_url
    mock_response.request = MagicMock(
        spec=requests.PreparedRequest
    )  # Mock the request attribute
    http_error = requests.exceptions.HTTPError("Mock 500 Error", response=mock_response)

    # Define an async function to raise the error
    async def mock_llm_raise_http():
        raise http_error
        yield

    with patch("core.planner.call_llm") as mock_call_llm:
        mock_call_llm.return_value = mock_llm_raise_http()

        plan = await generate_plan(
            objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url
        )

    assert plan is None
    mock_call_llm.assert_called_once()
    mock_agent_logger.exception.assert_called_once_with(
        f"[Planner] Error calling LLM for planning: {http_error}"
    )


@pytest.mark.asyncio
async def test_generate_plan_llm_timeout(
    mock_agent_logger, mock_tool_descriptions, mock_llm_url
):
    """Test plan generation when call_llm raises a Timeout."""
    objective = "Test objective for timeout"
    timeout_error = requests.exceptions.Timeout("Mock Timeout Error")

    # Define an async function to raise the error
    async def mock_llm_raise_timeout():
        raise timeout_error
        yield

    with patch("core.planner.call_llm") as mock_call_llm:
        mock_call_llm.return_value = mock_llm_raise_timeout()

        plan = await generate_plan(
            objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url
        )

    assert plan is None
    mock_call_llm.assert_called_once()
    mock_agent_logger.exception.assert_called_once_with(
        f"[Planner] Error calling LLM for planning: {timeout_error}"
    )


@pytest.mark.asyncio
async def test_generate_plan_invalid_json_response(
    mock_agent_logger, mock_tool_descriptions, mock_llm_url
):
    """Test plan generation when LLM returns a non-JSON string."""
    objective = "Test objective for invalid json"
    mock_llm_content = "This is not valid JSON { nor a block ```json ... ```"

    async def mock_llm_return_invalid():
        yield mock_llm_content

    with patch("core.planner.call_llm") as mock_call_llm:
        mock_call_llm.return_value = mock_llm_return_invalid()

        plan = await generate_plan(
            objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url
        )

    assert plan is None
    mock_call_llm.assert_called_once()
    # Check that an error containing 'No JSON block found' or 'decode' was logged
    # Ensure the correct logger method (error) is checked
    error_logs = [
        call
        for call in mock_agent_logger.error.call_args_list
        if ("Failed to decode JSON" in str(call.args[0]))
        or ("No JSON block found" in str(call.args[0]))
        or ("Fallback failed" in str(call.args[0]))
    ]
    assert len(error_logs) > 0, "Expected JSON decode/find error log message"


@pytest.mark.asyncio
async def test_generate_plan_valid_json_not_list(
    mock_agent_logger, mock_tool_descriptions, mock_llm_url
):
    """Test plan generation when LLM returns valid JSON but not a list."""
    objective = "Test objective valid json not list"
    mock_llm_content = json.dumps(
        {"plan": "This is not a list"}
    )  # Valid JSON, wrong type

    async def mock_llm_return_dict():
        yield mock_llm_content

    with patch("core.planner.call_llm") as mock_call_llm:
        mock_call_llm.return_value = mock_llm_return_dict()

        plan = await generate_plan(
            objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url
        )

    assert plan is None
    mock_call_llm.assert_called_once()
    # Check that an error containing 'not a list' was logged
    error_logs = [
        call
        for call in mock_agent_logger.error.call_args_list
        if "LLM response is not a list" in str(call.args[0])
    ]
    assert len(error_logs) > 0, "Expected 'not a list' error log message"


@pytest.mark.asyncio
async def test_generate_plan_list_not_strings(
    mock_agent_logger, mock_tool_descriptions, mock_llm_url
):
    """Test plan generation when 'plan' is a list but contains non-strings."""
    objective = "Test objective plan not strings"
    mock_llm_content = json.dumps(["Step 1", {"step": 2}])  # List contains a dict

    async def mock_llm_return_mixed_list():
        yield mock_llm_content

    with patch("core.planner.call_llm") as mock_call_llm:
        mock_call_llm.return_value = mock_llm_return_mixed_list()

        plan = await generate_plan(
            objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url
        )

    assert plan is None
    mock_call_llm.assert_called_once()
    # Check that an error containing 'not a list of strings' was logged
    error_logs = [
        call
        for call in mock_agent_logger.error.call_args_list
        if "Plan list contains non-string elements" in str(call.args[0])
    ]
    assert len(error_logs) > 0, "Expected 'list not strings' error log message"


# --- Test for build_planning_prompt ---


def test_build_planning_prompt(mock_tool_descriptions):
    """Test that the planning prompt is built correctly."""
    objective = "My Test Objective"
    from core.planner import PLANNER_SYSTEM_PROMPT  # Import the system prompt

    messages = build_planning_prompt(
        objective, mock_tool_descriptions, PLANNER_SYSTEM_PROMPT
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[0]["content"] == PLANNER_SYSTEM_PROMPT
    assert objective in messages[1]["content"]
    assert mock_tool_descriptions in messages[1]["content"]
