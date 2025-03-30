
# tests/test_planner.py
import pytest
import json
import logging
import requests # Import requests *only* for exception types
from unittest import mock
from unittest.mock import MagicMock, patch, AsyncMock # Import AsyncMock

# Import the functions/constants to be tested
from core.planner import generate_plan, PLANNING_SCHEMA
from core.prompt_builder import build_planning_prompt
# Do NOT import call_llm directly if we are patching it by string path

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
    return "http://mock-llm-url-planner/v1/chat/completions" # Use a distinct URL if needed

# --- Test Cases for generate_plan (now async) ---

@pytest.mark.asyncio
async def test_generate_plan_success(mock_agent_logger, mock_tool_descriptions, mock_llm_url):
    """Test successful plan generation."""
    objective = "Test objective"
    expected_plan = ["Step 1", "Step 2"]
    mock_llm_content = json.dumps(expected_plan) # Planner expects a JSON list string

    # Patch the correct target: where call_llm is LOOKED UP when generate_plan runs
    # Use new_callable=AsyncMock for async functions
    with patch('core.planner.call_llm', new_callable=AsyncMock) as mock_call_llm:
        mock_call_llm.return_value = mock_llm_content # Set return value on the AsyncMock

        plan = await generate_plan(objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url)

    assert plan == expected_plan
    mock_call_llm.assert_awaited_once()
    call_args, call_kwargs = mock_call_llm.await_args # Use await_args for async mock
    messages = call_args[0]
    assert any(objective in msg['content'] for msg in messages if msg['role'] == 'user')
    mock_agent_logger.info.assert_any_call(f"[Planner] Plan generated successfully with {len(expected_plan)} steps.")

@pytest.mark.asyncio
async def test_generate_plan_llm_http_error(mock_agent_logger, mock_tool_descriptions, mock_llm_url):
    """Test plan generation when call_llm raises an HTTPError."""
    objective = "Test objective for HTTP error"
    http_error = requests.exceptions.HTTPError("Mock 500 Error", response=MagicMock())

    # Patch with AsyncMock and set side_effect
    with patch('core.planner.call_llm', new_callable=AsyncMock) as mock_call_llm:
        mock_call_llm.side_effect = http_error

        plan = await generate_plan(objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url)

    assert plan is None
    mock_call_llm.assert_awaited_once()
    # Check the exception log message precisely if possible
    mock_agent_logger.exception.assert_called_once_with(f"[Planner] Error calling LLM for planning: {http_error}")

@pytest.mark.asyncio
async def test_generate_plan_llm_timeout(mock_agent_logger, mock_tool_descriptions, mock_llm_url):
    """Test plan generation when call_llm raises a Timeout."""
    objective = "Test objective for timeout"
    timeout_error = requests.exceptions.Timeout("Mock Timeout Error")

    # Patch with AsyncMock and set side_effect
    with patch('core.planner.call_llm', new_callable=AsyncMock) as mock_call_llm:
        mock_call_llm.side_effect = timeout_error

        plan = await generate_plan(objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url)

    assert plan is None
    mock_call_llm.assert_awaited_once()
    mock_agent_logger.exception.assert_called_once_with(f"[Planner] Error calling LLM for planning: {timeout_error}")

@pytest.mark.asyncio
async def test_generate_plan_invalid_json_response(mock_agent_logger, mock_tool_descriptions, mock_llm_url):
    """Test plan generation when LLM returns a non-JSON string."""
    objective = "Test objective for invalid json"
    mock_llm_content = "This is not valid JSON { nor a block ```json ... ```"

    # Patch with AsyncMock and set return_value
    with patch('core.planner.call_llm', new_callable=AsyncMock) as mock_call_llm:
        mock_call_llm.return_value = mock_llm_content

        plan = await generate_plan(objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url)

    assert plan is None
    mock_call_llm.assert_awaited_once()
    # Check that an error containing 'No JSON block found' or 'decode' was logged
    error_logs = [call for call in mock_agent_logger.error.call_args_list if 'JSON' in str(call.args[0])]
    assert len(error_logs) > 0, "Expected JSON decode/find error log message"

@pytest.mark.asyncio
async def test_generate_plan_valid_json_not_list(mock_agent_logger, mock_tool_descriptions, mock_llm_url):
    """Test plan generation when LLM returns valid JSON but not a list."""
    objective = "Test objective valid json not list"
    mock_llm_content = json.dumps({"plan": "This is not a list"}) # Valid JSON, wrong type

    # Patch with AsyncMock and set return_value
    with patch('core.planner.call_llm', new_callable=AsyncMock) as mock_call_llm:
        mock_call_llm.return_value = mock_llm_content

        plan = await generate_plan(objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url)

    assert plan is None
    mock_call_llm.assert_awaited_once()
    # Check that an error containing 'not a list' was logged
    error_logs = [call for call in mock_agent_logger.error.call_args_list if 'not a list of strings' in str(call.args[0])]
    assert len(error_logs) > 0, "Expected 'not a list' error log message"

@pytest.mark.asyncio
async def test_generate_plan_list_not_strings(mock_agent_logger, mock_tool_descriptions, mock_llm_url):
    """Test plan generation when 'plan' is a list but contains non-strings."""
    objective = "Test objective plan not strings"
    mock_llm_content = json.dumps(["Step 1", {"step": 2}]) # List contains a dict

    # Patch with AsyncMock and set return_value
    with patch('core.planner.call_llm', new_callable=AsyncMock) as mock_call_llm:
        mock_call_llm.return_value = mock_llm_content

        plan = await generate_plan(objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url)

    assert plan is None
    mock_call_llm.assert_awaited_once()
    # Check that an error containing 'not a list of strings' was logged
    error_logs = [call for call in mock_agent_logger.error.call_args_list if 'not a list of strings' in str(call.args[0])]
    assert len(error_logs) > 0, "Expected 'list not strings' error log message"

# --- Test for build_planning_prompt --- 

def test_build_planning_prompt(mock_tool_descriptions):
    """Test that the planning prompt is built correctly."""
    objective = "My Test Objective"
    from core.planner import PLANNER_SYSTEM_PROMPT # Import the system prompt

    messages = build_planning_prompt(objective, mock_tool_descriptions, PLANNER_SYSTEM_PROMPT)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[0]["content"] == PLANNER_SYSTEM_PROMPT
    assert objective in messages[1]["content"]
    assert mock_tool_descriptions in messages[1]["content"]
