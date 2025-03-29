# tests/test_planner.py
import pytest
import json
import logging
from unittest import mock
from unittest.mock import MagicMock, patch
import requests # Import requests to mock its methods

# Import the functions to be tested
from core.planner import generate_plan, build_planning_prompt, PLANNING_SCHEMA

# Mock logger to avoid real logging during tests
@pytest.fixture(autouse=True)
def mock_logging(mocker):
    mocker.patch('logging.getLogger', return_value=MagicMock()) # Mock the logger instance

# Mock agent logger specifically if needed, though the above should cover it
@pytest.fixture
def mock_agent_logger():
    return MagicMock(spec=logging.Logger)

# Mock tool descriptions
@pytest.fixture
def mock_tool_descriptions():
    return "Tool A: Does A.\nTool B: Does B."

# Mock LLM URL
@pytest.fixture
def mock_llm_url():
    return "http://mock-llm-url/v1/chat/completions"

# --- Test Cases --- 

def test_generate_plan_success(mock_agent_logger, mock_tool_descriptions, mock_llm_url):
    """Test successful plan generation."""
    objective = "Test objective"
    expected_plan = ["Step 1", "Step 2"]
    mock_response_content = json.dumps({"plan": expected_plan})
    mock_llm_response = MagicMock()
    # Configure the mock response JSON
    mock_llm_response.json.return_value = {
        "choices": [
            {"message": {"content": mock_response_content}}
        ]
    }
    mock_llm_response.raise_for_status = MagicMock() # Mock raise_for_status to do nothing

    # Patch requests.post used within generate_plan
    with patch('core.planner.requests.post', return_value=mock_llm_response) as mock_post:
        plan = generate_plan(objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url)

    assert plan == expected_plan
    mock_post.assert_called_once()
    # Optional: Add more detailed assertions on the payload sent to the LLM
    call_args, call_kwargs = mock_post.call_args
    assert call_kwargs['json']['response_format']['schema'] == PLANNING_SCHEMA
    mock_agent_logger.info.assert_any_call("[Planner] Generating plan...")
    mock_agent_logger.info.assert_any_call(f"[Planner] Plan generated successfully with {len(expected_plan)} steps.")

def test_generate_plan_llm_error_response(mock_agent_logger, mock_tool_descriptions, mock_llm_url):
    """Test plan generation when LLM returns an error status code."""
    objective = "Test objective for error"
    mock_llm_response = MagicMock()
    # Simulate an HTTPError
    http_error = requests.exceptions.HTTPError("Mock 404 Error", response=MagicMock())
    http_error.response.text = "Not Found Error Detail"
    mock_llm_response.raise_for_status.side_effect = http_error

    with patch('core.planner.requests.post', return_value=mock_llm_response) as mock_post:
        plan = generate_plan(objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url)

    assert plan is None
    mock_post.assert_called_once()
    # Check that the specific HTTP error log was called (expects ONE string argument)
    mock_agent_logger.error.assert_any_call(mock.ANY) 

def test_generate_plan_llm_timeout(mock_agent_logger, mock_tool_descriptions, mock_llm_url):
    """Test plan generation when the LLM call times out."""
    objective = "Test objective for timeout"
    timeout_error = requests.exceptions.Timeout("Mock Timeout Error")

    with patch('core.planner.requests.post', side_effect=timeout_error) as mock_post:
        plan = generate_plan(objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url)

    assert plan is None
    mock_post.assert_called_once()
    # Check that the specific timeout error log was called (expects ONE string argument)
    mock_agent_logger.error.assert_any_call(mock.ANY)

def test_generate_plan_invalid_json_content(mock_agent_logger, mock_tool_descriptions, mock_llm_url):
    """Test plan generation when LLM returns invalid JSON in content."""
    objective = "Test objective for invalid json"
    mock_response_content = "This is not valid JSON {"
    mock_llm_response = MagicMock()
    mock_llm_response.json.return_value = {
        "choices": [
            {"message": {"content": mock_response_content}}
        ]
    }
    mock_llm_response.raise_for_status = MagicMock()

    with patch('core.planner.requests.post', return_value=mock_llm_response) as mock_post:
        plan = generate_plan(objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url)

    assert plan is None
    mock_post.assert_called_once()
    # Check that the JSON parsing error log was called (expects ONE string argument)
    mock_agent_logger.error.assert_any_call(mock.ANY) 

def test_generate_plan_missing_plan_key(mock_agent_logger, mock_tool_descriptions, mock_llm_url):
    """Test plan generation when LLM returns valid JSON but misses the 'plan' key."""
    objective = "Test objective missing key"
    mock_response_content = json.dumps({"thought": "Thinking..."}) # Missing 'plan'
    mock_llm_response = MagicMock()
    mock_llm_response.json.return_value = {
        "choices": [
            {"message": {"content": mock_response_content}}
        ]
    }
    mock_llm_response.raise_for_status = MagicMock()

    with patch('core.planner.requests.post', return_value=mock_llm_response) as mock_post:
        plan = generate_plan(objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url)

    assert plan is None
    mock_post.assert_called_once()
    # Check that the specific structural error log was called (expects ONE string argument)
    mock_agent_logger.error.assert_any_call(mock.ANY) 

def test_generate_plan_plan_not_list(mock_agent_logger, mock_tool_descriptions, mock_llm_url):
    """Test plan generation when LLM returns 'plan' key but it's not a list."""
    objective = "Test objective plan not list"
    mock_response_content = json.dumps({"plan": "This is not a list"}) 
    mock_llm_response = MagicMock()
    mock_llm_response.json.return_value = {
        "choices": [
            {"message": {"content": mock_response_content}}
        ]
    }
    mock_llm_response.raise_for_status = MagicMock()

    with patch('core.planner.requests.post', return_value=mock_llm_response) as mock_post:
        plan = generate_plan(objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url)

    assert plan is None
    mock_post.assert_called_once()
    # Check that the specific structural error log was called (expects ONE string argument)
    mock_agent_logger.error.assert_any_call(mock.ANY)

def test_generate_plan_plan_list_not_strings(mock_agent_logger, mock_tool_descriptions, mock_llm_url):
    """Test plan generation when 'plan' is a list but contains non-strings."""
    objective = "Test objective plan not strings"
    mock_response_content = json.dumps({"plan": ["Step 1", {"step": 2}]}) # Contains a dict
    mock_llm_response = MagicMock()
    mock_llm_response.json.return_value = {
        "choices": [
            {"message": {"content": mock_response_content}}
        ]
    }
    mock_llm_response.raise_for_status = MagicMock()

    with patch('core.planner.requests.post', return_value=mock_llm_response) as mock_post:
        plan = generate_plan(objective, mock_tool_descriptions, mock_agent_logger, mock_llm_url)

    assert plan is None
    mock_post.assert_called_once()
    # Check that the specific structural error log was called (expects ONE string argument)
    mock_agent_logger.error.assert_any_call(mock.ANY)

def test_build_planning_prompt(mock_agent_logger, mock_tool_descriptions):
    """Test that the planning prompt is built correctly."""
    objective = "My Test Objective"
    messages = build_planning_prompt(objective, mock_tool_descriptions, mock_agent_logger)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert objective in messages[0]["content"] # System prompt should contain objective
    assert objective in messages[1]["content"] # User prompt should contain objective
    assert mock_tool_descriptions in messages[0]["content"] # System prompt includes tools
    assert json.dumps(PLANNING_SCHEMA, indent=2) in messages[0]["content"] # System prompt includes schema 