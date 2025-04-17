import pytest
import logging
import json
from unittest.mock import patch, MagicMock
from typing import List, Optional, AsyncGenerator

# Assuming the core modules are importable relative to the tests directory
from a3x.core.planner import generate_plan, PLANNER_SYSTEM_PROMPT
from a3x.core.llm_interface import (
    LLMInterface,
)  # We need the class for type hinting/mocking

# Create a logger for tests (can be basic)
logger = logging.getLogger("test_planner")
logger.setLevel(logging.DEBUG)


# Helper to create async generator from a string
async def string_to_async_gen(text: str) -> AsyncGenerator[str, None]:
    yield text


# Mock LLM Interface
class MockLLMInterface(LLMInterface):  # Inherit for type consistency if needed
    def __init__(
        self,
        response_text: str = "",
        raise_exception: Optional[Exception] = None,
        stream: bool = False,
    ):
        self.response_text = response_text
        self.raise_exception = raise_exception
        self.stream = stream
        self.last_call_messages = None

    async def call_llm(
        self, messages: List[dict], stream: bool = False, **kwargs
    ) -> AsyncGenerator[str, None]:
        self.last_call_messages = messages  # Store messages for assertion
        if self.raise_exception:
            raise self.raise_exception

        # Simulate streaming vs non-streaming based on mock setup or call arg
        # In this simplified mock, we ignore the 'stream' argument passed to call_llm
        # and rely on the mock's configured behavior, but a more complex mock could handle it.
        # For simplicity, we just return the full text as one chunk.
        yield self.response_text


# --- Test Cases ---


@pytest.mark.asyncio
@patch("a3x.core.planner.build_planning_prompt")  # Mock the prompt builder
async def test_generate_plan_success(mock_build_prompt):
    """Tests successful plan generation with valid JSON response."""
    objective = "Test objective"
    tools = "Tool description"
    expected_plan = ["step 1", "step 2"]
    llm_response = f"```json\n{json.dumps(expected_plan)}\n```"

    mock_llm = MockLLMInterface(response_text=llm_response)
    mock_build_prompt.return_value = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
    ]

    plan = await generate_plan(objective, tools, logger, mock_llm)

    assert plan == expected_plan
    mock_build_prompt.assert_called_once_with(
        objective=objective,
        tool_descriptions=tools,
        planner_system_prompt=PLANNER_SYSTEM_PROMPT,
        heuristics_context=None,
    )
    assert mock_llm.last_call_messages == mock_build_prompt.return_value


@pytest.mark.asyncio
@patch("a3x.core.planner.build_planning_prompt")
async def test_generate_plan_llm_error(mock_build_prompt):
    """Tests plan generation when the LLM call raises an exception."""
    objective = "Test objective"
    tools = "Tool description"
    mock_llm = MockLLMInterface(raise_exception=Exception("LLM API Error"))
    mock_build_prompt.return_value = [{"role": "user", "content": "..."}]
    mock_logger = MagicMock(spec=logging.Logger)

    plan = await generate_plan(objective, tools, mock_logger, mock_llm)

    assert plan is None
    mock_logger.exception.assert_called_once()
    # Optional: Check the exception message if needed
    # args, kwargs = mock_logger.exception.call_args
    # assert "LLM API Error" in args[0]


@pytest.mark.asyncio
@patch("a3x.core.planner.build_planning_prompt")
async def test_generate_plan_invalid_json(mock_build_prompt):
    """Tests plan generation when the LLM returns invalid JSON."""
    objective = "Test objective"
    tools = "Tool description"
    llm_response = "This is not JSON"
    mock_llm = MockLLMInterface(response_text=llm_response)
    mock_build_prompt.return_value = [{"role": "user", "content": "..."}]
    mock_logger = MagicMock(spec=logging.Logger)

    plan = await generate_plan(objective, tools, mock_logger, mock_llm)

    assert plan is None
    # It might call error multiple times during parsing attempts
    assert mock_logger.error.call_count > 0
    # Example: Check the first error message
    # args, kwargs = mock_logger.error.call_args_list[0]
    # assert "No JSON block found" in args[0]


@pytest.mark.asyncio
@patch("a3x.core.planner.build_planning_prompt")
async def test_generate_plan_empty_list(mock_build_prompt):
    """Tests plan generation when the LLM returns an empty list."""
    objective = "Test objective"
    tools = "Tool description"
    llm_response = "```json\n[]\n```"
    mock_llm = MockLLMInterface(response_text=llm_response)
    mock_build_prompt.return_value = [{"role": "user", "content": "..."}]
    mock_logger = MagicMock(spec=logging.Logger)

    plan = await generate_plan(objective, tools, mock_logger, mock_llm)

    assert plan == []
    mock_logger.warning.assert_called_once()
    # Optional: Check the warning message
    # args, kwargs = mock_logger.warning.call_args
    # assert "empty plan list" in args[0]


@pytest.mark.asyncio
@patch("a3x.core.planner.build_planning_prompt")
async def test_generate_plan_non_list_json(mock_build_prompt):
    """Tests plan generation when the LLM returns valid JSON that is not a list."""
    objective = "Test objective"
    tools = "Tool description"
    llm_response = '```json\n{"not": "a list"}\n```'
    mock_llm = MockLLMInterface(response_text=llm_response)
    mock_build_prompt.return_value = [{"role": "user", "content": "..."}]
    mock_logger = MagicMock(spec=logging.Logger)

    plan = await generate_plan(objective, tools, mock_logger, mock_llm)

    assert plan is None
    mock_logger.error.assert_called_once()
    # Optional: Check the error message
    # args, kwargs = mock_logger.error.call_args
    # assert "not a list" in args[0]


@pytest.mark.asyncio
@patch("a3x.core.planner.build_planning_prompt")
async def test_generate_plan_list_with_non_strings(mock_build_prompt):
    """Tests plan generation when the LLM returns a list containing non-strings."""
    objective = "Test objective"
    tools = "Tool description"
    llm_response = '```json\n["step 1", 2, "step 3"]\n```'
    mock_llm = MockLLMInterface(response_text=llm_response)
    mock_build_prompt.return_value = [{"role": "user", "content": "..."}]
    mock_logger = MagicMock(spec=logging.Logger)

    plan = await generate_plan(objective, tools, mock_logger, mock_llm)

    assert plan is None
    mock_logger.error.assert_called_once()
    # Optional: Check the error message
    # args, kwargs = mock_logger.error.call_args
    # assert "non-string elements" in args[0]


# Add more tests as needed, e.g., for heuristics_context
