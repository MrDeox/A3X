# tests/test_agent_run_errors.py
import pytest
import json
from unittest.mock import AsyncMock, ANY

# <<< IMPORT parse_llm_response >>>
from a3x.core.agent_parser import parse_llm_response

# Import necessary components
# Import exception types if needed

# Assuming ReactAgent, core modules are importable via tests/conftest.py setup
# from core.agent import ReactAgent # Keep importing ReactAgent
# from core.agent import MaxIterationsError # Remove direct import

# <<< IMPORT CONSTANTS FOR CORRECT URL >>>
from tests.conftest import TEST_SERVER_HOST, TEST_SERVER_PORT

# Add a fixture for the mock URL
# @pytest.fixture
# def mock_llm_url():
#    \"\"\"Provides the mock LLM URL specifically for error tests.\"\"\"
#    # This URL should point to a mock server designed to return errors or specific responses
#    return \"http://mock-llm-errors/v1/chat/completions\"

# Marker for integration tests
integration_marker = pytest.mark.integration

# --- Fixtures ---


@pytest.fixture
def INVALID_JSON_STRING():
    return "This is not valid JSON { unmatched_bracket"


@pytest.fixture
def LLM_JSON_RESPONSE_LIST_FILES():  # Needed for max_iterations test
    return json.dumps(
        {
            "Thought": "I need to list files in the current directory.",
            "Action": "list_directory",
            "Action Input": {"directory": "."},
        }
    )


# --- Error Handling Specific Tests ---


@integration_marker
@pytest.mark.asyncio
async def test_react_agent_run_handles_parsing_error(
    agent_instance, mock_db, mocker, INVALID_JSON_STRING
):
    """Testa se o agente lida com erro de parsing na resposta do LLM."""
    mock_save_state = mocker.patch("a3x.core.agent.save_agent_state", return_value=None)
    agent = agent_instance
    objective = "Test objective"

    # <<< NEW: Mock _process_llm_response directly on the agent instance >>>
    mock_process_response = mocker.patch.object(
        agent,  # Patch the method on the specific agent instance
        "_process_llm_response",
        new_callable=AsyncMock,
        # Simulate the return value when parsing fails inside the method
        return_value={
            "type": "error",
            "content": "Failed to parse LLM response: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",  # Simulate a specific JSONDecodeError message
        },
    )

    # Mock reflector to return appropriate advice for parsing error
    # F841: mock_reflector = mocker.patch(
    mocker.patch(
        "a3x.core.agent_reflector.reflect_on_observation",
        new_callable=AsyncMock,
        return_value=("stop_processing", "Invalid JSON response from LLM"),
    )

    # Mock planner to avoid planning phase issues
    mocker.patch("a3x.core.agent.generate_plan", return_value=["Step 1"])

    # Execute
    results = []
    final_event = None
    try:
        async for result in agent.run(objective):
            results.append(result)
            final_event = result  # Keep track of the last yielded event
    except Exception:
        # We don't expect an exception here, but maybe a specific error type later
        # For parsing failure, the agent yields an error dict
        pass

    # Assertions
    mock_process_response.assert_awaited_once()  # Check that our patched method was awaited

    # <<< ADJUST ASSERTIONS: Check the final *summarized* response >>>
    assert final_event is not None, "Agent loop finished without yielding a final event"
    assert (
        final_event.get("type") == "final_answer"
    ), f"Expected final event type to be 'final_answer', but got {final_event.get('type')}"
    # Check if the summarization includes the error message stored previously
    # F841: expected_error_content = "Agent did not specify an action."  # This is the error logged when parser fails now
    # <<< ADJUST CHECK: The actual final error should be max iterations, which is yielded separately >>>
    # assert expected_error_content in final_event.get("content", ""), f"Expected error '{expected_error_content}' not found in final event: {final_event}"
    # Check that the error message from the mocked _process_llm_response is included in the final summary content
    expected_error_content = "Failed to parse LLM response: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
    final_content = final_event.get("content", "")
    assert (
        expected_error_content in final_content
    ), f"Expected error message '{expected_error_content}' not found in final content: '{final_content}'"

    # Ensure DB save is still attempted even on error
    mock_save_state.assert_called_once()


@integration_marker
@pytest.mark.asyncio
async def test_react_agent_run_handles_max_iterations(agent_instance, mock_db, mocker):
    """Testa se o agente para e YIELDS um erro ao atingir o limite TOTAL de iterações."""
    mock_save_state = mocker.patch("a3x.core.agent.save_agent_state", return_value=None)
    agent = agent_instance
    agent.max_iterations = 1  # Max iterations PER STEP
    objective = "List files repeatedly across multiple steps"
    mock_plan = ["Step 1", "Step 2", "Step 3"]  # Make plan long enough
    max_total_iterations = agent.max_iterations * len(mock_plan)  # = 3

    # Mock the planner to return the multi-step plan
    mocker.patch("a3x.core.agent.generate_plan", return_value=mock_plan)

    # <<< DEFINE LOCAL ReAct RESPONSE STRING >>>
    LLM_REACT_RESPONSE_LIST_FILES = """
Thought: I need to list files in the current directory.
Action: list_directory
Action Input: {"directory": "."}
"""

    # <<< MOCK _process_llm_response TO ALWAYS RETURN A VALID ACTION (FROM ReAct STRING) >>>
    # Parse the local ReAct string to get the structured data
    parsed_list_files_response = parse_llm_response(
        LLM_REACT_RESPONSE_LIST_FILES, agent.agent_logger
    )
    mock_process_response = mocker.patch.object(
        agent,
        "_process_llm_response",
        new_callable=AsyncMock,
        return_value={
            "thought": parsed_list_files_response[0],
            "action_name": parsed_list_files_response[1],
            "action_input": parsed_list_files_response[2],
        },
    )

    # Mock tool execution to return success
    # <<< CORRECT PATCH TARGET FOR execute_tool >>>
    # mock_executor = mocker.patch("core.tool_executor.execute_tool", ...)
    mock_executor = mocker.patch(
        "a3x.core.agent.execute_tool",  # <<< Use core.agent.execute_tool >>>
        new_callable=AsyncMock,
        return_value={
            "status": "success",
            "action": "directory_listed",
            "data": {"items": ["file1.txt", "file2.py"]},
        },
    )

    # Mock reflector to always suggest continuing
    # F841: mock_reflector = mocker.patch(
    mocker.patch(
        "a3x.core.agent_reflector.reflect_on_observation",
        new_callable=AsyncMock,
        return_value=("continue_processing", None),
    )

    # Execute and collect all yielded results (no exception expected)
    results = []
    async for result in agent.run(objective):
        results.append(result)

    # Assertions
    # Check the mocks were called up to the total limit
    assert mock_process_response.await_count == max_total_iterations
    assert mock_executor.call_count == max_total_iterations
    # <<< REFLECTOR ASSERTION MIGHT BE WRONG if agent stops before reflecting on last iter >>>
    # Let's assert it was called *at least* once, maybe less than max_total_iterations
    # <<< REMOVE REFLECTOR ASSERTION - Not called when max_iterations per step is 1 >>>
    # assert mock_reflector.call_count >= 1

    # Check the last yielded item is the max iterations error dictionary
    assert len(results) > 0, "Agent run yielded no results"
    final_event = results[-1]
    assert (
        final_event.get("type") == "final_answer"
    ), "Final event should be the summarized answer"
    # Check if the summarization includes the error message stored previously
    # F841: expected_error_content = "Agent did not specify an action."  # This is the error logged when parser fails now
    # <<< ADJUST CHECK: The actual final error should be max iterations, which is yielded separately >>>
    # assert expected_error_content in final_event.get("content", ""), f"Expected error '{expected_error_content}' not found in final event: {final_event}"
    # Assert that the specific max iterations error was yielded at some point
    max_iter_error_event = {"type": "error", "content": "Max total iterations reached."}
    assert (
        max_iter_error_event in results
    ), f"Expected max iteration error event not found in results: {results}"

    # Check DB save was called
    mock_save_state.assert_called_once()


@integration_marker
@pytest.mark.asyncio
async def test_react_agent_run_handles_failed_planning(agent_instance, mock_db, mocker):
    """Testa se o agente lida com a falha na geração do plano inicial."""
    mock_save_state = mocker.patch("a3x.core.agent.save_agent_state", return_value=None)
    agent = agent_instance  # Use configured agent
    agent.llm_url = f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}/v1/chat/completions"

    objective = "Objective that causes planning failure"
    mock_planner = mocker.patch(
        "a3x.core.agent.generate_plan", new_callable=AsyncMock, return_value=None
    )

    # <<< MOCK _process_llm_response TO RETURN THE FALLBACK ANSWER >>>
    fallback_response_content = "Could not determine steps for the complex objective."
    mock_process_response = mocker.patch.object(
        agent,
        "_process_llm_response",
        new_callable=AsyncMock,
        return_value={
            "thought": "Planning failed, I'll try the objective directly...",
            "action_name": "final_answer",
            "action_input": {"answer": fallback_response_content},
        },
    )

    # Mock reflector (might not be strictly needed if plan fails early, but good practice)
    # F841: mock_reflector = mocker.patch(
    mocker.patch(
        "a3x.core.agent_reflector.reflect_on_observation", new_callable=AsyncMock
    )

    # Execute
    results = []
    async for result in agent.run(objective):
        results.append(result)
    final_response_dict = results[-1] if results else None

    # Verificações
    # <<< ADJUSTED PLANNER ASSERTION >>>
    # We need to check the arguments used when generate_plan is called
    # The actual call is: generate_plan(objective, tool_desc, agent_logger, self.llm_url)
    # We will mock/check the objective and llm_url, use ANY for the others.

    mock_planner.assert_awaited_once_with(
        objective,  # Check the objective string
        ANY,  # Tool descriptions (complex to match exactly)
        ANY,  # Logger instance
        agent.llm_url,  # Check the LLM URL used by the agent
    )

    # Check if LLM was called (since planning failed, it should proceed with the objective)
    mock_process_response.assert_awaited_once()

    assert final_response_dict is not None
    assert final_response_dict.get("type") == "final_answer"
    assert fallback_response_content in final_response_dict.get("content", "")

    mock_save_state.assert_called_once()
