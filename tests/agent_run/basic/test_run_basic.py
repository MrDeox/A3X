# tests/test_agent_run_basic.py
import pytest
import json
from unittest.mock import MagicMock, call, AsyncMock, patch
import logging
import requests

# Import necessary components (adjust paths if needed)
from core.agent import ReactAgent
from core.tools import get_tool_descriptions
from core.config import MAX_REACT_ITERATIONS
# Import exception type if needed for specific error tests, e.g.:
from requests.exceptions import HTTPError, Timeout, RequestException

# Add a fixture for the mock URL
@pytest.fixture
def mock_llm_url():
    return "http://mock-llm-basic/v1/chat/completions"

# --- Mock Data Fixtures for test_react_agent_run_list_files ---

@pytest.fixture
def LLM_JSON_RESPONSE_LIST_FILES():
    """Mock LLM response suggesting the list_files tool."""
    return json.dumps({
        "thought": "The user wants to list files. I should use list_files.",
        "Action": "list_files",
        "action_input": {"directory": "."}
    })

@pytest.fixture
def LIST_FILES_RESULT_SUCCESS():
    """Mock successful result from the list_files tool."""
    return {"status": "success", "action": "directory_listed", "data": {"files": ["file1.txt", "subdir/"], "message": "Found 2 items."}}

@pytest.fixture
def LLM_JSON_RESPONSE_LIST_FILES_FINAL(LIST_FILES_RESULT_SUCCESS):
    """Mock LLM response providing the final answer after listing files."""
    formatted_result = json.dumps(LIST_FILES_RESULT_SUCCESS['data']['files'], indent=2)
    return json.dumps({
        "thought": f"The tool worked. Result: {formatted_result}. I will report this.",
        "Action": "final_answer",
        "action_input": {"answer": f"Found files:\n{formatted_result}"}
    })

@pytest.fixture
def mock_list_files_tool(LIST_FILES_RESULT_SUCCESS):
    """Mocks the tool_executor.execute_tool specifically for 'list_files'."""
    async def mock_execute(tool_name: str, action_input: dict, tools_dict: dict, agent_logger: logging.Logger):
        if tool_name == "list_files": # Simplified check for the basic test
             return LIST_FILES_RESULT_SUCCESS
        return {"status": "error", "message": "Mock received unexpected tool call"}
    return AsyncMock(side_effect=mock_execute)

# --- Fixture for test_react_agent_run_final_answer_direct ---
@pytest.fixture
def LLM_JSON_RESPONSE_HELLO_FINAL():
    return json.dumps({
        "thought": "Objective is simple, just say hello.",
        "Action": "final_answer",
        "action_input": {"answer": "Hello there!"}
    })

# --- Fixtures for test_react_agent_run_handles_llm_call_error ---
# (No specific fixtures needed other than agent_instance, mock_db, mocker)

# --- Fixtures for test_react_agent_run_handles_tool_execution_error ---
@pytest.fixture
def LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE():
    return json.dumps({
        "thought": "User wants to execute risky code.",
        "Action": "execute_code",
        "action_input": {"code": "print(1/0)", "language": "python"}
    })

@pytest.fixture
def EXECUTE_CODE_RESULT_ERROR():
    """Simulates error from execute_code tool"""
    return {
        "status": "error",
        "action": "execute_code_failed",
        "data": {
            "message": "Erro ao executar código python: division by zero",
            "stdout": "",
            "stderr": "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nZeroDivisionError: division by zero"
        }
    }

# Mock fixture for code tools if needed (can be empty if execute_tool is patched directly)
@pytest.fixture
def mock_code_tools():
    return MagicMock() # Placeholder

# --- Basic Agent Run Tests ---

@pytest.mark.asyncio
async def test_react_agent_run_list_files(
    agent_instance, # Uses the agent_instance global from conftest.py
    managed_llama_server, # Uses the server fixture from conftest.py
    LLM_JSON_RESPONSE_LIST_FILES,
    LLM_JSON_RESPONSE_LIST_FILES_FINAL,
    mock_list_files_tool, # Mock for execute_tool
    LIST_FILES_RESULT_SUCCESS # Used by LLM_JSON_RESPONSE_LIST_FILES_FINAL and mock_list_files_tool
):
    """
    Tests a simple run where the agent lists files using the manage_files tool.
    Verifies the flow: LLM -> Tool -> LLM -> Final Answer.
    Mocks the LLM calls and the tool execution via patching module functions.
    """
    # Mock das chamadas de função relevantes nos módulos core
    # Patching a função `call_llm` no módulo `core.agent`
    # Patching a função `execute_tool` no módulo `core.tool_executor`
    # Patching a função `generate_plan` no módulo `core.planner`
    # Patching a função `reflect_on_observation` no módulo `core.agent_reflector`
    with patch('core.agent.call_llm', new_callable=AsyncMock) as mock_call_llm, \
         patch('core.tool_executor.execute_tool', new=mock_list_files_tool), \
         patch('core.planner.generate_plan', new_callable=AsyncMock) as mock_generate_plan, \
         patch('core.agent_reflector.reflect_on_observation', new_callable=AsyncMock) as mock_reflect:

        # 1. Mock Planning: Assume planner generates a simple plan
        mock_generate_plan.return_value = ["Step 1: List files in the current directory.", "Step 2: Provide the final answer."]

        # 2. Mock Reflector: Continue after tool, complete after final answer
        async def mock_reflect_side_effect(*args, **kwargs):
            yield ("continue_plan", None)
            yield ("plan_complete", None)
        # Reflector should also be an async generator if it uses await internally
        # For this test, assume it returns directly, but use async mock for consistency
        mock_reflect.side_effect = mock_reflect_side_effect()

        # 3. Mock LLM calls for the ReAct cycle within _execute_react_cycle
        async def mock_llm_responses():
            yield LLM_JSON_RESPONSE_LIST_FILES
            yield LLM_JSON_RESPONSE_LIST_FILES_FINAL
        mock_call_llm.return_value = mock_llm_responses()

        # 4. Run the agent
        objective = "List the files in the current directory."
        # Need agent logger for assertions
        agent_logger = logging.getLogger("core.agent") 
        agent_instance.agent_logger = agent_logger # Attach logger if not done automatically
        # Need tool descriptions for assertion
        tool_desc = get_tool_descriptions()

        results = []
        async for result in agent_instance.run(objective):
            results.append(result)
        final_response_dict = results[-1] if results else None # Get the last yielded item

        # --- Asserts ---
        # mock_generate_plan.assert_awaited_once_with(objective, tool_desc, agent_instance.agent_logger, agent_instance.llm_url) # Should be called, not awaited if generator
        mock_generate_plan.assert_called_once_with(objective, tool_desc, agent_instance.agent_logger, agent_instance.llm_url)
        assert mock_call_llm.call_count == 2 # ReAct cycle calls
        
        # Check the tool was called correctly. Arguments for list_files are passed as kwargs
        # mock_list_files_tool.assert_awaited_once_with(directory='.') 
        mock_list_files_tool.assert_called_once_with(directory='.') 

        # Check reflector calls
        assert mock_reflect.call_count == 2

        # Verify the final response (Adjust based on what run returns on success)
        final_answer_content = json.loads(LLM_JSON_RESPONSE_LIST_FILES_FINAL)["action_input"]["answer"]
        assert final_response_dict is not None
        assert final_answer_content in final_response_dict # Check if the final answer is in the returned string

@pytest.mark.asyncio
async def test_react_agent_run_final_answer_direct(
    agent_instance, mock_db, mocker, LLM_JSON_RESPONSE_HELLO_FINAL
):
    """Testa o caso onde o LLM retorna final_answer diretamente (agora após plano de 1 passo)."""
    agent = agent_instance # Use the configured agent

    objective = "Just say hello."
    mock_plan = [objective] # Plan with a single step
    # Use AsyncMock for planner as it's async
    mock_planner = mocker.patch('core.planner.generate_plan', new_callable=AsyncMock, return_value=mock_plan)

    # Patch call_llm within this test's context
    with patch('core.agent.call_llm', new_callable=AsyncMock) as mock_call_llm:
        async def mock_llm_response():
            yield LLM_JSON_RESPONSE_HELLO_FINAL
        mock_call_llm.return_value = mock_llm_response()

    # Mock reflector to indicate plan completion
    # Use AsyncMock
    mock_reflector = mocker.patch('core.agent_reflector.reflect_on_observation', new_callable=AsyncMock,
                               return_value=("plan_complete", None))

    # Execute
    results = []
    async for result in agent.run(objective):
        results.append(result)
    final_response = results[-1] if results else None

    # Verificações
    mock_planner.assert_called_once()
    mock_call_llm.assert_called_once() # Only one call expected
    mock_reflector.assert_called_once()
    assert "Hello there!" in final_response
    mock_db.assert_called_once()

@pytest.mark.asyncio
async def test_react_agent_run_handles_llm_call_error(
    agent_instance, mock_code_tools, mock_db, mocker
):
    """Testa se o agente lida com um erro de chamada LLM (agora com plano)."""
    agent = agent_instance

    objective = "Test LLM call error handling......"
    mock_plan = ["Step 1: This step\'s LLM call will fail."]
    # Use AsyncMock
    mock_planner = mocker.patch('core.planner.generate_plan', new_callable=AsyncMock, return_value=mock_plan)

    error_message = "LLM API call failed"
    # Create a mock response object to pass to the exception
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.request = MagicMock() # Add a mock request object if needed by the code
    llm_error = HTTPError(error_message, response=mock_response)

    # Patch call_llm within this test's context
    with patch('core.agent.call_llm', new_callable=AsyncMock) as mock_call_llm:
        # Define an async generator function that raises the error then yields
        async def mock_llm_raise_error():
            raise llm_error
            yield # Necessary to be an async generator
        # Set side_effect to the generator function itself
        mock_call_llm.side_effect = mock_llm_raise_error

    # Mock reflector (assume it stops on LLM error for this test)
    # Use AsyncMock
    mock_reflector = mocker.patch('core.agent_reflector.reflect_on_observation', new_callable=AsyncMock,
                               return_value=("stop_plan", None))

    # Execute
    results = []
    async for result in agent.run(objective):
        results.append(result)
    final_response = results[-1] if results else None

    # Verificações
    mock_planner.assert_called_once()
    mock_call_llm.assert_called_once() # Called once, raised error
    # Reflector is called with the error observation
    mock_reflector.assert_called_once()
    # Check call_args directly for non-async mocks, or handle async mock args if reflector is async
    reflector_call_args = mock_reflector.call_args.kwargs # Get kwargs of the call
    assert reflector_call_args.get('observation_dict', {}).get('status') == 'error'
    assert reflector_call_args.get('observation_dict', {}).get('action') == 'llm_call_failed'
    assert error_message in final_response
    mock_db.assert_called_once() # Should still save state

@pytest.mark.asyncio
async def test_react_agent_run_handles_tool_execution_error(
    agent_instance, mock_code_tools, mock_db, mocker,
    LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE, EXECUTE_CODE_RESULT_ERROR # Use dict fixture
):
    """Testa se o agente lida com um erro retornado pela execução de uma tool (agora com plano e reflector)."""
    agent = agent_instance
    mock_execute = mock_code_tools # Get the mock function

    objective = "Execute este código Python: print(1/0)"
    mock_plan = ["Step 1: Execute the failing code."]
    # Use AsyncMock
    mock_planner = mocker.patch('core.planner.generate_plan', new_callable=AsyncMock, return_value=mock_plan)

    # Patch call_llm within this test's context
    with patch('core.agent.call_llm', new_callable=AsyncMock) as mock_call_llm:
        async def mock_llm_response():
            yield LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE
        mock_call_llm.return_value = mock_llm_response()

    # Mock the tool executor to return the error
    mock_tool_executor = mocker.patch('core.agent.tool_executor.execute_tool', return_value=EXECUTE_CODE_RESULT_ERROR)

    # Mock reflector to decide to stop on error
    # Use AsyncMock
    mock_reflector = mocker.patch('core.agent_reflector.reflect_on_observation', new_callable=AsyncMock,
                               return_value=("stop_plan", None))

    # Execute
    results = []
    async for result in agent.run(objective):
        results.append(result)
    final_response = results[-1] if results else None

    # Verificações
    mock_planner.assert_called_once()
    mock_call_llm.assert_called_once() # LLM call to decide action
    mock_tool_executor.assert_called_once() # Check tool was called
    mock_reflector.assert_called_once() # Check reflector was called
    reflector_call_args = mock_reflector.call_args.kwargs
    assert reflector_call_args.get('action_name') == 'execute_code'
    assert reflector_call_args.get('observation_dict') == EXECUTE_CODE_RESULT_ERROR
    assert "Erro: Plano interrompido pelo Reflector." in final_response
    assert EXECUTE_CODE_RESULT_ERROR['data']['message'] in final_response
    mock_db.assert_called_once()
