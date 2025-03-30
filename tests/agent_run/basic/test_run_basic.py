# tests/test_agent_run_basic.py
import pytest
import json
from unittest.mock import MagicMock, call, AsyncMock, patch
import logging

# Import necessary components (adjust paths if needed)
from core.agent import ReactAgent
from core.tools import TOOLS, get_tool_descriptions
from core.config import MAX_REACT_ITERATIONS
# Import exception type if needed for specific error tests, e.g.:
from requests.exceptions import HTTPError, Timeout, RequestException

# --- Mock Data Fixtures for test_react_agent_run_list_files ---

@pytest.fixture
def LLM_JSON_RESPONSE_LIST_FILES():
    """Mock LLM response suggesting the list_files tool."""
    return json.dumps({
        "thought": "The user wants to list files in the current directory. I should use the list_files tool.",
        "Action": "list_files",
        "action_input": {"directory": "."}
    })

@pytest.fixture
def LIST_FILES_RESULT_SUCCESS():
    """Mock successful result from the list_files (manage_files) tool."""
    # Simulate finding a couple of files/directories
    return {"status": "success", "files": ["file1.txt", "subdir", "file2.py"]}

@pytest.fixture
def LLM_JSON_RESPONSE_LIST_FILES_FINAL(LIST_FILES_RESULT_SUCCESS):
    """Mock LLM response providing the final answer after listing files."""
    # Escape the JSON string within the final answer content if needed,
    # or format it clearly for the LLM context.
    formatted_result = json.dumps(LIST_FILES_RESULT_SUCCESS, indent=2)
    return json.dumps({
        "thought": f"The 'list_files' tool executed successfully and returned the file list. I should present this result to the user using the final_answer tool. The result was: {formatted_result}",
        "Action": "final_answer",
        "action_input": {"answer": f"Successfully listed files in '.':\n```json\n{formatted_result}\n```"}
    })

@pytest.fixture
def mock_list_files_tool(LIST_FILES_RESULT_SUCCESS):
    """Mocks the tool_executor.execute_tool specifically for 'list_files'."""
    mock_execute = AsyncMock()
    # Configure the mock to return the success result ONLY when called with
    # tool_name='list_files' and action_input={'directory': '.'}
    def side_effect(tool_name: str, action_input: dict, tools_dict: dict, agent_logger: logging.Logger):
        if tool_name == "list_files" and action_input.get("directory") == ".":
             return LIST_FILES_RESULT_SUCCESS
        # Return a default value or raise an error for unexpected calls if needed
        return {"status": "error", "message": "Mock received unexpected tool call"}
    mock_execute.side_effect = side_effect
    return mock_execute

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
        reflector_decisions = [("continue_plan", None), ("plan_complete", None)] # Updated to return tuples
        mock_reflect.side_effect = reflector_decisions

        # 3. Mock LLM calls for the ReAct cycle within _execute_react_cycle
        mock_call_llm.side_effect = [
            LLM_JSON_RESPONSE_LIST_FILES,
            LLM_JSON_RESPONSE_LIST_FILES_FINAL,
        ]

        # 4. Run the agent
        objective = "List the files in the current directory."
        # Need agent logger for assertions
        agent_logger = logging.getLogger("core.agent") 
        agent_instance.agent_logger = agent_logger # Attach logger if not done automatically
        # Need tool descriptions for assertion
        tool_desc = get_tool_descriptions()

        final_response = await agent_instance.run(objective)

        # --- Asserts ---
        # Verificar se o plano foi gerado (Adjust args based on actual planner.generate_plan signature)
        # Expect tool descriptions string, not the tools dictionary
        mock_generate_plan.assert_awaited_once_with(objective, tool_desc, agent_instance.agent_logger, agent_instance.llm_url)

        # Verificar chamadas ao LLM (duas vezes dentro do ciclo ReAct)
        assert mock_call_llm.call_count == 2
        # (Optional: check specific LLM call args if needed)

        # Verify tool executor call (mock_list_files_tool)
        mock_list_files_tool.assert_awaited_once_with(
            tool_name="list_files",
            action_input={"directory": "."},
            tools_dict=agent_instance.tools,
            agent_logger=agent_instance.agent_logger
        )

        # Verify Reflector calls
        assert mock_reflect.call_count == 2
        # (Optional: check observation_dict passed to reflector)

        # Verify the final response (Adjust based on what run returns on success)
        final_answer_content = json.loads(LLM_JSON_RESPONSE_LIST_FILES_FINAL)["action_input"]["answer"]
        assert final_response is not None
        assert final_answer_content in final_response # Check if the final answer is in the returned string

@pytest.mark.asyncio
async def test_react_agent_run_final_answer_direct(
    agent_instance, mock_db, mocker, LLM_JSON_RESPONSE_HELLO_FINAL
):
    """Testa o caso onde o LLM retorna final_answer diretamente (agora após plano de 1 passo)."""
    agent, mock_call_llm = agent_instance

    objective = "Just say hello."
    mock_plan = [objective] # Plan with a single step
    mock_planner = mocker.patch('core.agent.planner.generate_plan', return_value=mock_plan)

    mock_call_llm.return_value = LLM_JSON_RESPONSE_HELLO_FINAL

    # Mock reflector to indicate plan completion
    mock_reflector = mocker.patch('core.agent.agent_reflector.reflect_on_observation',
                               return_value=("plan_complete", None))

    # Execute
    final_response = await agent.run(objective)

    # Verificações
    mock_planner.assert_awaited_once()
    mock_call_llm.assert_awaited_once() # Only one call expected
    assert "Hello there!" in final_response
    mock_db.assert_called_once()

# --- Error Handling Tests (Moved to separate file/class later ideally) ---

@pytest.mark.asyncio
async def test_react_agent_run_handles_llm_call_error(
    agent_instance, mock_code_tools, mock_db, mocker
):
    """Testa se o agente lida com um erro de chamada LLM (agora com plano)."""
    agent, mock_call_llm = agent_instance

    objective = "Test LLM call error handling......"
    mock_plan = ["Step 1: This step\'s LLM call will fail."]
    mock_planner = mocker.patch('core.agent.planner.generate_plan', return_value=mock_plan)

    error_message = "LLM API call failed"
    # Create a mock response object to pass to the exception
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.request = MagicMock() # Add a mock request object if needed by the code
    llm_error = HTTPError(error_message, response=mock_response)

    mock_call_llm.side_effect = llm_error

    # Mock reflector (assume it stops on LLM error for this test)
    mock_reflector = mocker.patch('core.agent.agent_reflector.reflect_on_observation',
                               return_value=("stop_plan", None))

    # Execute
    final_response = await agent.run(objective)

    # Verificações
    mock_planner.assert_awaited_once()
    mock_call_llm.assert_awaited_once() # Called once, raised error
    # Reflector is called with the error observation
    mock_reflector.assert_awaited_once()
    reflector_call_args = mock_reflector.await_args[1] # Get kwargs of the call
    assert reflector_call_args['observation_dict']['status'] == 'error'
    assert reflector_call_args['observation_dict']['action'] == 'llm_call_failed'
    assert error_message in final_response
    mock_db.assert_called_once() # Should still save state

@pytest.mark.asyncio
async def test_react_agent_run_handles_tool_execution_error(
    agent_instance, mock_code_tools, mock_db, mocker,
    LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE, EXECUTE_CODE_RESULT_ERROR # Use dict fixture
):
    """Testa se o agente lida com um erro retornado pela execução de uma tool (agora com plano e reflector)."""
    agent, mock_call_llm = agent_instance
    mock_execute = mock_code_tools # Get the mock function

    objective = "Execute este código Python: print(1/0)"
    mock_plan = ["Step 1: Execute the failing code."]
    mock_planner = mocker.patch('core.agent.planner.generate_plan', return_value=mock_plan)

    mock_call_llm.return_value = LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE

    # Mock the tool executor to return the error
    # mock_execute.return_value = EXECUTE_CODE_RESULT_ERROR
    mocker.patch('core.agent.tool_executor.execute_tool', return_value=EXECUTE_CODE_RESULT_ERROR)

    # Mock reflector to decide to stop on error
    mock_reflector = mocker.patch('core.agent.agent_reflector.reflect_on_observation',
                               return_value=("stop_plan", None))

    # Execute
    final_response = await agent.run(objective)

    # Verificações
    mock_planner.assert_awaited_once()
    mock_call_llm.assert_awaited_once() # LLM call to decide action
    # mock_execute.assert_called_once() # Check tool was called
    mock_reflector.assert_awaited_once() # Check reflector was called
    reflector_call_args = mock_reflector.await_args[1]
    assert reflector_call_args['action_name'] == 'execute_code'
    assert reflector_call_args['observation_dict'] == EXECUTE_CODE_RESULT_ERROR
    assert "Erro: Plano interrompido pelo Reflector." in final_response
    assert EXECUTE_CODE_RESULT_ERROR['data']['message'] in final_response
    mock_db.assert_called_once()
