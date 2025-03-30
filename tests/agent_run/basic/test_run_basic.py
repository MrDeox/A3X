# tests/test_agent_run_basic.py
import pytest
import json
from unittest.mock import MagicMock, call, AsyncMock

# Import necessary components (adjust paths if needed)
from core.agent import ReactAgent
from core.tools import TOOLS
from core.config import MAX_REACT_ITERATIONS
# Import exception type if needed for specific error tests, e.g.:
from requests.exceptions import HTTPError, Timeout, RequestException

# --- Basic Agent Run Tests ---

@pytest.mark.asyncio
async def test_react_agent_run_list_files(
    agent_instance, mock_db, mocker,
    LLM_JSON_RESPONSE_LIST_FILES, LLM_JSON_RESPONSE_LIST_FILES_FINAL,
    LIST_FILES_RESULT_SUCCESS, mock_list_files_tool # Use fixtures
):
    """Testa um fluxo básico com a ferramenta list_files e final_answer, agora com planejamento."""
    agent, mock_call_llm = agent_instance 

    objective = "Liste os arquivos no diretório atual."
    mock_plan = ["Step 1: List files in current directory.", "Step 2: Provide the final answer."]
    mock_planner = mocker.patch('core.agent.planner.generate_plan', return_value=mock_plan)

    # Configure side effects for the two LLM calls needed
    mock_call_llm.side_effect = [
        LLM_JSON_RESPONSE_LIST_FILES,       # Response for step 1
        LLM_JSON_RESPONSE_LIST_FILES_FINAL  # Response for step 2
    ]

    # Tool mock already done by mock_list_files_tool fixture
    mock_tool_func = mock_list_files_tool # Get the mock function from fixture

    # Execute o run
    final_response = await agent.run(objective)

    # Verificações
    mock_planner.assert_awaited_once_with(objective, mocker.ANY, mocker.ANY, agent.llm_url)
    assert mock_call_llm.await_count == 2
    mock_tool_func.assert_called_once_with(action_input={'directory': '.'})
    assert "Files listed: mock_file.txt" in final_response
    mock_db.assert_called_once() # Check if state was saved

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
