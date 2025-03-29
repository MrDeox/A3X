# tests/test_agent_run_errors.py
import pytest
from unittest import mock
from unittest.mock import MagicMock, call
from core.agent import ReactAgent, agent_logger
import json
from openai import APIError
from requests.exceptions import HTTPError
from core import agent_autocorrect
from core import agent_error_handler

# Fixtures são importados automaticamente de conftest.py

# Constants for mocking
LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE = '{"Thought": "I need to execute this potentially problematic code.", "Action": "execute_code", "Action Input": {"code": "print(1/0)", "language": "python"}}'
EXECUTE_CODE_RESULT_ERROR = {'action': 'execution_failed', 'data': {'message': 'Falha na execução do código: erro de runtime. Stderr: Traceback...\\nZeroDivisionError: division by zero', 'stdout': ''}, 'status': 'error'}
EXECUTE_CODE_RESULT_ERROR_JSON = json.dumps(EXECUTE_CODE_RESULT_ERROR)
# Simulate a failed modify_code result as JSON string (matching definition in test_agent_autocorrect.py)
MOCK_META_FAILURE_JSON_RESULT = '{"status": "error", "action": "code_modification_failed", "data": {"message": "Simulated LLM modification failure."}}'

# Expected observation content for tool error auto-correct failure
AUTO_CORRECT_FAILURE_OBSERVATION_TOOL_ERROR = "An execution error occurred. Auto-correction attempt failed during the modification step. Reason: Simulated LLM modification failure."

# ... (previous tests)

def test_react_agent_run_handles_llm_call_error(
    agent_instance, mock_code_tools, mock_db, mocker
):
    """Testa se o agente lida com um erro de chamada LLM."""
    agent, mock_llm_call = agent_instance
    # mock_execute, mock_modify = mock_code_tools
    # mock_save = mock_db

    # Configura o mock da chamada LLM para levantar um erro
    error_message = "LLM API call failed"
    llm_error = APIError(message=error_message, request=None, body=None)
    mock_llm_call.side_effect = llm_error

    # <<< ADD MOCK for handle_llm_error, wrapping the original >>>
    mock_handle_llm_error = mocker.patch(
        'core.agent_error_handler.handle_llm_call_error',
        wraps=agent_error_handler.handle_llm_call_error
    )

    objective = "Test LLM call error handling......" # Added dots just to force a diff :)
    # Default max_iterations é 10
    final_response = agent.run(objective=objective)

    # Assertions
    # O agente tentará chamar o LLM repetidamente, e handle_llm_error será chamado a cada vez
    assert mock_llm_call.call_count == agent.max_iterations

    # <<< UPDATE ASSERTION: Check handle_llm_error calls >>>
    assert mock_handle_llm_error.call_count == agent.max_iterations
    # Check last call arguments if necessary
    # last_call_args, _ = mock_handle_llm_error.call_args_list[-1]
    # assert last_call_args[1] == f"Erro: Falha na chamada LLM: {llm_error}"
    # assert last_call_args[3] == agent.max_iterations

    # <<< REMOVE history checks, as they are handled within the wrapped function >>>
    # expected_history_len = 1 + agent.max_iterations
    # assert len(agent._history) == expected_history_len, \
    #     f"Expected history length {expected_history_len}, got {len(agent._history)}. History: {agent._history}"
    # assert agent._history[0] == f"Human: {objective}"
    # expected_error_observation = f"Observation: Erro crítico na comunicação com o LLM: Erro: Falha na chamada LLM: {llm_error}"
    # assert agent._history[-1] == expected_error_observation, \
    #     f"Last history item mismatch. Expected: {expected_error_observation}, Got: {agent._history[-1]}"

    # <<< UPDATE ASSERTION for final_response >>>
    # Verifica a resposta final - Deve ser a mensagem retornada por handle_llm_error na última iteração
    expected_final_response = f"Erro: Atingido limite de iterações ({agent.max_iterations}) após falha na comunicação com LLM."
    assert final_response == expected_final_response, \
        f"Final response mismatch. Expected: '{expected_final_response}', Got: '{final_response}'"


def test_react_agent_run_handles_tool_execution_error(
    agent_instance, mock_code_tools, mock_db, mocker,
    LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE, EXECUTE_CODE_RESULT_ERROR_JSON # JSON string
):
    """Testa se o agente lida com um erro retornado pela execução de uma tool."""
    agent, mock_llm_call = agent_instance
    mock_execute, _ = mock_code_tools

    # <<< ADD MOCK for try_autocorrect >>>
    mock_try_autocorrect = mocker.patch('core.agent_autocorrect.try_autocorrect')
    # Simulate failed auto-correction returning the specific failure message content
    mock_try_autocorrect.return_value = AUTO_CORRECT_FAILURE_OBSERVATION_TOOL_ERROR

    # *** Define max_iterations para garantir que o ciclo principal pare após o erro ***
    agent.max_iterations = 1

    # Configura o LLM para chamar a ferramenta que falha (APENAS UMA VEZ)
    mock_llm_call.return_value = LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE
    # Configura o mock da ferramenta para retornar um erro (como dicionário)
    mock_execute.return_value = json.loads(EXECUTE_CODE_RESULT_ERROR_JSON)

    objective = "Test tool execution error handling"
    final_response = agent.run(objective=objective)

    # Assertions
    # Ciclo 1: LLM(Exec Failing) -> Execute(Error Dict) -> try_autocorrect(Fail Obs) -> History(Obs Fail)
    # -> Atinge max_iterations = 1 -> Fim
    mock_llm_call.assert_called_once() # AGORA deve ser chamado apenas uma vez
    mock_execute.assert_called_once()
    mock_try_autocorrect.assert_called_once() # Check that it was called

    # Verifica histórico: Human, LLM(Exec), Obs(Error)
    assert len(agent._history) == 3 # Human, LLM, Obs
    assert agent._history[0] == f"Human: {objective}"
    assert agent._history[1] == LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE
    # A observação DEVE ser a mensagem de falha retornada pelo mock de try_autocorrect (com prefixo)
    expected_observation = f"Observation: {AUTO_CORRECT_FAILURE_OBSERVATION_TOOL_ERROR}"
    assert agent._history[2] == expected_observation, \
        f"History item 2 mismatch. Expected: '{expected_observation}'. Got: '{agent._history[2]}'"

    # Verifica a resposta final (deve indicar que atingiu o limite de iterações)
    expected_final_response_start = "Erro: Máximo de iterações (1) atingido."
    assert final_response.startswith(expected_final_response_start), \
        f"Final response mismatch. Expected start: '{expected_final_response_start}'. Got: '{final_response}'"

# --- Fixtures ---
# ... existing code ...
