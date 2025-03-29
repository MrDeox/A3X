# tests/test_agent_run_errors.py
import pytest
from unittest import mock
from unittest.mock import MagicMock, call, patch
from core.agent import ReactAgent, agent_logger, AGENT_STATE_ID
from core.db_utils import save_agent_state, load_agent_state
from core import planner
import json
from openai import APIError
from requests.exceptions import HTTPError

# Fixtures são importados automaticamente de conftest.py

# Constants for mocking
LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE = '{"Thought": "I need to execute this potentially problematic code.", "Action": "execute_code", "Action Input": {"code": "print(1/0)", "language": "python"}}'
EXECUTE_CODE_RESULT_ERROR = {'action': 'execution_failed', 'data': {'message': 'Falha na execução do código: erro de runtime. Stderr: Traceback...\\nZeroDivisionError: division by zero', 'stdout': ''}, 'status': 'error'}
EXECUTE_CODE_RESULT_ERROR_JSON = json.dumps(EXECUTE_CODE_RESULT_ERROR)
# Simulate a failed modify_code result as JSON string (matching definition in test_agent_autocorrect.py)
MOCK_META_FAILURE_JSON_RESULT = '{"status": "error", "action": "code_modification_failed", "data": {"message": "Simulated LLM modification failure."}}'

# Expected observation content for tool error auto-correct failure
# AUTO_CORRECT_FAILURE_OBSERVATION_TOOL_ERROR = "An execution error occurred. Auto-correction attempt failed during the modification step. Reason: Simulated LLM modification failure."

# ... (previous tests)

def test_react_agent_run_handles_llm_call_error(
    agent_instance, mock_code_tools, mock_db, mocker
):
    """Testa se o agente lida com um erro de chamada LLM (agora com plano)."""
    agent, mock_llm_call = agent_instance
    # mock_execute, mock_modify = mock_code_tools
    # mock_save = mock_db

    # <<< Mock the planner >>>
    objective = "Test LLM call error handling......"
    mock_plan = ["Step 1: This step's LLM call will fail."]
    mock_planner = mocker.patch('core.agent.planner.generate_plan')
    mock_planner.return_value = mock_plan

    # Configura o mock da chamada LLM para levantar um erro na primeira chamada
    error_message = "LLM API call failed"
    llm_error = APIError(message=error_message, request=None, body=None)
    mock_llm_call.side_effect = llm_error

    # Execute the agent
    final_response = agent.run(objective=objective)

    # Assertions
    mock_planner.assert_called_once() # Planner was called
    # O agente tentará chamar o LLM UMA VEZ para o único passo do plano
    assert mock_llm_call.call_count == 1 

    # Verifica se o agente termina e retorna a mensagem de erro esperada
    # The cycle now returns the specific LLM error when the reflector decides to stop
    assert "Erro: Falha ao comunicar com LLM" in final_response
    assert "LLM API call failed" in final_response # Check specific error message

    # Check History (Human, Observation)
    assert len(agent._history) == 2, f"History: {agent._history}"
    assert agent._history[0] == f"Human: {objective}"
    assert agent._history[1].startswith("Observation: {")
    assert '"status": "error"' in agent._history[1]
    assert '"action": "llm_call_failed"' in agent._history[1]
    assert "Erro: Falha na chamada LLM: LLM API call failed" in agent._history[1] # Check specific msg within data
    
    # --- REMOVED old assertions checking for max_iterations or specific error format ---

def test_react_agent_run_handles_tool_execution_error(
    agent_instance, mock_code_tools, mock_db, mocker,
    LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE, EXECUTE_CODE_RESULT_ERROR_JSON # JSON string
):
    """Testa se o agente lida com um erro retornado pela execução de uma tool (agora com plano)."""
    agent, mock_llm_call = agent_instance
    mock_execute = mock_code_tools

    # <<< Mock the planner >>>
    objective = "Execute este código Python: print(1/0)"
    mock_plan = ["Step 1: Execute the failing code."]
    mock_planner = mocker.patch('core.agent.planner.generate_plan')
    mock_planner.return_value = mock_plan

    # Configura o mock da chamada LLM para retornar a ação de executar código falho (para o único passo)
    mock_llm_call.return_value = LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE

    # Configura o mock da ferramenta execute_code para retornar um erro
    mock_execute.return_value = json.loads(EXECUTE_CODE_RESULT_ERROR_JSON)

    # Executa o agente
    result = agent.run(objective)

    # Assertions
    mock_planner.assert_called_once()
    # Verifica se a chamada LLM foi feita UMA VEZ (para o único passo)
    assert mock_llm_call.call_count == 1
    # Verifica se a ferramenta foi chamada UMA VEZ
    mock_execute.assert_called_once()

    # Verifica o histórico da conversa (Human, LLM1, Obs1)
    assert len(agent._history) == 3, f"History: {agent._history}"
    assert agent._history[0] == f"Human: {objective}"
    assert LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE in agent._history[1] # Raw LLM JSON
    # Check Observation
    try:
        actual_obs_json_str = agent._history[2].removeprefix("Observation: ").strip()
        actual_obs_obj = json.loads(actual_obs_json_str)
        expected_obs_obj = json.loads(EXECUTE_CODE_RESULT_ERROR_JSON)
        assert actual_obs_obj == expected_obs_obj, f"Observation JSON mismatch. Expected: {expected_obs_obj}, Got: {actual_obs_obj}"
    except (IndexError, json.JSONDecodeError) as e:
        pytest.fail(f"Failed to validate history observation [2]: {e}. History: {agent._history}")

    # Verifica se o agente termina e retorna a mensagem de erro esperada
    # The reflector currently returns 'stop_plan' for execution_failed
    assert "Erro: Plano interrompido pelo Reflector" in result
    # Check for key parts of the original error detail propagated by the reflector
    assert "Falha na execução do código" in result
    assert "ZeroDivisionError: division by zero" in result

# --- Fixtures ---
# ... existing code ...
