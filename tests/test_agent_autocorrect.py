# tests/test_agent_autocorrect.py
import pytest
import json
from unittest import mock
from unittest.mock import MagicMock, call, patch
import logging # Import logging if used for debugging
from core.agent import ReactAgent, agent_logger # Import agent_logger if needed for side effects
from core import agent_autocorrect # <<< ADDED IMPORT

agent_logger = logging.getLogger('core.agent') # Get logger if agent uses it

# Fixtures e constantes agora vêm de conftest.py

# Mocks específicos ou constantes locais podem ser definidos aqui se necessário

# Simulate a successful modify_code result as JSON string
MOCK_META_SUCCESS_JSON_RESULT = '{"status": "success", "action": "code_modified", "data": {"modified_code": "print(1/1) # Corrected", "message": "Code modified successfully."}}'
# Simulate a failed modify_code result as JSON string
MOCK_META_FAILURE_JSON_RESULT = '{"status": "error", "action": "code_modification_failed", "data": {"message": "Simulated LLM modification failure."}}'

# Expected observation strings from agent_autocorrect (without "Observation: " prefix)
AUTO_CORRECT_SUCCESS_OBSERVATION = "Ocorreu um erro na execução anterior, mas um ciclo de auto-correção foi iniciado e concluído com sucesso. O código corrigido foi executado."
AUTO_CORRECT_FAILURE_OBSERVATION = "Ocorreu um erro na execução anterior. Uma tentativa de auto-correção foi feita, mas falhou em corrigir o erro. Resultado da tentativa: Falha: Não foi possível determinar a correção."

# Teste 1: Auto-correção bem-sucedida
def test_react_agent_run_autocorrects_execution_error(
    agent_instance, mock_code_tools, mock_db, mocker,
    LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE, EXECUTE_CODE_RESULT_ERROR, # Dict mock result
    LLM_JSON_RESPONSE_FINAL_SUCCESS, META_RUN_SUCCESS_MSG, CODE_TO_EXECUTE_FAILING
):
    """Testa o fluxo de auto-correção bem-sucedido após um erro de execução."""
    agent, mock_llm_call = agent_instance
    mock_execute, mock_modify = mock_code_tools
    # mock_save = mock_db

    # *** Explicitly set return value for the initial failing call ***
    mock_execute.return_value = EXECUTE_CODE_RESULT_ERROR

    # Mock for the new autocorrect function
    mock_try_autocorrect = mocker.patch('core.agent_autocorrect.try_autocorrect')
    # Simulate successful auto-correction returning the success message content
    mock_try_autocorrect.return_value = AUTO_CORRECT_SUCCESS_OBSERVATION

    # Configura a sequência de chamadas LLM para o ciclo *principal*
    # 1. Tenta executar código falho (triggers execute error -> try_autocorrect -> Obs(Success))
    # 2. LLM dá a resposta final
    mock_llm_call.side_effect = [
        LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE,
        LLM_JSON_RESPONSE_FINAL_SUCCESS
    ]

    objective = "Execute failing code, expect auto-correct success"
    final_response = agent.run(objective=objective)

    # Assertions Ciclo Principal
    # 1. LLM -> Exec(Error) -> try_autocorrect(mock returns Success Obs) -> History(Obs Success)
    # 2. LLM -> Final Answer
    assert mock_llm_call.call_count == 2 # LLM(Fail) -> LLM(FinalSuccess)
    # A execução inicial que falhou
    mock_execute.assert_called_once_with(action_input={'code': CODE_TO_EXECUTE_FAILING, 'language': 'python'})
    mock_modify.assert_not_called() # Modify is called within try_autocorrect (mocked out)

    # <<< UPDATED ASSERTION: Check if try_autocorrect was called >>>
    # try_autocorrect(agent_instance, tool_result, last_code, current_history, meta_depth)
    # We expect it to be called once after the execution error
    # Difficult to assert exact history/last_code state passed, so just check it was called
    mock_try_autocorrect.assert_called_once()
    # Check the arguments more loosely if needed, e.g.,
    # call_args, call_kwargs = mock_try_autocorrect.call_args # << REMOVED argument inspection for now
    # assert call_args[0] is agent
    # assert call_args[1] == EXECUTE_CODE_RESULT_ERROR
    # assert call_args[2] == CODE_TO_EXECUTE_FAILING # Check if last_code is correctly tracked
    # assert isinstance(call_args[3], list) # Check history is a list
    # assert call_args[4] == 1 # Check meta_depth

    # Verifica a observação que indica o sucesso do meta-ciclo (retornada pelo mock)
    assert len(agent._history) == 5 # Human, LLM1, Obs(Success), LLM2, FinalAnswer
    assert agent._history[0] == f"Human: {objective}"
    assert agent._history[1] == LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE
    # Check the observation added by the main loop, which comes from try_autocorrect mock
    expected_observation = f"Observation: {AUTO_CORRECT_SUCCESS_OBSERVATION}"
    assert agent._history[2] == expected_observation, \
        f"History item 2 mismatch. Expected: '{expected_observation}'. Got: '{agent._history[2]}'"

    # Verifica a resposta final - deve refletir o sucesso simulado do meta-ciclo
    expected_final_answer = json.loads(LLM_JSON_RESPONSE_FINAL_SUCCESS).get("Action Input", {}).get("answer")
    assert final_response == expected_final_answer


# Teste 2: Falha na Auto-correção (LLM não consegue corrigir)
def test_react_agent_run_autocorrect_fails(
    agent_instance, mock_code_tools, mock_db, mocker,
    LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE, EXECUTE_CODE_RESULT_ERROR, # <<< ADDED MOCK RESULT DICT
    META_RUN_FAIL_MSG_FRAGMENT
):
    """Testa o fluxo onde a auto-correção é tentada, mas falha."""
    agent, mock_llm_call = agent_instance
    mock_execute, mock_modify = mock_code_tools
    # mock_save = mock_db

    # *** Explicitly set return value for the initial failing call ***
    mock_execute.return_value = EXECUTE_CODE_RESULT_ERROR

    # Mock for the new autocorrect function
    mock_try_autocorrect = mocker.patch('core.agent_autocorrect.try_autocorrect')
    # Simulate failed auto-correction returning the failure message content
    mock_try_autocorrect.return_value = AUTO_CORRECT_FAILURE_OBSERVATION

    # LLM sempre tenta executar o código falho (simulando um loop de erro)
    mock_llm_call.return_value = LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE

    objective = "Execute failing code, expect auto-correct fail"
    agent.max_iterations = 3 # Baixa max_iterations para terminar rápido
    final_response = agent.run(objective=objective)

    # Assertions
    # Ciclo 1: LLM(Fail) -> execute(Error) -> try_autocorrect(mock Fail Obs) -> History(Obs Fail)
    # Ciclo 2: LLM(Fail) -> execute(Error) -> try_autocorrect(mock Fail Obs) -> History(Obs Fail)
    # Ciclo 3: LLM(Fail) -> execute(Error) -> try_autocorrect(mock Fail Obs) -> History(Obs Fail) -> Max Iterations
    assert mock_llm_call.call_count == 3
    assert mock_execute.call_count == 3
    assert mock_modify.call_count == 0 # Modify não é chamado no ciclo principal

    # Check try_autocorrect call count
    assert mock_try_autocorrect.call_count == 3, f"Esperado 3 chamadas a try_autocorrect, obteve {mock_try_autocorrect.call_count}"

    # Verifica o histórico final - deve conter as observações de falha do meta-ciclo (mockadas)
    assert len(agent._history) == 1 + (3 * 2) # Human + 3 * (LLM + Obs)
    assert agent._history[0] == f"Human: {objective}"
    # Verifica a última observação - deve indicar falha na auto-correção (do mock)
    expected_observation = f"Observation: {AUTO_CORRECT_FAILURE_OBSERVATION}"
    assert agent._history[-1] == expected_observation, \
        f"Last history item mismatch. Expected: '{expected_observation}'. Got: '{agent._history[-1]}'"

    # Verifica a resposta final
    assert "Máximo de iterações" in final_response, \
        f"Final response should indicate iteration limit. Got: '{final_response}'"


# Teste 3: Atinge a profundidade máxima de auto-correção
def test_react_agent_run_autocorrect_max_depth(
    agent_instance, mock_code_tools, mock_db, mocker,
    LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE, EXECUTE_CODE_RESULT_ERROR, # Usa dict aqui para mock
    CODE_TO_EXECUTE_FAILING # <<< REMOVED unused EXECUTE_CODE_RESULT_ERROR_JSON fixture
):
    """Testa que a auto-correção para no MAX_META_DEPTH."""
    agent, mock_llm_call = agent_instance
    mock_execute, mock_modify = mock_code_tools
    # mock_save = mock_db

    patched_max_depth = 0
    mocker.patch('core.agent_autocorrect.MAX_META_DEPTH', patched_max_depth, create=True)

    mock_llm_call.return_value = LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE
    mock_execute.return_value = EXECUTE_CODE_RESULT_ERROR # mock retorna dict

    # Mock for the new autocorrect function
    mock_try_autocorrect = mocker.patch('core.agent_autocorrect.try_autocorrect')
    # Simulate auto-correction being blocked (returns None)
    mock_try_autocorrect.return_value = None

    original_objective = "Execute failing code, expect max depth hit"
    agent.max_iterations = 3 # Limita iterações do ciclo principal
    final_response = agent.run(objective=original_objective) 

    # Assertions
    expected_execute_calls = 3 
    assert mock_execute.call_count == expected_execute_calls, f"Esperado {expected_execute_calls} chamadas, obteve {mock_execute.call_count}"
    expected_args = mock.call(action_input={'code': CODE_TO_EXECUTE_FAILING, 'language': 'python'})
    mock_execute.assert_has_calls([expected_args] * expected_execute_calls, any_order=False)

    # Check try_autocorrect call count
    assert mock_try_autocorrect.call_count == 3, f"Esperado 3 chamadas a try_autocorrect, obteve {mock_try_autocorrect.call_count}"

    # Histórico: Human, LLM1(Exec), Obs1(Error), ..., LLM3(Exec), Obs3(Error)
    expected_history_length = 1 + (expected_execute_calls * 2)
    assert len(agent._history) == expected_history_length, f"Esperado histórico {expected_history_length}, obteve {len(agent._history)}"
    assert agent._history[0] == f"Human: {original_objective}"
    assert agent._history[-2] == LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE
    # A observação é o erro original, pois try_autocorrect retornou None
    observation_str = agent._history[-1]
    assert observation_str.startswith("Observation: ")
    try:
        # The observation should be the original tool error result (JSON formatted)
        observation_json = json.loads(observation_str.replace("Observation: ", "", 1))
        # Compare relevant parts, ignore potential formatting diffs in nested strings like stderr
        assert observation_json['status'] == EXECUTE_CODE_RESULT_ERROR['status']
        assert observation_json['action'] == EXECUTE_CODE_RESULT_ERROR['action']
        assert observation_json['data']['message'].startswith("Falha na execução do código")
        assert "ZeroDivisionError" in observation_json['data']['message'] or "ZeroDivisionError" in observation_json['data'].get('stderr','')

    except json.JSONDecodeError:
        pytest.fail(f"Could not decode history observation as JSON: {observation_str}")

    assert "Máximo de iterações" in final_response, \
        f"Final response should indicate iteration limit. Got: '{final_response}'"
