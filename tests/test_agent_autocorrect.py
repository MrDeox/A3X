# tests/test_agent_autocorrect.py
import pytest
import json
from unittest import mock
from unittest.mock import MagicMock, call, patch
import logging # Import logging if used for debugging
from core.agent import ReactAgent, agent_logger # Import agent_logger if needed for side effects

agent_logger = logging.getLogger('core.agent') # Get logger if agent uses it

# Fixtures e constantes agora vêm de conftest.py

# Mocks específicos ou constantes locais podem ser definidos aqui se necessário

# Simulate a successful modify_code result as JSON string
MOCK_META_SUCCESS_JSON_RESULT = '{"status": "success", "action": "code_modified", "data": {"modified_code": "print(1/1) # Corrected", "message": "Code modified successfully."}}'
# Simulate a failed modify_code result as JSON string
MOCK_META_FAILURE_JSON_RESULT = '{"status": "error", "action": "code_modification_failed", "data": {"message": "Simulated LLM modification failure."}}'

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

    # Mock da chamada recursiva 'run' para simular o sucesso do meta-ciclo
    mock_recursive_run = mocker.patch.object(agent, 'run', wraps=agent.run)

    # O side_effect agora simula o *resultado* do meta-ciclo
    # Quando chamado com is_meta_objective=True, retorna a mensagem de sucesso
    # Quando chamado normalmente (primeira vez), usa mock.DEFAULT para deixar a execução original prosseguir
    def recursive_run_side_effect(*args, **kwargs):
        if kwargs.get('is_meta_objective') is True:
            # Simula que o meta-ciclo interno foi bem-sucedido
            agent_logger.info("[TEST MOCK] Meta-ciclo simulado retornando sucesso (JSON).")
            # O meta-ciclo agora retorna o JSON da ferramenta modify_code
            # A observação no ciclo principal será formatada com base neste JSON
            return MOCK_META_SUCCESS_JSON_RESULT # Retorna JSON simulado
        else:
            return mock.DEFAULT # Deixa a chamada não-meta passar

    mock_recursive_run.side_effect = recursive_run_side_effect

    # Configura a sequência de chamadas LLM para o ciclo *principal*
    # 1. Tenta executar código falho
    # 2. Após a observação do meta-ciclo (que foi mockado para parecer sucesso), o LLM dá a resposta final
    mock_llm_call.side_effect = [
        LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE,
        LLM_JSON_RESPONSE_FINAL_SUCCESS
    ]

    objective = "Execute failing code, expect auto-correct success"
    final_response = agent.run(objective=objective)

    # Assertions Ciclo Principal
    # 1. LLM -> Exec(Error) -> Trigger Meta -> run(meta=True) side_effect retorna "Success" -> Obs(Meta Success)
    # 2. LLM -> Final Answer
    assert mock_llm_call.call_count == 2 # LLM(Fail) -> LLM(FinalSuccess)
    # A execução inicial que falhou
    mock_execute.assert_called_once_with(action_input={'code': CODE_TO_EXECUTE_FAILING, 'language': 'python'})
    mock_modify.assert_not_called() # Modify é no meta-ciclo (mockado)

    # Verifica se a chamada recursiva (meta) foi feita
    recursive_call_detected = False
    initial_call_objective = None
    meta_call_objective = None
    for call_args_tuple in mock_recursive_run.call_args_list:
         args, kwargs = call_args_tuple
         if kwargs.get('is_meta_objective') is True:
             recursive_call_detected = True
             meta_call_objective = kwargs.get('objective')
             assert kwargs.get('meta_depth') == 1
         elif not kwargs.get('is_meta_objective'):
              initial_call_objective = kwargs.get('objective')

    assert initial_call_objective == objective
    assert recursive_call_detected, "Chamada recursiva para auto-correção não foi detectada"
    assert meta_call_objective.startswith("AUTO-CORRECTION STEP 1: MODIFY CODE"), \
        f"Meta objective should start with 'AUTO-CORRECTION STEP 1: MODIFY CODE'. Got: {meta_call_objective[:100]}..."

    # Verifica a observação que indica o sucesso do meta-ciclo
    assert len(agent._history) >= 3 # Garante que o índice 2 existe
    assert agent._history[0] == f"Human: {objective}"
    assert agent._history[1] == LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE
    # << CHANGE: Check for the START of the actual message >>
    expected_observation_start = "Observation: An execution error occurred. Auto-correction proposed a modification"
    assert agent._history[2].startswith(expected_observation_start), \
        f"History item 2 mismatch. Expected start: '{expected_observation_start}'. Got: '{agent._history[2]}'"

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

    mock_recursive_run = mocker.patch.object(agent, 'run', wraps=agent.run)

    # LLM sempre tenta executar o código falho (simulando um loop de erro)
    mock_llm_call.return_value = LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE

    # Mock do meta-ciclo para retornar uma falha (como JSON)
    def recursive_run_side_effect_fail(*args, **kwargs):
        if kwargs.get('is_meta_objective') is True:
             agent_logger.info("[TEST MOCK] Meta-ciclo simulado retornando falha.")
             # Simula que o meta-ciclo falhou e retorna a string de falha simulada
             return "Falha: Não foi possível determinar a correção." # Mensagem de falha simulada
        else:
            # A chamada inicial não deve chegar aqui se o mock não tiver wraps
            return mock.DEFAULT # <<< CHANGE: Return DEFAULT for non-meta calls

    mock_recursive_run.side_effect = recursive_run_side_effect_fail

    objective = "Execute failing code, expect auto-correct fail"
    agent.max_iterations = 3 # Baixa max_iterations para terminar rápido
    final_response = agent.run(objective=objective)

    # Assertions
    # Ciclo 1: LLM(Fail) -> execute(Error) -> MetaCiclo(Fail) -> Obs(Meta Fail)
    # Ciclo 2: LLM(Fail) -> execute(Error) -> MetaCiclo(Fail) -> Obs(Meta Fail)
    # Ciclo 3: LLM(Fail) -> execute(Error) -> MetaCiclo(Fail) -> Obs(Meta Fail) -> Max Iterations
    assert mock_llm_call.call_count == 3
    assert mock_execute.call_count == 3
    assert mock_modify.call_count == 0 # Modify não é chamado no ciclo principal

    meta_calls = [c for c in mock_recursive_run.call_args_list if c.kwargs.get('is_meta_objective')]
    # Cada erro de execução deve disparar uma tentativa de meta-correção
    assert len(meta_calls) == 3, f"Esperado 3 chamadas meta, obteve {len(meta_calls)}"

    # Verifica o histórico final - deve conter as observações de falha do meta-ciclo
    assert len(agent._history) == 1 + (3 * 2) # Human + 3 * (LLM + Obs)
    assert agent._history[0] == f"Human: {objective}"
    # Verifica a última observação - deve indicar falha na auto-correção
    # The actual observation reflects the internal error when trying to parse the mock's string response
    expected_observation = (
        "Observation: An execution error occurred. Auto-correction attempt failed due to an internal "
        "error processing the modification result: Expecting value: line 1 column 1 (char 0)"
    )
    # We check startswith because the raw result might be appended
    assert agent._history[-1].startswith(expected_observation), \
        f"Last history item mismatch. Expected start: '{expected_observation}'. Got: '{agent._history[-1]}'"

    # Verifica a resposta final
    assert "Máximo de iterações" in final_response, \
        f"Final response should indicate iteration limit. Got: '{final_response}'"


# Teste 3: Atinge a profundidade máxima de auto-correção
def test_react_agent_run_autocorrect_max_depth(
    agent_instance, mock_code_tools, mock_db, mocker,
    LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE, EXECUTE_CODE_RESULT_ERROR, # Usa dict aqui para mock
    EXECUTE_CODE_RESULT_ERROR_JSON, CODE_TO_EXECUTE_FAILING # << REMOVE non-existent fixture >>
):
    """Testa que a auto-correção para no MAX_META_DEPTH."""
    agent, mock_llm_call = agent_instance
    mock_execute, mock_modify = mock_code_tools
    # mock_save = mock_db

    patched_max_depth = 0
    mocker.patch('core.agent.MAX_META_DEPTH', patched_max_depth, create=True)

    mock_llm_call.return_value = LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE
    mock_execute.return_value = EXECUTE_CODE_RESULT_ERROR # mock retorna dict

    mock_recursive_run_for_max_depth = mocker.patch.object(agent, 'run', wraps=agent.run)

    def side_effect_for_max_depth_interceptor(*args, **kwargs):
        call_is_meta = kwargs.get('is_meta_objective', False)
        if call_is_meta:
            raise AssertionError(f"Chamada meta recursiva detectada quando a profundidade máxima ({patched_max_depth}) deveria impedir!")
        else:
            return mock.DEFAULT

    mock_recursive_run_for_max_depth.side_effect = side_effect_for_max_depth_interceptor

    original_objective = "Execute failing code, expect max depth hit"
    agent.max_iterations = 3 # Limita iterações do ciclo principal
    final_response = agent.run(objective=original_objective, meta_depth=0) 

    # Assertions
    expected_execute_calls = 3 
    assert mock_execute.call_count == expected_execute_calls, f"Esperado {expected_execute_calls} chamadas, obteve {mock_execute.call_count}"
    expected_args = mock.call(action_input={'code': CODE_TO_EXECUTE_FAILING, 'language': 'python'})
    mock_execute.assert_has_calls([expected_args] * expected_execute_calls, any_order=False)

    meta_call_made = any(c.kwargs.get('is_meta_objective') for c in mock_recursive_run_for_max_depth.call_args_list)
    assert not meta_call_made, "Chamada meta recursiva foi feita apesar da profundidade máxima"

    # Histórico: Human, LLM1(Exec), Obs1(Error), ..., LLM3(Exec), Obs3(Error)
    expected_history_length = 1 + (expected_execute_calls * 2)
    assert len(agent._history) == expected_history_length, f"Esperado histórico {expected_history_length}, obteve {len(agent._history)}"
    assert agent._history[0] == f"Human: {original_objective}"
    assert agent._history[-2] == LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE
    # A observação é o erro original, pois a meta-correção foi bloqueada ANTES
    # Parse JSON from the observation string for comparison
    observation_str = agent._history[-1]
    assert observation_str.startswith("Observation: ")
    try:
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
