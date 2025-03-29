# tests/test_agent_run_basic.py
import pytest
import json
from unittest.mock import MagicMock, call, patch

# Fixtures e constantes agora vêm de conftest.py

# Import necessary modules
from core.agent import ReactAgent, AGENT_STATE_ID
from core import planner # <<< Import planner
from core.db_utils import load_agent_state, save_agent_state

# Test 1: Flow involving a refactored skill (list_files)
def test_react_agent_run_list_files(agent_instance, mock_list_files_tool, mock_db, mocker, LLM_JSON_RESPONSE_LIST_FILES, LLM_JSON_RESPONSE_LIST_FILES_FINAL, LIST_FILES_RESULT_JSON):
    """Testa um fluxo básico com a ferramenta list_files e final_answer, agora com planejamento."""
    agent, mock_llm_call = agent_instance
    # mock_save = mock_db # Not needed directly
    
    # <<< Mock the planner >>>
    mock_plan = ["Step 1: List files in current directory.", "Step 2: Provide the final answer."]
    mock_planner = mocker.patch('core.agent.planner.generate_plan')
    mock_planner.return_value = mock_plan

    # Set up the sequence of LLM responses for the TWO steps
    mock_llm_call.side_effect = [
        LLM_JSON_RESPONSE_LIST_FILES,      # Response for Step 1 (list_files)
        LLM_JSON_RESPONSE_LIST_FILES_FINAL # Response for Step 2 (final_answer)
    ]

    # Mock the list_files tool result
    mock_list_files_tool.return_value = json.loads(LIST_FILES_RESULT_JSON)

    objective = "List the files in my current directory."
    final_response = agent.run(objective)

    # Assertions
    mock_planner.assert_called_once() # Ensure planner was called
    assert mock_llm_call.call_count == 2 # One LLM call per plan step
    mock_list_files_tool.assert_called_once_with(action_input={'directory': '.'}) # Tool called correctly with full input dict
    assert final_response == "Files listed: mock_file.txt" # Final answer received

    # Check History (Human, LLM1, Obs1, LLM2, FinalAnswer)
    assert len(agent._history) == 5, f"History length mismatch. Got: {len(agent._history)}"
    assert agent._history[0] == f"Human: {objective}"
    assert LLM_JSON_RESPONSE_LIST_FILES in agent._history[1] # Raw LLM response for step 1
    # Observation includes the JSON result from the tool
    expected_obs_str = f"Observation: {LIST_FILES_RESULT_JSON}"
    # Need to compare JSON objects due to potential formatting differences
    try:
        # Remove prefix and parse actual observation
        actual_obs_json_str = agent._history[2].removeprefix("Observation: ").strip()
        actual_obs_obj = json.loads(actual_obs_json_str)
        expected_obs_obj = json.loads(LIST_FILES_RESULT_JSON)
        assert actual_obs_obj == expected_obs_obj, f"Observation JSON mismatch. Expected: {expected_obs_obj}, Got: {actual_obs_obj}"
    except (IndexError, json.JSONDecodeError) as e:
        pytest.fail(f"Failed to validate history observation [2]: {e}. History: {agent._history}")
        
    assert LLM_JSON_RESPONSE_LIST_FILES_FINAL in agent._history[3] # Raw LLM response for step 2
    assert agent._history[4] == "Final Answer: Files listed: mock_file.txt"


# Test 2: Flow stopping directly with final_answer
def test_react_agent_run_final_answer_direct(agent_instance, mock_db, mocker, LLM_JSON_RESPONSE_HELLO_FINAL):
    """Testa o caso onde o LLM retorna final_answer diretamente (agora após plano de 1 passo)."""
    agent, mock_llm_call = agent_instance
    
    # <<< Mock the planner >>>
    objective = "Just give me the final answer immediately."
    mock_plan = [objective] # Fallback plan or simple plan
    mock_planner = mocker.patch('core.agent.planner.generate_plan')
    # Simulate planner failing or returning a single step plan
    # Let's simulate successful generation of a 1-step plan
    mock_planner.return_value = mock_plan 

    # LLM returns final_answer on the first call for the single plan step
    mock_llm_call.return_value = LLM_JSON_RESPONSE_HELLO_FINAL

    final_response = agent.run(objective)

    # Assertions
    mock_planner.assert_called_once()
    mock_llm_call.assert_called_once() # Only one call needed for the single step
    assert final_response == "Hello there!"
    # Check History (Human, LLM, FinalAnswer)
    assert len(agent._history) == 3
    assert agent._history[0] == f"Human: {objective}"
    assert LLM_JSON_RESPONSE_HELLO_FINAL in agent._history[1]
    assert agent._history[2] == "Final Answer: Hello there!"
