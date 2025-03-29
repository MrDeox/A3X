# tests/test_agent_run_basic.py
import pytest
import json
from unittest.mock import MagicMock, call

# Fixtures e constantes agora vêm de conftest.py

# Test 1: Flow involving a refactored skill (list_files)
def test_react_agent_run_list_files(agent_instance, mock_list_files_tool, mock_db, LLM_JSON_RESPONSE_LIST_FILES, LLM_JSON_RESPONSE_LIST_FILES_FINAL, LIST_FILES_RESULT_JSON):
    """Testa um fluxo básico com a ferramenta list_files e final_answer."""
    agent, mock_llm_call = agent_instance
    # mock_save = mock_db # mock_db agora é autouse, não precisamos capturá-lo a menos que precisemos do mock retornado

    # Set up the sequence of LLM responses using JSON mocks from conftest
    mock_llm_call.side_effect = [
        LLM_JSON_RESPONSE_LIST_FILES,
        LLM_JSON_RESPONSE_LIST_FILES_FINAL
    ]

    objective = "List the files in my current directory."
    final_response = agent.run(objective)

    # Assertions
    assert mock_llm_call.call_count == 2
    # Check the first LLM call structure
    first_call_messages = mock_llm_call.call_args_list[0][0][0]
    assert first_call_messages[0]['role'] == 'system'
    assert objective in first_call_messages[1]['content']

    # Check that list_files tool was called correctly (fixture mock_list_files_tool)
    mock_list_files_tool.assert_called_once_with(action_input={'directory': '.'})

    # Check the final answer returned by the agent
    assert final_response == "Files listed: mock_file.txt"

    # Check agent history after successful run
    assert len(agent._history) == 5 # Human, LLM1 (JSON), Obs1 (JSON), LLM2 (JSON), FinalAnswer
    assert agent._history[0] == f"Human: {objective}"
    assert agent._history[1] == LLM_JSON_RESPONSE_LIST_FILES # Raw LLM JSON
    assert agent._history[2] == f"Observation: {LIST_FILES_RESULT_JSON}" # Tool result as JSON
    assert agent._history[3] == LLM_JSON_RESPONSE_LIST_FILES_FINAL # Raw LLM JSON
    assert agent._history[4] == "Final Answer: Files listed: mock_file.txt" # Final answer string

    # Check if state was saved at the end (mock_db is autouse, implicitly checked via fixture)
    # Se precisarmos verificar a chamada explicitamente:
    # mock_save = mock_db # Captura o mock retornado pelo yield em conftest
    # mock_save.assert_called_once()


# Test 2: Flow stopping directly with final_answer
def test_react_agent_run_final_answer_direct(agent_instance, mock_list_files_tool, mock_db, LLM_JSON_RESPONSE_HELLO_FINAL):
    """Testa um fluxo onde o LLM retorna final_answer imediatamente."""
    agent, mock_llm_call = agent_instance
    # mock_save = mock_db

    # LLM returns final_answer immediately as JSON from conftest
    mock_llm_call.return_value = LLM_JSON_RESPONSE_HELLO_FINAL

    objective = "Just say hello"
    final_response = agent.run(objective)

    # Assertions
    mock_llm_call.assert_called_once()
    # Check messages passed to LLM
    call_messages = mock_llm_call.call_args[0][0]
    assert call_messages[0]['role'] == 'system'
    assert objective in call_messages[1]['content']

    # Ensure the list_files tool was NOT called
    mock_list_files_tool.assert_not_called()

    # Check the final answer
    assert final_response == "Hello there!"

    # Check history
    assert len(agent._history) == 3 # Human, LLM (JSON), FinalAnswer
    assert agent._history[0] == f"Human: {objective}"
    assert agent._history[1] == LLM_JSON_RESPONSE_HELLO_FINAL
    assert agent._history[2] == "Final Answer: Hello there!"

    # Check if state was saved (mock_db is autouse)
    # mock_save.assert_called_once()
