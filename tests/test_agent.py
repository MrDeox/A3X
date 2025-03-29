# tests/test_agent.py
import pytest
import json
from unittest.mock import MagicMock, call, patch
from core.agent import ReactAgent

# Configure logging for tests if the agent uses it extensively
# setup_logging()

# Mock LLM responses
LLM_RESPONSE_LIST_FILES = """
Thought: The user wants to list files in the current directory. I should use the manage_files tool with the 'list' action.
Action: list_files
Action Input: {"action": "list", "directory": "."}
"""

LLM_RESPONSE_LIST_FILES_FINAL = """
Thought: I have received the list of files. I should present this to the user using final_answer.
Action: final_answer
Action Input: {"answer": "Files listed: mock_file.txt"}
"""

LLM_RESPONSE_HELLO_FINAL = """
Thought: The user just wants a greeting. I should respond directly.
Action: final_answer
Action Input: {"answer": "Hello there!"}
"""

# Mock LLM JSON responses
LLM_JSON_RESPONSE_LIST_FILES = json.dumps({
    "Thought": "The user wants to list files in the current directory. I should use the list_files tool.",
    "Action": "list_files",
    "Action Input": {"directory": "."}
})

LLM_JSON_RESPONSE_LIST_FILES_FINAL = json.dumps({
    "Thought": "I have received the list of files. I should present this to the user using final_answer.",
    "Action": "final_answer",
    "Action Input": {"answer": "Files listed: mock_file.txt"}
})

LLM_JSON_RESPONSE_HELLO_FINAL = json.dumps({
    "Thought": "The user just wants a greeting. I should respond directly.",
    "Action": "final_answer",
    "Action Input": {"answer": "Hello there!"}
})

LLM_JSON_RESPONSE_MISSING_ACTION = json.dumps({
    "Thought": "Something is wrong.",
    "Action Input": {}
})

LLM_JSON_RESPONSE_INVALID_INPUT_TYPE = json.dumps({
    "Thought": "Trying to list files with bad input.",
    "Action": "list_files",
    "Action Input": "not a dictionary"
})

INVALID_JSON_STRING = "This is not JSON { definitely not"
JSON_ARRAY_STRING = "[1, 2, 3]" # Valid JSON, but not an object

# Mock Skill Results
MANAGE_FILES_LIST_RESULT = {
    "status": "success",
    "action": "directory_listed", # Match action string used in agent's observation formatting if applicable
    "data": {"message": "Mock list successful", "items": ["mock_file.txt"]}
}

LIST_FILES_RESULT_SUCCESS = {
    "status": "success",
    "action": "list_files_success", # Action string within result can vary
    "data": {"message": "Mock list successful", "files": ["mock_file.txt"]}
}
LIST_FILES_RESULT_JSON = json.dumps(LIST_FILES_RESULT_SUCCESS) # For history assertion

@pytest.fixture(autouse=True) # Apply to all tests in this module
def mock_db(mocker):
    """Mocks database load and save functions."""
    mocker.patch('core.agent.load_agent_state', return_value={})
    mock_save = mocker.patch('core.agent.save_agent_state')
    return mock_save

@pytest.fixture
def mock_list_files_tool(mocker):
    """Mocks the list_files tool function."""
    mock_func = MagicMock(return_value=LIST_FILES_RESULT_SUCCESS)
    # Patch the TOOLS dictionary where ReactAgent looks for it
    mocker.patch.dict('core.agent.TOOLS', {
        'list_files': {'function': mock_func, 'description': 'Mock description'},
        # Add other tools if needed by tests, e.g., final_answer if it was a real tool
        'final_answer': {'function': None, 'description': 'Final Answer Tool'} # Placeholder if checked by agent
    }, clear=True) # Clear ensures only these mocks exist for isolation
    return mock_func

@pytest.fixture
def agent_instance(mocker):
    """Fixture to create a ReactAgent instance with mocked _call_llm."""
    # Patch load_agent_state again just in case (though autouse=True should cover it)
    mocker.patch('core.agent.load_agent_state', return_value={})
    # Mock the schema loading in case the file doesn't exist in test env
    mocker.patch('core.agent.LLM_RESPONSE_SCHEMA', {"type": "object"}) # Provide a dummy schema
    agent = ReactAgent(llm_url="http://mock-llm-url/v1", system_prompt="Mock system prompt")
    # Mock the internal _call_llm method after instantiation
    mock_llm_call = mocker.patch.object(agent, '_call_llm')
    return agent, mock_llm_call

# Test 1: Flow involving a refactored skill (list_files)
def test_react_agent_run_list_files(agent_instance, mock_list_files_tool, mock_db):
    agent, mock_llm_call = agent_instance
    mock_save = mock_db # Get the mock_save object

    # Set up the sequence of LLM responses using JSON mocks
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

    # Check that list_files tool was called correctly
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

    # Check if state was saved at the end
    mock_save.assert_called_once()


# Test 2: Flow stopping directly with final_answer
def test_react_agent_run_final_answer_direct(agent_instance, mock_list_files_tool, mock_db):
    agent, mock_llm_call = agent_instance
    mock_save = mock_db

    # LLM returns final_answer immediately as JSON
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

    # Check if state was saved
    mock_save.assert_called_once()


def test_react_agent_run_handles_json_decode_error(agent_instance, mock_list_files_tool, mock_db):
    agent, mock_llm_call = agent_instance
    mock_save = mock_db

    # First call: Invalid JSON, Second call: Valid final answer
    mock_llm_call.side_effect = [
        INVALID_JSON_STRING,
        LLM_JSON_RESPONSE_HELLO_FINAL
    ]

    objective = "Try something that breaks parsing"
    final_response = agent.run(objective)

    # Assertions
    assert mock_llm_call.call_count == 2
    # Check history for the observation about the parse error
    # History: Human, LLM1 (Bad), Obs1 (Error), LLM2 (Good), FinalAnswer
    assert len(agent._history) == 5
    assert agent._history[0] == f"Human: {objective}"
    assert agent._history[1] == INVALID_JSON_STRING # The bad response
    assert "Observation: Erro crítico - sua resposta anterior não estava no formato JSON esperado" in agent._history[2]
    assert agent._history[3] == LLM_JSON_RESPONSE_HELLO_FINAL # Good response after correction
    assert agent._history[4] == "Final Answer: Hello there!" # Final answer added to history

    # Check final answer came from the second, valid response
    assert final_response == "Hello there!"
    mock_list_files_tool.assert_not_called()
    mock_save.assert_called_once()


def test_react_agent_run_handles_llm_call_error(agent_instance, mock_list_files_tool, mock_db):
    agent, mock_llm_call = agent_instance
    mock_save = mock_db

    LLM_INTERNAL_ERROR_MSG = "Erro: Falha ao conectar com o servidor LLM (Timeout)."

    # First call: Internal error, Second call: Valid final answer
    mock_llm_call.side_effect = [
        LLM_INTERNAL_ERROR_MSG,
        LLM_JSON_RESPONSE_HELLO_FINAL
    ]

    objective = "Try something that causes LLM call error"
    final_response = agent.run(objective)

    # Assertions
    assert mock_llm_call.call_count == 2
    # Check history for the observation about the call error
    assert len(agent._history) == 4 # Human, Obs1 (Call Error), LLM2 (Good JSON), FinalAnswer
    assert agent._history[0] == f"Human: {objective}"
    assert agent._history[1] == f"Observation: Erro crítico na comunicação com o LLM: {LLM_INTERNAL_ERROR_MSG}"
    assert agent._history[2] == LLM_JSON_RESPONSE_HELLO_FINAL # Good response after correction

    # Check final answer came from the second, valid response
    assert final_response == "Hello there!"
    mock_list_files_tool.assert_not_called()
    mock_save.assert_called_once()

def test_react_agent_run_hits_max_iterations_on_persistent_json_error(agent_instance, mock_db):
    agent, mock_llm_call = agent_instance
    mock_save = mock_db

    # LLM keeps returning invalid JSON
    mock_llm_call.return_value = INVALID_JSON_STRING
    agent.max_iterations = 2 # Lower max iterations for testing

    objective = "Keep failing"
    final_response = agent.run(objective)

    # Assertions
    assert mock_llm_call.call_count == 2 # Called twice before giving up
    # Check the specific error message for max iterations
    assert "Desculpe, falha ao processar a resposta do LLM após 2 tentativas." in final_response

    # Check history contains the error observations
    assert len(agent._history) == 5 # Human, LLM1, Obs1(Err), LLM2, Obs2(Err)
    assert "Observation: Erro crítico - sua resposta anterior não estava no formato JSON esperado" in agent._history[2]
    assert "Observation: Erro crítico - sua resposta anterior não estava no formato JSON esperado" in agent._history[4]

    mock_save.assert_called_once() # Should still save state on failure

# You can add more tests, e.g., for _trim_history if needed

