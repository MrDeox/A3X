# tests/test_agent.py
import pytest
import json
from unittest.mock import MagicMock, call, patch
from unittest import mock
from core.agent import ReactAgent
from core.config import MAX_META_DEPTH

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

# Test Auto-Correction Success
def test_react_agent_run_autocorrects_execution_error(agent_instance, mocker, mock_db):
    agent, mock_llm_call = agent_instance
    mock_save = mock_db

    # 1. Mock Tools (execute_code, modify_code)
    mock_execute_code_func = MagicMock()
    # Corrected string escaping for stderr
    execute_error_result = {
        "status": "error",
        "action": "execution_failed",
        "data": {
            "message": "NameError: name 'prnt' is not defined",
            "stderr": "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nNameError: name 'prnt' is not defined"
        }
    }
    # Note: The successful execution happens *inside* the mocked meta-run call.
    # The main agent cycle only sees the initial failure.
    mock_execute_code_func.return_value = execute_error_result # Only the initial error is returned to the main loop

    mock_modify_code_func = MagicMock(return_value={
        "status": "success",
        "action": "code_modified",
        "data": {"message": "Code modified successfully", "modified_code": "print('Corrected Hello')"}
    })

    # Patch the TOOLS dict *within this test's scope*
    original_tools = agent.tools # Backup original tools if needed elsewhere
    # Define the dictionary for patching separately to avoid linter issues
    tools_to_patch = {
        'execute_code': {'function': mock_execute_code_func, 'description': 'Executes Python code'},
        'modify_code': {'function': mock_modify_code_func, 'description': 'Modifies code based on instructions'},
        'final_answer': {'function': None, 'description': 'Provides the final answer'} # Ensure final_answer exists
    }
    mocker.patch.dict(agent.tools, tools_to_patch, clear=True) # Apply the patch

    # 2. Mock LLM Responses
    initial_bad_code = "prnt('Bad Hello')"
    llm_response_exec_bad_code = json.dumps({
        "Thought": "I need to execute this Python code.",
        "Action": "execute_code",
        "Action Input": {"code": initial_bad_code}
    })
    llm_response_final_after_correction = json.dumps({
        "Thought": "The auto-correction was successful, I can now give the final answer.",
        "Action": "final_answer",
        "Action Input": {"answer": "Task completed after auto-correction."}
    })
    mock_llm_call.side_effect = [
        llm_response_exec_bad_code,
        llm_response_final_after_correction
    ]

    # 3. Mock the Recursive Agent Run Call
    # Use wraps=agent.run initially to ensure the first call goes through the original logic.
    mock_recursive_run = mocker.patch.object(agent, 'run', wraps=agent.run)

    expected_meta_result = "Correção aplicada e testada com sucesso."

    def meta_run_side_effect(*args, **kwargs):
        # This side effect intercepts *all* calls to agent.run
        is_meta_objective_arg = kwargs.get('is_meta_objective', False)
        meta_depth_arg = kwargs.get('meta_depth', 0)
        objective_arg = args[0] if args else kwargs.get('objective', '')

        if is_meta_objective_arg and meta_depth_arg == 1:
            print(f"\nIntercepted meta run call (depth {meta_depth_arg}) with objective:\n{objective_arg}\n") # Debug print
            # --- Assertions on the meta-objective ---
            assert "A tentativa anterior de executar código falhou." in objective_arg
            assert "NameError: name 'prnt' is not defined" in objective_arg # Check specific error message
            assert f"```python\n{initial_bad_code}\n```" in objective_arg
            assert "Usar a ferramenta 'modify_code'" in objective_arg
            # Modify assertion to be less sensitive to exact formatting/escaping
            assert any("use a ferramenta 'execute_code' para testar a versão corrigida" in line for line in objective_arg.split('\n'))
            # --- End Assertions ---
            # Simulate the meta-cycle succeeding by returning the specific string
            return expected_meta_result
        else:
            # For non-meta calls, return mock.DEFAULT to let the original `wraps` execute.
            return mock.DEFAULT

    # Apply the side effect logic to the mock
    mock_recursive_run.side_effect = meta_run_side_effect

    # 4. Run the Agent
    initial_objective = "Execute some code that will initially fail."
    final_result = agent.run(objective=initial_objective)

    # 5. Assertions
    # Check LLM calls (initial exec, final answer)
    assert mock_llm_call.call_count == 2

    # Check that execute_code was called *once* in the main loop (with the bad code)
    mock_execute_code_func.assert_called_once_with(action_input={'code': initial_bad_code})

    # Check that the recursive run method was called correctly
    assert mock_recursive_run.call_count >= 2 # At least initial call + meta call
    # Check the *first* call was the original objective
    first_call_args, first_call_kwargs = mock_recursive_run.call_args_list[0]
    assert not first_call_args # Should have been called with keyword args
    assert first_call_kwargs.get('objective') == initial_objective
    assert not first_call_kwargs.get('is_meta_objective', False)
    assert first_call_kwargs.get('meta_depth', 0) == 0
    # The side_effect performs assertions on the meta-call's arguments

    # Check History
    # Expected: Human, LLM1(exec bad), Obs(Meta Success), LLM2(final), FinalAnswer
    assert len(agent._history) == 5
    assert agent._history[0] == f"Human: {initial_objective}"
    assert agent._history[1] == llm_response_exec_bad_code # LLM asks to execute bad code
    assert "Observation: Ocorreu um erro na execução anterior, mas um ciclo de auto-correção foi iniciado e concluído com sucesso." in agent._history[2] # Observation from meta-result
    assert agent._history[3] == llm_response_final_after_correction # LLM gives final answer
    assert agent._history[4] == "Final Answer: Task completed after auto-correction." # Final answer string

    # Check Final Result
    assert final_result == "Task completed after auto-correction."

    # Check DB save
    mock_save.assert_called_once()

    # Restore original tools if necessary (optional, as mocker scope should handle it)
    agent.tools = original_tools

# Optional: Add tests for failure and max depth later
# def test_react_agent_run_autocorrect_fails(...):
#     pass
#
# def test_react_agent_run_autocorrect_max_depth(...):
#     pass

# Placeholder to ensure the file ends correctly if it was truncated before