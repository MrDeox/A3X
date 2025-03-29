# tests/test_agent.py
import pytest
from unittest.mock import MagicMock, call
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

# Mock Skill Results
MANAGE_FILES_LIST_RESULT = {
    "status": "success",
    "action": "directory_listed", # Match action string used in agent's observation formatting if applicable
    "data": {"message": "Mock list successful", "items": ["mock_file.txt"]}
}

@pytest.fixture
def mock_dependencies(mocker):
    """Fixture to mock external dependencies for ReactAgent."""
    # Mock DB functions used in __init__ and run
    mocker.patch('core.agent.load_agent_state', return_value={})
    mock_save = mocker.patch('core.agent.save_agent_state')

    # --- CORREÇÃO: Mockar a referência da função DENTRO do dict TOOLS --- 
    # Criar um mock para a função
    mock_manage_func = MagicMock(return_value=MANAGE_FILES_LIST_RESULT)
    # Patchar o dicionário TOOLS importado pelo agente para usar o mock
    mocker.patch.dict('core.agent.TOOLS', {
        'list_files': {
             'function': mock_manage_func, 
             # O resto da definição não importa para este mock, mas idealmente copiaria 
             # ou garantiria que o patch não remove outras ferramentas necessárias.
             # Como os testes atuais só usam list_files e final_answer, isto deve bastar.
        }
    }, clear=False) # clear=False garante que outras ferramentas no dict não sejam removidas

    return mock_save, mock_manage_func # Retorna o mock da função para asserção

@pytest.fixture
def react_agent_instance(mocker):
    """Fixture to create a ReactAgent instance with mocked _call_llm."""
    # We also need to mock load_agent_state here because it's called in __init__
    mocker.patch('core.agent.load_agent_state', return_value={})
    agent = ReactAgent(llm_url="http://mock-llm-url/v1", system_prompt="Mock system prompt")
    # Mock the internal _call_llm method after instantiation
    mock_llm_call = mocker.patch.object(agent, '_call_llm', return_value="") # Default empty return
    return agent, mock_llm_call

# Test 1: Flow involving a refactored skill (list_files)
def test_react_agent_run_list_files(react_agent_instance, mock_dependencies, mocker):
    agent, mock_llm_call = react_agent_instance
    mock_save, mock_manage_func = mock_dependencies

    # Set up the sequence of LLM responses
    mock_llm_call.side_effect = [
        LLM_RESPONSE_LIST_FILES,
        LLM_RESPONSE_LIST_FILES_FINAL
    ]

    objective = "List the files in my current directory."
    final_response = agent.run(objective)

    # Assertions
    assert mock_llm_call.call_count == 2
    # Check the first LLM call structure (optional but good)
    first_call_messages = mock_llm_call.call_args_list[0][0][0] # Args are positional, messages is the first arg
    assert first_call_messages[0]['role'] == 'system'
    assert objective in first_call_messages[1]['content']

    # Check that skill_manage_files was called correctly
    # Note: We mock skill_manage_files in skills.manage_files, accessed via TOOLS
    # The agent passes only action_input to the refactored skill
    mock_manage_func.assert_called_once_with(action_input={'action': 'list', 'directory': '.'})

    # Check the final answer returned by the agent
    assert final_response == "Files listed: mock_file.txt"

    # Check if state was saved at the end
    mock_save.assert_called_once()


# Test 2: Flow stopping directly with final_answer
def test_react_agent_run_final_answer_direct(react_agent_instance, mock_dependencies, mocker):
    agent, mock_llm_call = react_agent_instance
    mock_save, mock_manage_func = mock_dependencies # We need mock_save

    # LLM returns final_answer immediately
    mock_llm_call.return_value = LLM_RESPONSE_HELLO_FINAL

    objective = "Just say hello"
    final_response = agent.run(objective)

    # Assertions
    mock_llm_call.assert_called_once()
    # Check messages passed to LLM
    call_messages = mock_llm_call.call_args[0][0]
    assert call_messages[0]['role'] == 'system'
    assert objective in call_messages[1]['content']

    # Ensure the manage_files skill was NOT called
    mock_manage_func.assert_not_called()

    # Check the final answer
    assert final_response == "Hello there!"

    # Check if state was saved
    mock_save.assert_called_once()

