# tests/conftest.py
import pytest
import json
from unittest import mock
from unittest.mock import MagicMock, call, patch

# Importações do seu projeto (ajuste os caminhos se necessário)
from core.agent import ReactAgent, TOOLS
# from core.config import MAX_META_DEPTH # Importar se necessário diretamente, ou usar patch

# --- Constantes e Mocks de Dados Compartilhados (Agora como Fixtures) ---

# Mock LLM JSON responses
@pytest.fixture
def LLM_JSON_RESPONSE_LIST_FILES():
    return json.dumps({
        "Thought": "The user wants to list files in the current directory. I should use the list_files tool.",
        "Action": "list_files",
        "Action Input": {"directory": "."}
    })

@pytest.fixture
def LLM_JSON_RESPONSE_LIST_FILES_FINAL():
    return json.dumps({
        "Thought": "I have received the list of files. I should present this to the user using final_answer.",
        "Action": "final_answer",
        "Action Input": {"answer": "Files listed: mock_file.txt"}
    })

@pytest.fixture
def LLM_JSON_RESPONSE_HELLO_FINAL():
    return json.dumps({
        "Thought": "The user just wants a greeting. I should respond directly.",
        "Action": "final_answer",
        "Action Input": {"answer": "Hello there!"}
    })

@pytest.fixture
def LLM_JSON_RESPONSE_MISSING_ACTION():
    return json.dumps({
        "Thought": "Something is wrong.",
        "Action Input": {}
    })

@pytest.fixture
def LLM_JSON_RESPONSE_INVALID_INPUT_TYPE():
    return json.dumps({
        "Thought": "Trying to list files with bad input.",
        "Action": "list_files",
        "Action Input": "not a dictionary"
    })

@pytest.fixture
def INVALID_JSON_STRING():
    return "This is not JSON { definitely not"

@pytest.fixture
def JSON_ARRAY_STRING():
    return "[1, 2, 3]" # Valid JSON, but not an object

# Mock Skill Results (List Files)
@pytest.fixture
def LIST_FILES_RESULT_SUCCESS():
    return {
        "status": "success",
        "action": "list_files_success",
        "data": {"message": "Mock list successful", "files": ["mock_file.txt"]}
    }

@pytest.fixture
def LIST_FILES_RESULT_JSON(LIST_FILES_RESULT_SUCCESS):
    return json.dumps(LIST_FILES_RESULT_SUCCESS)

# --- Mocks para Testes de Auto-Correção (Agora como Fixtures) ---
@pytest.fixture
def CODE_TO_EXECUTE_FAILING():
    return "print(1/0)"

@pytest.fixture
def CODE_TO_EXECUTE_SUCCESS():
    return "print('Success!')"

@pytest.fixture
def LLM_JSON_RESPONSE_EXECUTE_FAILING_CODE(CODE_TO_EXECUTE_FAILING):
    return json.dumps({
        "Thought": "I need to execute this potentially problematic code.",
        "Action": "execute_code",
        "Action Input": {"code": CODE_TO_EXECUTE_FAILING, "language": "python"}
    })

# Tool Execution Results (as dictionaries and JSON strings)
@pytest.fixture
def EXECUTE_CODE_RESULT_ERROR():
    return {
        "status": "error", "action": "execution_failed", "data": {"message": "Falha na execução do código: erro de runtime. Stderr: Traceback...\\nZeroDivisionError: division by zero", "stderr": "Traceback (most recent call last):\\n  File \\\"<string>\\\", line 1, in <module>\\nZeroDivisionError: division by zero", "stdout": "", "returncode": 1}
    }

@pytest.fixture
def EXECUTE_CODE_RESULT_ERROR_JSON(EXECUTE_CODE_RESULT_ERROR):
    return json.dumps(EXECUTE_CODE_RESULT_ERROR)

@pytest.fixture
def EXECUTE_CODE_RESULT_SUCCESS():
    return {
        "status": "success", "action": "execution_succeeded", "data": {"message": "Code executed successfully", "stdout": "Success!", "stderr": "", "returncode": 0}
    }

@pytest.fixture
def EXECUTE_CODE_RESULT_SUCCESS_JSON(EXECUTE_CODE_RESULT_SUCCESS):
    return json.dumps(EXECUTE_CODE_RESULT_SUCCESS)

# --- Fixtures Compartilhadas ---

@pytest.fixture(autouse=True)
def mock_db(mocker):
    """Mocks database load and save functions for all tests."""
    mocker.patch('core.agent.load_agent_state', return_value={})
    mock_save = mocker.patch('core.agent.save_agent_state')
    # Retorna o mock da função save para que possa ser verificado nos testes
    yield mock_save # yield permite setup/teardown se necessário, aqui apenas retorna

@pytest.fixture
def agent_instance(mocker):
    """Fixture para criar uma instância de ReactAgent com _call_llm mockado."""
    # Garante que o estado inicial seja vazio
    mocker.patch('core.agent.load_agent_state', return_value={})
    # Mock o carregamento do schema caso o arquivo não exista no ambiente de teste
    mocker.patch('core.agent.LLM_RESPONSE_SCHEMA', {"type": "object"}) # Fornece um schema dummy
    agent = ReactAgent(llm_url="http://mock-llm-url/v1", system_prompt="Mock system prompt")
    # Mock o método interno _call_llm após a instanciação
    mock_llm_call = mocker.patch.object(agent, '_call_llm')
    # Retorna o agente e o mock da chamada LLM
    return agent, mock_llm_call

@pytest.fixture
def mock_list_files_tool(mocker, LIST_FILES_RESULT_SUCCESS):
    """Mocks a ferramenta list_files."""
    # Use the injected fixture value for return_value
    mock_func = MagicMock(return_value=LIST_FILES_RESULT_SUCCESS)
    # Patch no dicionário TOOLS onde ReactAgent o procura
    mocker.patch.dict('core.agent.TOOLS', {
        'list_files': {'function': mock_func, 'description': 'Mock list_files description'},
        'final_answer': {'function': None, 'description': 'Final Answer Tool'} # Placeholder
    }, clear=True) # clear=True garante isolamento
    return mock_func

@pytest.fixture
def mock_code_tools(mocker):
    """Mocks as ferramentas execute_code e modify_code and patches them into TOOLS."""
    mock_execute = MagicMock(name="mock_execute_code")
    # Patch these mocks into the TOOLS dictionary
    mocker.patch.dict('core.agent.TOOLS', {
        'execute_code': {'function': mock_execute, 'description': 'Mock execute_code'},
        'final_answer': {'function': None, 'description': 'Final Answer Tool'} # Placeholder
    }, clear=False) # Set clear=False
    # Return the mocks so tests can configure return_values/side_effects
    return mock_execute # Only return execute mock
