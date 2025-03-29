# tests/test_code_generation.py
import pytest
import requests
import json
from unittest.mock import MagicMock, patch

# Importa a função refatorada
# Certifique-se que a estrutura de diretórios permite essa importação
# Se rodar pytest da raiz do projeto, 'skills' deve ser encontrável
try:
    from skills.code_generation import skill_generate_code
except ImportError:
    pytest.skip("Não foi possível importar skill_generate_code", allow_module_level=True)

# Importa constantes do config se necessário para verificar a chamada
try:
    from core.config import LLAMA_SERVER_URL, LLAMA_DEFAULT_HEADERS
except ImportError:
    # Define valores padrão se config não puder ser importado (menos ideal)
    LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
    LLAMA_DEFAULT_HEADERS = {"Content-Type": "application/json"}
    pytest.skip("Não foi possível importar core.config, usando URLs/Headers padrão", allow_module_level=True)


# --- Mocks de Resposta ---
MOCK_CODE = "def hello():\n    print('Hello from mock!')"
MOCK_LLM_RESPONSE_SUCCESS = {
    "choices": [{"message": {"content": MOCK_CODE}}]
}
MOCK_LLM_RESPONSE_EMPTY = {
    "choices": [{"message": {"content": ""}}]
}
MOCK_LLM_RESPONSE_WITH_MARKDOWN = {
    "choices": [{"message": {"content": f"```python\n{MOCK_CODE}\n```"}}]
}
MOCK_LLM_RESPONSE_UNEXPECTED_FORMAT = {
    "error": "unexpected format" # Sem 'choices'
}

# --- Testes com Pytest e Mocker ---

@patch('skills.code_generation.requests.post')
def test_generate_code_success(mock_post, mocker):
    """Testa geração de código bem-sucedida."""
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_LLM_RESPONSE_SUCCESS
    mock_response.raise_for_status = MagicMock() # Simula resposta HTTP 200 OK
    mock_post.return_value = mock_response

    action_input = {
        "purpose": "Say hello",
        "language": "python",
        "construct_type": "function",
        "context": "No specific context needed" # Testando com contexto opcional
    }
    result = skill_generate_code(action_input)

    assert result.get("status") == "success"
    assert result.get("action") == "code_generated"
    assert isinstance(result.get("data"), dict)
    assert result["data"].get("code") == MOCK_CODE
    assert result["data"].get("language") == "python"
    assert result["data"].get("construct_type") == "function"
    assert "Código gerado com sucesso" in result["data"].get("message", "")

    # Verifica a chamada a requests.post
    mock_post.assert_called_once()
    call_args, call_kwargs = mock_post.call_args
    assert call_args[0] == LLAMA_SERVER_URL
    assert call_kwargs.get("headers") == LLAMA_DEFAULT_HEADERS
    payload = call_kwargs.get("json")
    assert isinstance(payload, dict)
    assert payload.get("messages")[0]["role"] == "user"
    assert "Say hello" in payload["messages"][0]["content"] # Verifica purpose no prompt
    assert "python" in payload["messages"][0]["content"]    # Verifica language no prompt
    assert "function" in payload["messages"][0]["content"] # Verifica construct_type
    assert "No specific context needed" in payload["messages"][0]["content"] # Verifica context
    assert "APENAS O CÓDIGO" in payload["messages"][0]["content"] # Verifica instrução de formato

@patch('skills.code_generation.requests.post')
def test_generate_code_success_with_markdown_cleanup(mock_post, mocker):
    """Testa se a limpeza de markdown funciona."""
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_LLM_RESPONSE_WITH_MARKDOWN
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    action_input = {"purpose": "Say hello with markdown"}
    result = skill_generate_code(action_input)

    assert result.get("status") == "success"
    assert result.get("action") == "code_generated"
    assert isinstance(result.get("data"), dict)
    # Verifica se o código retornado NÃO contém os marcadores ```
    assert result["data"].get("code") == MOCK_CODE
    assert "```" not in result["data"].get("code", "")

def test_generate_code_missing_purpose():
    """Testa falha quando 'purpose' está ausente."""
    action_input = {"language": "python"} # Sem purpose
    result = skill_generate_code(action_input)

    assert result.get("status") == "error"
    assert result.get("action") == "code_generation_failed"
    assert isinstance(result.get("data"), dict)
    assert "propósito do código (purpose) não foi especificado" in result["data"].get("message", "")

@patch('skills.code_generation.requests.post')
def test_generate_code_request_exception(mock_post):
    """Testa falha na chamada de rede."""
    mock_post.side_effect = requests.exceptions.RequestException("Network Error")

    action_input = {"purpose": "Trigger network error"}
    result = skill_generate_code(action_input)

    assert result.get("status") == "error"
    assert result.get("action") == "code_generation_failed"
    assert isinstance(result.get("data"), dict)
    assert "Erro de comunicação com o servidor LLM: Network Error" in result["data"].get("message", "")

@patch('skills.code_generation.requests.post')
def test_generate_code_http_error(mock_post):
    """Testa falha por erro HTTP (ex: 4xx, 5xx)."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
    mock_post.return_value = mock_response

    action_input = {"purpose": "Trigger HTTP error"}
    result = skill_generate_code(action_input)

    assert result.get("status") == "error"
    assert result.get("action") == "code_generation_failed"
    assert isinstance(result.get("data"), dict)
    # A exceção HTTPError é capturada pelo except requests.exceptions.RequestException
    assert "Erro de comunicação com o servidor LLM: 404 Not Found" in result["data"].get("message", "")

@patch('skills.code_generation.requests.post')
def test_generate_code_empty_code_response(mock_post):
    """Testa falha quando LLM retorna código vazio."""
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_LLM_RESPONSE_EMPTY
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    action_input = {"purpose": "Generate nothing"}
    result = skill_generate_code(action_input)

    assert result.get("status") == "error"
    assert result.get("action") == "code_generation_failed"
    assert isinstance(result.get("data"), dict)
    assert "LLM retornou uma resposta vazia ou inválida" in result["data"].get("message", "")

@patch('skills.code_generation.requests.post')
def test_generate_code_unexpected_response_format(mock_post):
    """Testa falha quando formato da resposta LLM é inesperado."""
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_LLM_RESPONSE_UNEXPECTED_FORMAT # Sem 'choices'
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    action_input = {"purpose": "Trigger bad format"}
    result = skill_generate_code(action_input)

    assert result.get("status") == "error"
    assert result.get("action") == "code_generation_failed"
    assert isinstance(result.get("data"), dict)
    assert "Formato de resposta inesperado do LLM" in result["data"].get("message", "")