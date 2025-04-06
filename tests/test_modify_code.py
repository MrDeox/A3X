# tests/test_modify_code.py
import requests
import json
from unittest.mock import MagicMock, patch

# Importa a função refatorada
from skills.modify_code import skill_modify_code

# Importa constantes do config se necessário para verificar a chamada
from core.config import LLAMA_SERVER_URL, LLAMA_DEFAULT_HEADERS

# --- Configuração de Logging (Opcional, para debug dos testes) ---
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# --- Mocks e Dados de Teste ---
ORIGINAL_CODE = """def greet(name):
    print(f"Hello, {name}")"""

MODIFIED_CODE_COMMENT = """def greet(name):
    # Simple greeting function
    print(f"Hello, {name}")"""

MODIFIED_CODE_RETURN = """def greet(name):
    return f"Hello, {name}"""

# Respostas LLM Mock
MOCK_LLM_RESPONSE_SUCCESS_COMMENT = {
    "choices": [{"message": {"content": MODIFIED_CODE_COMMENT}}]
}
MOCK_LLM_RESPONSE_SUCCESS_RETURN = {
    "choices": [{"message": {"content": MODIFIED_CODE_RETURN}}]
}
MOCK_LLM_RESPONSE_WITH_MARKDOWN = {
    "choices": [{"message": {"content": f"```python\n{MODIFIED_CODE_COMMENT}\n```"}}]
}
MOCK_LLM_RESPONSE_NO_CHANGE = {
    "choices": [{"message": {"content": ORIGINAL_CODE}}]  # Retorna o original
}
MOCK_LLM_RESPONSE_EMPTY = {"choices": [{"message": {"content": ""}}]}  # Conteúdo vazio
MOCK_LLM_RESPONSE_UNEXPECTED_FORMAT = {"error": "unexpected format"}  # Sem 'choices'

# --- Testes ---


@patch("skills.modify_code.requests.post")
def test_modify_code_success_add_comment(mock_post):
    """Testa modificação bem-sucedida (adicionar comentário)."""
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_LLM_RESPONSE_SUCCESS_COMMENT
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    action_input = {
        "modification": "Add a docstring comment to the function",
        "code_to_modify": ORIGINAL_CODE,
        "language": "python",
    }
    result = skill_modify_code(action_input)

    assert result.get("status") == "success"
    assert result.get("action") == "code_modified"
    data = result.get("data", {})
    assert data.get("original_code") == ORIGINAL_CODE
    assert data.get("modified_code") == MODIFIED_CODE_COMMENT
    assert data.get("language") == "python"
    assert "Código modificado com sucesso" in data.get("message", "")

    # Verifica a chamada a requests.post
    mock_post.assert_called_once()
    call_args, call_kwargs = mock_post.call_args
    assert call_args[0] == LLAMA_SERVER_URL
    assert (
        call_kwargs.get("headers") == LLAMA_DEFAULT_HEADERS
    )  # Verifica headers padrão
    payload = call_kwargs.get("json")
    assert isinstance(payload, dict)
    assert payload.get("messages")[0]["role"] == "user"
    user_prompt = payload["messages"][0]["content"]
    assert action_input["modification"] in user_prompt
    assert ORIGINAL_CODE in user_prompt
    assert "Return ONLY the complete, modified code block" in user_prompt


@patch("skills.modify_code.requests.post")
def test_modify_code_success_change_logic(mock_post):
    """Testa modificação bem-sucedida (alterar print para return)."""
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_LLM_RESPONSE_SUCCESS_RETURN
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    action_input = {
        "modification": "Change the function to return the string instead of printing it",
        "code_to_modify": ORIGINAL_CODE,
        "language": "python",
    }
    result = skill_modify_code(action_input)

    assert result.get("status") == "success"
    assert result.get("action") == "code_modified"
    data = result.get("data", {})
    assert data.get("original_code") == ORIGINAL_CODE
    assert data.get("modified_code") == MODIFIED_CODE_RETURN  # Verifica código alterado
    assert data.get("language") == "python"


@patch("skills.modify_code.requests.post")
def test_modify_code_success_markdown_cleanup(mock_post):
    """Testa se a limpeza de markdown funciona na modificação."""
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_LLM_RESPONSE_WITH_MARKDOWN
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    action_input = {
        "modification": "Add a comment (with markdown expected)",
        "code_to_modify": ORIGINAL_CODE,
        # language default é python
    }
    result = skill_modify_code(action_input)

    assert result.get("status") == "success"
    data = result.get("data", {})
    assert (
        data.get("modified_code") == MODIFIED_CODE_COMMENT
    )  # Código correto após limpeza
    assert "```" not in data.get("modified_code", "")  # Garante que ``` foram removidos


@patch("skills.modify_code.requests.post")
def test_modify_code_no_change(mock_post):
    """Testa o caso onde o LLM retorna o código original (sem mudanças)."""
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_LLM_RESPONSE_NO_CHANGE  # Retorna o original
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    action_input = {
        "modification": "Make an unnecessary change",
        "code_to_modify": ORIGINAL_CODE,
    }
    result = skill_modify_code(action_input)

    assert result.get("status") == "no_change"
    assert result.get("action") == "code_modification_no_change"
    data = result.get("data", {})
    assert data.get("original_code") == ORIGINAL_CODE
    assert data.get("modified_code") == ORIGINAL_CODE  # Código é o mesmo
    assert "LLM não aplicou a modificação" in data.get("message", "")


def test_modify_code_missing_modification():
    """Testa falha quando 'modification' está ausente."""
    action_input = {"code_to_modify": ORIGINAL_CODE}  # Sem modification
    result = skill_modify_code(action_input)

    assert result.get("status") == "error"
    assert result.get("action") == "modify_code_failed"
    data = result.get("data", {})
    assert (
        "Erro: A instrução de modificação (modification) não foi especificada."
        in data.get("message", "")
    )


def test_modify_code_missing_code():
    """Testa falha quando 'code_to_modify' está ausente."""
    action_input = {"modification": "Add something"}  # Sem code_to_modify
    result = skill_modify_code(action_input)

    assert result.get("status") == "error"
    assert result.get("action") == "modify_code_failed"
    data = result.get("data", {})
    assert (
        "Erro: O código a ser modificado (code_to_modify) não foi fornecido."
        in data.get("message", "")
    )


@patch("skills.modify_code.requests.post")
def test_modify_code_request_exception(mock_post):
    """Testa falha na chamada de rede (RequestException)."""
    mock_post.side_effect = requests.exceptions.RequestException("Network Error")

    action_input = {
        "modification": "Trigger network error",
        "code_to_modify": ORIGINAL_CODE,
    }
    result = skill_modify_code(action_input)

    assert result.get("status") == "error"
    assert result.get("action") == "modify_code_failed"
    data = result.get("data", {})
    assert "Erro de comunicação com o servidor LLM: Network Error" in data.get(
        "message", ""
    )


@patch("skills.modify_code.requests.post")
def test_modify_code_http_error(mock_post):
    """Testa falha por erro HTTP (ex: 404, 500)."""
    mock_response = MagicMock()
    # Simula raise_for_status levantando HTTPError
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "500 Server Error"
    )
    mock_post.return_value = mock_response

    action_input = {
        "modification": "Trigger HTTP error",
        "code_to_modify": ORIGINAL_CODE,
    }
    result = skill_modify_code(action_input)

    assert result.get("status") == "error"
    assert result.get("action") == "modify_code_failed"
    data = result.get("data", {})
    # A exceção HTTPError é uma RequestException, então a mensagem deve ser a mesma
    assert "Erro de comunicação com o servidor LLM: 500 Server Error" in data.get(
        "message", ""
    )


@patch("skills.modify_code.requests.post")
def test_modify_code_empty_llm_response(mock_post):
    """Testa falha quando LLM retorna conteúdo vazio."""
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_LLM_RESPONSE_EMPTY
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    action_input = {"modification": "Generate nothing", "code_to_modify": ORIGINAL_CODE}
    result = skill_modify_code(action_input)

    assert result.get("status") == "error"
    assert result.get("action") == "modify_code_failed"
    data = result.get("data", {})
    assert "LLM retornou uma resposta vazia ou inválida" in data.get("message", "")


@patch("skills.modify_code.requests.post")
def test_modify_code_unexpected_llm_format(mock_post):
    """Testa falha quando formato da resposta LLM é inesperado."""
    mock_response = MagicMock()
    mock_response.json.return_value = (
        MOCK_LLM_RESPONSE_UNEXPECTED_FORMAT  # Formato inválido
    )
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    action_input = {
        "modification": "Trigger bad format",
        "code_to_modify": ORIGINAL_CODE,
    }
    # A falha deve ocorrer ao tentar acessar 'choices' etc.
    # O except genérico deve capturar ValueError ou KeyError/TypeError
    result = skill_modify_code(action_input)

    assert result.get("status") == "error"
    assert result.get("action") == "modify_code_failed"
    data = result.get("data", {})
    # Verifica a mensagem do erro inesperado, que pode incluir o tipo da exceção original
    assert "Erro ao processar resposta do LLM" in data.get("message", "")
    assert "Formato de resposta inesperado" in data.get("message", "")


@patch("skills.modify_code.requests.post")
def test_modify_code_json_decode_error(mock_post):
    """Testa falha quando a resposta do LLM não é JSON válido."""
    mock_response = MagicMock()
    # Simula erro ao decodificar JSON
    mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
    mock_response.raise_for_status = MagicMock()
    # Adiciona um atributo text ao mock para a mensagem de erro
    mock_response.text = "Invalid JSON response from LLM"
    mock_post.return_value = mock_response

    action_input = {
        "modification": "Trigger JSON error",
        "code_to_modify": ORIGINAL_CODE,
    }
    result = skill_modify_code(action_input)

    assert result.get("status") == "error"
    assert result.get("action") == "modify_code_failed"
    data = result.get("data", {})
    assert "Erro ao processar resposta do LLM" in data.get("message", "")
    assert "Expecting value" in data.get("message", "")  # Detalhe do erro JSON


# Adicione mais testes se necessário, por exemplo, para diferentes linguagens
# ou tipos de modificações mais complexas.
