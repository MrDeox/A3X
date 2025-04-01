import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# Adiciona o diretório raiz ao path para importar os módulos
# <<< MODIFIED Path Insertion >>>
# Ensure the path is absolute and go up one level from 'tests' directory
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Assume que core.config pode ser importado para testar TAVILY_ENABLED
try:
    from skills.web_search import skill_search_web, TAVILY_ENABLED, TAVILY_API_KEY
except ImportError as e:
    # Log the error for better debugging if skip occurs
    print(f"[Test Setup Warning] Could not import skills.web_search: {e}")
    pytest.skip("Não foi possível importar skills.web_search", allow_module_level=True)

# Mock data for Tavily API response
MOCK_TAVILY_SUCCESS_RESPONSE = {
    "query": "teste de busca",
    "follow_up_questions": None,
    "answer": None,
    "images": None,
    "results": [
        {
            "title": "Resultado 1",
            "url": "http://example.com/1",
            "content": "Snippet do resultado 1...",
            "score": 0.95,
            "raw_content": None
        },
        {
            "title": "Resultado 2",
            "url": "http://example.com/2",
            "content": "Snippet do resultado 2...",
            "score": 0.92,
            "raw_content": None
        }
    ],
    "response_time": 0.5
}

@pytest.fixture
def mock_tavily_client(mocker):
    """Fixture para mockar o TavilyClient."""
    mock_constructor = mocker.patch('skills.web_search.TavilyClient')
    mock_client = MagicMock()
    mock_constructor.return_value = mock_client
    return mock_constructor, mock_client


def test_skill_search_web_success(mocker, mock_tavily_client):
    """
    Testa o cenário de sucesso da busca na web com Tavily.
    """
    mocker.patch('skills.web_search.TAVILY_ENABLED', True)
    mocker.patch('skills.web_search.TAVILY_API_KEY', 'fake_key')

    mock_constructor, mock_client = mock_tavily_client
    mock_client.search.return_value = MOCK_TAVILY_SUCCESS_RESPONSE

    action_input = {"query": "teste de busca", "max_results": 5}
    result = skill_search_web(action_input=action_input)

    mock_constructor.assert_called_once_with(api_key='fake_key')
    mock_client.search.assert_called_once_with(query="teste de busca", search_depth="basic", max_results=5)
    assert result["status"] == "success"
    assert result["action"] == "web_search_completed"
    assert "Busca na web (Tavily) por 'teste de busca' concluída" in result["data"]["message"]
    assert len(result["data"]["results"]) == 2
    assert result["data"]["results"][0]["title"] == "Resultado 1"


def test_skill_search_web_success_max_results(mocker, mock_tavily_client):
    """
    Testa se skill_search_web usa o parâmetro max_results corretamente.
    """
    mocker.patch('skills.web_search.TAVILY_ENABLED', True)
    mocker.patch('skills.web_search.TAVILY_API_KEY', 'fake_key')

    mock_constructor, mock_client = mock_tavily_client
    mock_client.search.return_value = MOCK_TAVILY_SUCCESS_RESPONSE # A resposta mock não precisa mudar

    action_input = {"query": "test query", "max_results": 3}
    result = skill_search_web(action_input=action_input)

    mock_constructor.assert_called_once_with(api_key='fake_key')
    mock_client.search.assert_called_once_with(query="test query", search_depth="basic", max_results=3) # Verifica max_results
    assert result["status"] == "success"
    assert result["action"] == "web_search_completed"
    assert "Busca na web (Tavily) por 'test query' concluída" in result["data"]["message"]


def test_skill_search_web_no_query(mocker):
    """
    Testa o erro quando o parâmetro 'query' está ausente.
    """
    mocker.patch('skills.web_search.TAVILY_ENABLED', True)
    mocker.patch('skills.web_search.TAVILY_API_KEY', 'fake_key')

    action_input = {}
    result = skill_search_web(action_input=action_input)

    assert result["status"] == "error"
    assert result["action"] == "web_search_failed"
    assert "Parâmetro 'query' ausente ou inválido." in result["data"]["message"]


def test_skill_search_web_api_error(mocker, mock_tavily_client):
    """
    Testa o tratamento de erro quando a API Tavily falha.
    """
    mocker.patch('skills.web_search.TAVILY_ENABLED', True)
    mocker.patch('skills.web_search.TAVILY_API_KEY', 'fake_key')

    mock_constructor, mock_client = mock_tavily_client
    mock_client.search.side_effect = Exception("Tavily Connection Error")

    action_input = {"query": "error query"}
    result = skill_search_web(action_input=action_input)

    mock_constructor.assert_called_once_with(api_key='fake_key')
    mock_client.search.assert_called_once_with(query="error query", search_depth="basic", max_results=5)
    assert result["status"] == "error"
    assert result["action"] == "web_search_failed"
    assert "Erro ao executar a busca na web (Tavily): Tavily Connection Error" in result["data"]["message"]


def test_skill_search_web_disabled(mocker):
    """
    Testa o comportamento quando a busca Tavily está desabilitada.
    """
    mocker.patch('skills.web_search.TAVILY_ENABLED', False)
    # API Key não importa se está desabilitado
    mocker.patch('skills.web_search.TAVILY_API_KEY', 'irrelevant_key')

    mock_constructor = mocker.patch('skills.web_search.TavilyClient') # Mock para garantir que NÃO seja chamado

    action_input = {"query": "any query"}
    result = skill_search_web(action_input=action_input)

    mock_constructor.assert_not_called() # Verifica se o cliente NÃO foi instanciado
    assert result["status"] == "error" # <<< CORRIGIDO: Deve ser error quando desabilitado >>>
    assert result["action"] == "web_search_disabled"
    assert "Busca na web (Tavily) está desabilitada na configuração" in result["data"]["message"]


def test_skill_search_web_no_api_key(mocker):
    """
    Testa o erro quando a chave da API Tavily não está configurada, mas está habilitado.
    """
    mocker.patch('skills.web_search.TAVILY_ENABLED', True)
    mocker.patch('skills.web_search.TAVILY_API_KEY', None) # Chave é None

    mock_constructor = mocker.patch('skills.web_search.TavilyClient') # Mock para garantir que NÃO seja chamado

    action_input = {"query": "any query"}
    result = skill_search_web(action_input=action_input)

    mock_constructor.assert_not_called()
    assert result["status"] == "error"
    assert result["action"] == "web_search_failed" # A falha agora é por falta de chave
    assert "Chave da API Tavily (TAVILY_API_KEY) não encontrada." in result["data"]["message"] 