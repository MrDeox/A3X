import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Adiciona o diretório raiz ao path para importar os módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from skills.web_search import skill_search_web
except ImportError:
    pytest.skip("Não foi possível importar skills.web_search", allow_module_level=True)

# Simula resultados da busca DuckDuckGo
MOCK_SEARCH_RESULTS = [
    {
        "title": "Título do Resultado 1",
        "body": "Este é o conteúdo do primeiro resultado.",
        "link": "https://exemplo1.com"
    },
    {
        "title": "Título do Resultado 2",
        "body": "Este é o conteúdo do segundo resultado.",
        "link": "https://exemplo2.com"
    }
]

def test_skill_search_web_success(mocker):
    """
    Testa se skill_search_web processa corretamente os resultados da busca
    quando a busca é bem-sucedida.
    """
    # Simula a classe DDGS e seu método text
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__.return_value.text.return_value = MOCK_SEARCH_RESULTS
    
    # Patch da classe DDGS
    with patch('skills.web_search.DDGS', return_value=mock_ddgs):
        # Entidades de exemplo
        test_entities = {"query": "teste de busca"}
        test_command = "busque na web sobre teste"
        
        # Chama a skill
        result = skill_search_web(test_entities, test_command)
        
        # Verifica o formato do resultado
        assert result["status"] == "success", "O status do resultado deve ser 'success'"
        assert result["action"] == "web_search_completed", "A ação deve ser 'web_search_completed'"
        assert "data" in result, "O resultado deve conter a chave 'data'"
        assert "query" in result["data"], "O resultado deve conter a query"
        assert "results" in result["data"], "O resultado deve conter os resultados"
        
        # Verifica os resultados
        results = result["data"]["results"]
        assert len(results) == 2, "Deve haver 2 resultados"
        
        # Verifica o primeiro resultado
        first_result = results[0]
        assert first_result["title"] == "Título do Resultado 1"
        assert first_result["snippet"] == "Este é o conteúdo do primeiro resultado."
        assert first_result["url"] == "https://exemplo1.com"

def test_skill_search_web_no_query():
    """
    Testa se skill_search_web retorna erro quando nenhuma query é fornecida.
    """
    # Entidades sem query
    test_entities = {}
    test_command = "busque na web"
    
    # Chama a skill
    result = skill_search_web(test_entities, test_command)
    
    # Verifica o erro
    assert result["status"] == "error", "O status deve ser 'error'"
    assert result["action"] == "web_search_failed", "A ação deve ser 'web_search_failed'"
    assert "Nenhuma query de busca fornecida" in result["data"]["message"]

def test_skill_search_web_error(mocker):
    """
    Testa se skill_search_web trata corretamente erros durante a busca.
    """
    # Simula um erro na busca
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__.return_value.text.side_effect = Exception("Erro de conexão")
    
    # Patch da classe DDGS
    with patch('skills.web_search.DDGS', return_value=mock_ddgs):
        # Entidades de exemplo
        test_entities = {"query": "teste de busca"}
        test_command = "busque na web sobre teste"
        
        # Chama a skill
        result = skill_search_web(test_entities, test_command)
        
        # Verifica o erro
        assert result["status"] == "error", "O status deve ser 'error'"
        assert result["action"] == "web_search_failed", "A ação deve ser 'web_search_failed'"
        assert "Erro ao realizar a busca" in result["data"]["message"] 