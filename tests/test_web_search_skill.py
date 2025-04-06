import sys
import os
import pytest
from unittest.mock import patch, MagicMock, ANY

# --- Add project root to sys.path ---
# Ensure core modules can be found when importing skills
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # print(f"[test_web_search_skill.py] Added project root to sys.path: {project_root}") # Optional debug print

# Importar a skill *depois* de ajustar o sys.path
from skills.web_search import web_search

# Marcar todos os testes neste arquivo para serem executados com pytest
pytestmark = pytest.mark.asyncio


async def async_generator_for(item):
    yield item


# <<< REMOVED Tavily patches >>>
# <<< ADDED patch for DDGS class used in the skill >>>
@patch("skills.web_search.DDGS")
async def test_web_search_skill_returns_results(mock_ddgs):
    """Verifica se a skill web_search retorna resultados usando DuckDuckGo."""
    query = "capital of France"
    expected_max_results = 5 # Default in the skill function

    # Configure the mock DDGS instance and its text method
    mock_ddgs_instance = MagicMock()
    # Mock the result of DDGS().text(), ensuring it's an iterable
    mock_ddgs_instance.text.return_value = [
        {'title': 'Paris - Wikipedia', 'href': 'https://en.wikipedia.org/wiki/Paris', 'body': 'Paris is the capital and most populous city of France...'},
        {'title': 'Official Website of Paris', 'href': 'https://www.paris.fr/en', 'body': 'The official website for the City of Paris.'}
    ]
    # Mock the context manager behavior (__enter__)
    mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance

    print(f"\nTesting web_search skill with query: '{query}' (using mocked DuckDuckGo)")

    # Call the skill (uses default max_results=5)
    result_dict = web_search(query=query)
    print(f"Raw result Dict: {result_dict}")

    assert result_dict is not None
    assert isinstance(result_dict, dict)
    assert result_dict["status"] == "success"
    assert result_dict["action"] == "web_search_results"
    assert "data" in result_dict
    assert result_dict["data"].get("search_method") == "DuckDuckGo" # Check method
    assert isinstance(result_dict["data"].get("results"), list)
    assert len(result_dict["data"]["results"]) == 2

    # Check mapped results
    first_result = result_dict["data"]["results"][0]
    assert first_result["title"] == "Paris - Wikipedia"
    assert first_result["url"] == "https://en.wikipedia.org/wiki/Paris" # Check 'url' (mapped from 'href')
    assert first_result["snippet"] == "Paris is the capital and most populous city of France..." # Check 'snippet' (mapped from 'body')

    # Verify the mocked DDGS().text() was called correctly
    mock_ddgs_instance.text.assert_called_once_with(query, max_results=expected_max_results)
    # Verify the context manager was used
    mock_ddgs.return_value.__enter__.assert_called_once()
    mock_ddgs.return_value.__exit__.assert_called_once()


# Para rodar este teste: execute 'pytest tests/test_web_search_skill.py' no terminal


# <<< REMOVED Tavily patches >>>
# <<< ADDED patch for DDGS class used in the skill >>>
@patch("skills.web_search.DDGS")
async def test_web_search_skill_no_results(mock_ddgs):
    """Verifica o comportamento quando a busca DuckDuckGo não retorna resultados."""
    query = "alskdjfhgqowieurytalskdjfhgqowieuryt_no_results_guaranteed_maybe"
    expected_max_results = 5 # Default in the skill function

    # Configure the mock DDGS instance and its text method to return empty list
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = [] # Simulate DDG returning no results
    mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance

    print(f"\nTesting web_search skill with query: '{query}' (expecting no results from DuckDuckGo)")

    # Call the skill
    result_dict = web_search(query=query)
    print(f"Raw result Dict (no results expected from DuckDuckGo): {result_dict}")

    assert result_dict is not None
    assert isinstance(result_dict, dict)
    assert result_dict["status"] == "success" # Should still be success
    assert result_dict["action"] == "web_search_results"
    assert "data" in result_dict
    assert result_dict["data"].get("search_method") == "DuckDuckGo"
    assert isinstance(result_dict["data"].get("results"), list)
    assert len(result_dict["data"]["results"]) == 0 # Expect empty results list

    # Verify the mocked DDGS().text() was called correctly
    mock_ddgs_instance.text.assert_called_once_with(query, max_results=expected_max_results)
    # Verify the context manager was used
    mock_ddgs.return_value.__enter__.assert_called_once()
    mock_ddgs.return_value.__exit__.assert_called_once()

# Teste para erro de API (se pudermos simular um)
# @pytest.mark.asyncio
# async def test_web_search_skill_api_error(mock_ddgs):
#     # Configure mock_ddgs_instance.text.side_effect = Exception("Simulated DDG Error")
#     # Call web_search
#     # Assert result_dict["status"] == "error"
#     pass
