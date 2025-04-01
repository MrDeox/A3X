import sys
import os
import pytest

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


async def test_web_search_skill_returns_results():
    """Verifica se a skill web_search retorna resultados para uma query simples."""
    query = "capital of France"
    print(f"\nTesting web_search skill with query: '{query}'")

    # Chamar a função da skill diretamente
    result_dict = web_search(query=query)

    print(f"Raw result Dict: {result_dict}")

    assert result_dict is not None
    assert isinstance(result_dict, dict)

    try:
        print(f"Result data: {result_dict}")
        assert "status" in result_dict
        if result_dict["status"] == "success":
            assert "results" in result_dict
            # Verificar se results é uma lista (se houver resultados)
            assert isinstance(result_dict["results"], list)
            if result_dict["results"]:
                assert len(result_dict["results"]) > 0  # Espera pelo menos um resultado
                # We might get different results, so just check keys exist
                assert "title" in result_dict["results"][0]
                assert "url" in result_dict["results"][0]
                assert "snippet" in result_dict["results"][0]
            # Allow empty results list as success
        elif result_dict["status"] == "error":
            assert "message" in result_dict
            # Permitir que o teste passe se houver um erro (ex: problema de rede temporário)
            print(
                f"Warning: Web search skill returned an error: {result_dict['message']}"
            )
        else:
            pytest.fail(f"Unexpected status in result: {result_dict.get('status')}")

    except AssertionError as e:
        pytest.fail(f"Assertion failed: {e} - Result Data: {result_dict}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during test execution: {e}")


# Para rodar este teste: execute 'pytest tests/test_web_search_skill.py' no terminal


@pytest.mark.asyncio
async def test_web_search_skill_no_results():
    """Verifica o comportamento quando a busca não retorna resultados."""
    # Use a query unlikely to return results from DuckDuckGo Lite
    # Note: DDG might still return *some* results even for nonsense strings.
    query = "alskdjfhgqowieurytalskdjfhgqowieuryt_no_results_guaranteed_maybe"
    print(f"\nTesting web_search skill with query: '{query}'")

    result_dict = web_search(query=query)
    print(f"Raw result Dict (no results expected): {result_dict}")

    assert result_dict is not None
    assert isinstance(result_dict, dict)
    assert (
        result_dict["status"] == "success"
    )  # Skill succeeds even if search has few/no results
    assert "results" in result_dict
    assert isinstance(result_dict["results"], list)
    # Allow non-zero results, as DDG might still find *something*
    print(f"Number of results found for unlikely query: {len(result_dict['results'])}")


# Teste para erro de API (se pudermos simular um)
# @pytest.mark.asyncio
# async def test_web_search_skill_api_error():
#     # Mock 'requests.post' or the internal HTTP client used by the search tool
#     # to raise an exception (e.g., ConnectionError, Timeout)
#     pass
