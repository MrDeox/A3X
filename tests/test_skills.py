import pytest
import sys
import os

# Adiciona o diretório raiz ao path para importar os módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    # from skills.code_generation import generate_code # F401
    pass # Keep try block structure
except ImportError:
    pytest.skip("Não foi possível importar skills.code_generation", allow_module_level=True)

# Simula uma resposta HTTP do servidor LLM contendo código e texto extra
MOCK_LLM_CODE_RESPONSE = {
    "content": """
```python
# Código de exemplo
def minha_funcao():
    print("Olá do teste!")
```

Aqui está uma explicação que não deveria aparecer...
Mais texto extra.
"""
}

# def test_skill_generate_code_extraction(mocker):
#     \"\"\"
#     Testa se skill_generate_code extrai corretamente o código
#     de uma resposta simulada do LLM (usando mocking).
#     \"\"\"
#     # Simula a função requests.post
#     mock_post = mocker.patch('skills.generate_code.requests.post')
#
#     # Configura o mock para retornar uma resposta simulada com status 200
#     mock_response = MagicMock()
#     mock_response.status_code = 200
#     mock_response.json.return_value = MOCK_LLM_CODE_RESPONSE
#     mock_response.raise_for_status = MagicMock()
#     mock_post.return_value = mock_response
#
#     # Entidades de exemplo para a skill
#     test_entities = {\"language\": \"python\"}
#     test_command = \"comando de teste\"
#
#     # Chama a skill
#     result = skill_generate_code(test_entities, test_command, intent=\"generate_code\")
#
#     # Verifica o formato do resultado
#     assert result[\"status\"] == \"success\", \"O status do resultado deve ser 'success'\"
#     assert \"data\" in result, \"O resultado deve conter a chave 'data'\"
#     assert \"code\" in result[\"data\"], \"O resultado deve conter a chave 'code' em 'data'\"
#
#     # Verifica o conteúdo do código
#     expected_code = '# Código de exemplo\\ndef minha_funcao():\\n    print(\"Olá do teste!\")'
#
#     # Verifica o conteúdo do código
#     expected_code = '# Código de exemplo\ndef minha_funcao():\n    print("Olá do teste!")'
#     assert expected_code in result["data"]["code"], "O código esperado não foi encontrado no resultado."
#     assert "Aqui está uma explicação" not in result["data"]["code"], "Texto extra foi incluído indevidamente no resultado."
#     assert "Mais texto extra" not in result["data"]["code"], "Texto extra foi incluído indevidamente no resultado."
#
#     # Verifica se requests.post foi chamado uma vez
#     mock_post.assert_called_once()
