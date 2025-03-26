import pytest
from skills.modify_code import skill_modify_code

def test_skill_modify_code_success(mocker):
    """Testa o cenário principal de modificação de código com sucesso."""
    # Código de exemplo para testar
    original_code = """def calcular_soma(a, b):
    return a + b"""

    # Histórico com o código original
    history = [
        {"role": "user", "content": "gere uma função que soma dois números"},
        {"role": "assistant", "content": f"Código python gerado:\n---\n{original_code}\n---"}
    ]

    # Simular resposta do LLM
    mock_response = {
        "content": """```python
def calcular_soma(a, b):
    # Calcula a soma de dois números
    return a + b
```"""
    }

    # Mock da chamada requests.post
    mock_post = mocker.patch('requests.post')
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.raise_for_status.return_value = None

    # Entidades de teste
    entities = {
        "target": "função calcular_soma",
        "modification": "adicione um comentário explicativo"
    }

    # Executar a skill
    result = skill_modify_code(entities, "adicione um comentário à função calcular_soma", history=history)

    # Verificações
    assert result["status"] == "success"
    assert result["action"] == "code_modified"
    assert "modified_code" in result["data"]
    assert "# Calcula a soma de dois números" in result["data"]["modified_code"]
    assert "def calcular_soma(a, b):" in result["data"]["modified_code"]
    assert "return a + b" in result["data"]["modified_code"]

    # Verificar se requests.post foi chamado com o payload correto
    mock_post.assert_called_once()
    call_args = mock_post.call_args[1]
    assert "json" in call_args
    payload = call_args["json"]
    assert "prompt" in payload
    assert "```python" in payload["prompt"]
    assert original_code in payload["prompt"]
    assert "adicione um comentário explicativo" in payload["prompt"]
    assert payload["temperature"] == 0.3
    assert payload["n_predict"] == 2048
    assert payload["stop"] == ["```"]

def test_skill_modify_code_no_modification():
    """Testa o caso onde não há instrução de modificação."""
    entities = {
        "target": "algum código",
        "modification": None
    }

    result = skill_modify_code(entities, "comando sem modificação", history=[])

    assert result["status"] == "error"
    assert result["action"] == "modify_code_failed"
    assert "Não entendi qual modificação fazer" in result["data"]["message"]

def test_skill_modify_code_llm_error(mocker):
    """Testa o caso onde o LLM retorna um erro."""
    # Código de exemplo para testar
    original_code = """def exemplo():
    pass"""

    # Histórico com o código original
    history = [
        {"role": "user", "content": "gere uma função de exemplo"},
        {"role": "assistant", "content": f"Código python gerado:\n---\n{original_code}\n---"}
    ]

    # Mock da chamada requests.post para simular erro
    mock_post = mocker.patch('requests.post')
    mock_post.side_effect = Exception("Erro de conexão com LLM")

    entities = {
        "target": "função exemplo",
        "modification": "alguma modificação"
    }

    result = skill_modify_code(entities, "comando com erro", history=history)

    assert result["status"] == "error"
    assert result["action"] == "modify_code_failed"
    assert "Erro inesperado ao modificar código: Erro de conexão com LLM" == result["data"]["message"] 