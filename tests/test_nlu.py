import pytest
import sys
import os

# Adiciona o diretório raiz ao path para importar os módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa a função que queremos testar
try:
    from core.nlu import interpret_command
except ImportError:
    pytest.skip("Não foi possível importar core.nlu", allow_module_level=True)

# Casos de teste para NLU (Comando, Intenção Esperada, Subconjunto de Entidades Esperadas)
NLU_TEST_CASES = [
    ("liste os arquivos aqui", "manage_files", {"action": "list"}),
    ("gere uma função python que soma dois numeros", "generate_code", {"language": "python", "construct_type": "function"}),
    ("busque na web sobre a história do Python", "search_web", {"query": "história do Python"}),
    ("lembre que meu email é teste@email.com", "remember_info", {"key": "meu email", "value": "teste@email.com"}),
    ("qual a senha do wifi?", "recall_info", {"key": "senha do wifi"}),
]

@pytest.mark.parametrize("command, expected_intent, expected_entities_subset", NLU_TEST_CASES)
def test_interpret_command_nlu(command, expected_intent, expected_entities_subset):
    """
    Testa a função interpret_command chamando o servidor LLM real.
    Verifica se a intenção e um subconjunto das entidades estão corretos.
    NOTA: Requer que o servidor llama.cpp esteja rodando!
    """
    print(f"\nTesting NLU for: '{command}'")
    interpretation = interpret_command(command, history=[])

    assert interpretation.get("intent") == expected_intent, \
        f"Intenção esperada '{expected_intent}', mas obteve '{interpretation.get('intent')}'"

    actual_entities = interpretation.get("entities", {})
    for key, value in expected_entities_subset.items():
        assert key in actual_entities and actual_entities[key] == value, \
            f"Entidade esperada '{key}={value}' não encontrada ou diferente em {actual_entities}"

    print(f"  -> NLU OK: {interpretation}") 