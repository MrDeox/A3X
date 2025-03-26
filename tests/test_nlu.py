import pytest
import sys
import os

# Adiciona o diretório raiz ao path para importar assistant_cli
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa a função que queremos testar (assume que está em assistant_cli.py)
# Se você renomeou o arquivo, ajuste o import
try:
    from assistant_cli import interpret_command
except ImportError:
    pytest.skip("Não foi possível importar assistant_cli.py", allow_module_level=True)

# Casos de teste para NLU (Comando, Intenção Esperada, Subconjunto de Entidades Esperadas)
# Usamos subconjunto de entidades porque a extração pode variar um pouco
NLU_TEST_CASES = [
    ("liste os arquivos aqui", "manage_files", {"action": "list"}),
    ("gere uma função python que soma dois numeros", "generate_code", {"language": "python", "construct_type": "function"}),
    ("qual a previsão do tempo para hoje?", "search_info", {"topic": "previsão do tempo"}),
    # Adicione mais casos de teste aqui cobrindo diferentes intenções e entidades
    ("cria um script chamado setup.py", "generate_code", {"file_name": "setup.py"}),
]

@pytest.mark.parametrize("command, expected_intent, expected_entities_subset", NLU_TEST_CASES)
def test_interpret_command_nlu(command, expected_intent, expected_entities_subset):
    """
    Testa a função interpret_command chamando o servidor LLM real.
    Verifica se a intenção e um subconjunto das entidades estão corretos.
    NOTA: Requer que o servidor llama.cpp esteja rodando!
    """
    print(f"\nTesting NLU for: '{command}'") # Mostra qual comando está sendo testado
    interpretation = interpret_command(command)

    assert interpretation.get("intent") == expected_intent, \
        f"Intenção esperada '{expected_intent}', mas obteve '{interpretation.get('intent')}'"

    actual_entities = interpretation.get("entities", {})
    for key, value in expected_entities_subset.items():
        assert key in actual_entities and actual_entities[key] == value, \
            f"Entidade esperada '{key}={value}' não encontrada ou diferente em {actual_entities}"

    print(f"  -> NLU OK: {interpretation}") # Mostra a interpretação bem-sucedida 