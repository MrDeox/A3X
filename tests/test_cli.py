"""
Testes para o módulo CLI do A³X.
"""

import pytest
from cli import execute

def test_comando_valido():
    """Testa execução de comando válido."""
    output = execute("echo testando")
    assert "testando" in output

def test_comando_bloqueado():
    """Testa bloqueio de comandos perigosos."""
    with pytest.raises(ValueError):
        execute("rm -rf /")

def test_comando_invalido():
    """Testa rejeição de comandos com caracteres inválidos."""
    with pytest.raises(ValueError):
        execute("!@#")

def test_sem_output():
    """Testa execução sem captura de saída."""
    execute("mkdir -p temp_test", capture_output=False) 