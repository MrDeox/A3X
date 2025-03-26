"""
Testes para o módulo Core do A³X.
"""

import pytest
from core import run_python_code

def test_codigo_simples():
    """Testa execução de código Python simples."""
    output = run_python_code("print(2 + 2)")
    assert "4" in output

def test_codigo_com_erro():
    """Testa tratamento de erros de execução."""
    output = run_python_code("print(variavel_inexistente)")
    assert "Variável não definida" in output

def test_codigo_proibido():
    """Testa bloqueio de código com palavras-chave proibidas."""
    with pytest.raises(ValueError):
        run_python_code("import os")

def test_codigo_longo():
    """Testa limite de tamanho do código."""
    with pytest.raises(ValueError):
        run_python_code("print(1)\n" * 1001) 