"""
Testes para o módulo de memória do A³X.
"""

import pytest
from memory import store, retrieve

def test_store_and_retrieve():
    """Testa armazenamento e recuperação básica."""
    store("teste_chave", "valor de teste", overwrite=True)
    assert retrieve("teste_chave") == "valor de teste"

def test_chave_invalida():
    """Testa rejeição de chaves com espaços."""
    with pytest.raises(ValueError):
        store("chave com espaço", "valor")

def test_valor_grande():
    """Testa limite de tamanho do valor."""
    valor = "x" * (10 * 1024 + 1)  # 10KB + 1 byte
    with pytest.raises(ValueError):
        store("chave_grande", valor)

def test_sobrescrita_bloqueada():
    """Testa proteção contra sobrescrita não autorizada."""
    store("nao_sobrescreva", "original", overwrite=True)
    with pytest.raises(KeyError):
        store("nao_sobrescreva", "novo", overwrite=False) 