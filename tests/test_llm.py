"""
Testes para o módulo LLM do A³X.
"""

import subprocess
import pytest
import os
from pathlib import Path
from llm.inference import run_llm

# Verifica se o binário do LLM existe
llm_binary = Path('bin/llama-cli')
has_llm = llm_binary.exists()

@pytest.mark.skipif(not has_llm, reason="Binário do LLM não encontrado")
def test_run_llm_mock(monkeypatch):
    """Testa execução do LLM com mock."""
    def fake_run(cmd, capture_output, text, check):
        class FakeResult:
            stdout = "<|im_start|>assistant Tudo certo!"
        return FakeResult()
    
    monkeypatch.setattr(subprocess, "run", fake_run)
    resposta = run_llm("Qual é a capital do Brasil?")
    assert "Tudo certo" in resposta 