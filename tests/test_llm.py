"""
Testes para o módulo LLM do A³X.
"""

import subprocess
import pytest
from llm.inference import run_llm

def test_run_llm_mock(monkeypatch):
    """Testa execução do LLM com mock."""
    def fake_run(cmd, capture_output, text, check):
        class FakeResult:
            stdout = "<|im_start|>assistant Tudo certo!"
        return FakeResult()
    
    monkeypatch.setattr(subprocess, "run", fake_run)
    resposta = run_llm("Qual é a capital do Brasil?")
    assert "Tudo certo" in resposta 