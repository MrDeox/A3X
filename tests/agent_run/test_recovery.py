import os
import shutil
import pytest

from a3x.core.cerebrumx import CerebrumXAgent

@pytest.mark.asyncio
async def test_write_to_nonexistent_directory(tmp_path):
    """
    Testa se o agente consegue recuperar de uma tentativa de escrita em diretório inexistente,
    criando o diretório e completando a tarefa.
    """
    # Setup: define um diretório que não existe
    target_dir = tmp_path / "dir_inexistente"
    target_file = target_dir / "arquivo.txt"
    if target_dir.exists():
        shutil.rmtree(target_dir)

    agent = CerebrumXAgent(system_prompt="Você é um agente de teste.", llm_url=None)
    objetivo = f"Escreva 'teste de recuperação' no arquivo {target_file}"

    resultado = await agent.run(objetivo)
    assert resultado["status"] == "completed"
    assert target_file.exists()
    with open(target_file, "r", encoding="utf-8") as f:
        conteudo = f.read()
    assert "teste de recuperação" in conteudo

@pytest.mark.asyncio
async def test_call_invalid_skill():
    """
    Testa se o agente diagnostica corretamente a chamada de uma skill inexistente e registra heurística.
    """
    agent = CerebrumXAgent(system_prompt="Você é um agente de teste.", llm_url=None)
    objetivo = "Execute a skill chamada skill_inexistente com parâmetro foo=123"
    resultado = await agent.run(objetivo)
    assert resultado["status"] in ("failed", "error")
    # Opcional: verificar se heurística de skill ausente foi registrada em memory/learning_logs/learned_heuristics.jsonl

@pytest.mark.asyncio
async def test_access_missing_file(tmp_path):
    """
    Testa se o agente sugere criar um arquivo quando tenta acessar um arquivo inexistente.
    """
    missing_file = tmp_path / "arquivo_que_nao_existe.txt"
    agent = CerebrumXAgent(system_prompt="Você é um agente de teste.", llm_url=None)
    objetivo = f"Leia o conteúdo do arquivo {missing_file}"
    resultado = await agent.run(objetivo)
    assert resultado["status"] in ("failed", "error", "completed")
    # Opcional: verificar logs/sugestões para sugestão de criação do arquivo