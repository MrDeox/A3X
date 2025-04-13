import pytest
import asyncio

from a3x.core.dynamic_replanner import dynamic_replan, detect_plan_stuck
from a3x.core.simulation import simulate_plan_execution, auto_evaluate_agent
from a3x.core.skill_autogen import autogen_skills_from_heuristics
from a3x.core.monetization_loop import MonetizationLoop

@pytest.mark.asyncio
async def test_dynamic_replanner_generates_new_plan():
    stuck_plan = ["Use the write_file tool to create directory X", "Use the write_file tool to create directory X"]
    execution_results = [
        {"status": "error", "data": {"message": "Directory already exists"}},
        {"status": "error", "data": {"message": "Directory already exists"}},
        {"status": "error", "data": {"message": "Directory already exists"}},
    ]
    assert detect_plan_stuck(execution_results)
    new_plan = await dynamic_replan(stuck_plan, execution_results)
    assert isinstance(new_plan, list)
    assert new_plan != stuck_plan

@pytest.mark.asyncio
async def test_simulation_module_runs_and_returns_results():
    plan = ["Use the write_file tool to create file test.txt", "Use the final_answer tool to confirm"]
    results = await simulate_plan_execution(plan)
    assert isinstance(results, list)
    assert all("status" in r for r in results)

@pytest.mark.asyncio
async def test_skill_autogen_creates_skill(tmp_path, monkeypatch):
    # Simula heurística de skill ausente
    heuristics = [{"type": "missing_skill_attempt", "skill_name": "nova_skill_teste"}]
    # Redireciona o diretório de skills auto-geradas para tmp_path
    monkeypatch.setattr("a3x.core.skill_autogen.save_skill_file", lambda name, code, skills_dir=tmp_path: tmp_path / f"{name}.py")
    generated = await autogen_skills_from_heuristics(heuristics)
    assert "nova_skill_teste" in generated

@pytest.mark.asyncio
async def test_monetization_loop_discovers_and_evaluates():
    loop = MonetizationLoop()
    opportunities = await loop.discover_opportunities()
    assert isinstance(opportunities, list)
    if opportunities:
        eval_result = await loop.evaluate_opportunity(opportunities[0])
        assert "retorno" in eval_result and "risco" in eval_result and "viabilidade" in eval_result

@pytest.mark.asyncio
async def test_monetization_loop_full_cycle():
    loop = MonetizationLoop()
    await loop.run(max_opportunities=1)
    # Não falha se rodar sem oportunidades reais, mas deve registrar tentativas

@pytest.mark.asyncio
async def test_auto_evaluate_agent_runs():
    plan = ["Use the write_file tool to create file test.txt", "Use the final_answer tool to confirm"]
    results = await auto_evaluate_agent([plan])
    assert "successes" in results and "failures" in results