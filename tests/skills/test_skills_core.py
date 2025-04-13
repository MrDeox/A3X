import pytest
import asyncio
from pathlib import Path

# Importa as skills core para teste
from a3x.skills.core.study import study_skill
from a3x.skills.core.llm_error_diagnosis import llm_error_diagnosis_skill
from a3x.skills.core.simulate_plan import simulate_plan_skill

@pytest.mark.asyncio
async def test_study_skill_basic():
    result = await study_skill(
        task="Aprender sobre redes neurais",
        context={"dominio": "IA"},
        vision=None,
        resources=["https://pt.wikipedia.org/wiki/Rede_neural_artificial"],
        llm_url=None
    )
    assert result["status"] == "success"
    assert "study" in result["data"]

@pytest.mark.asyncio
async def test_llm_error_diagnosis_skill_basic():
    result = await llm_error_diagnosis_skill(
        error_message="FileNotFoundError: arquivo.txt não encontrado",
        traceback="Traceback (most recent call last): ...",
        execution_context={"step": "read_file"},
        llm_url=""
    )
    assert result["status"] == "success"
    assert "diagnosis" in result["data"]
    assert "suggested_actions" in result["data"]

@pytest.mark.asyncio
async def test_simulate_plan_skill_basic():
    plan = [
        "Use the write_file tool to create file test.txt",
        "Use the final_answer tool to confirm"
    ]
    result = await simulate_plan_skill(
        plan=plan,
        context=None,
        heuristics=None,
        llm_url=None
    )
    assert result["status"] == "success"
    assert "simulation_results" in result["data"]