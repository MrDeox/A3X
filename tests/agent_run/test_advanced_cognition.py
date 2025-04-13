import pytest
import asyncio
from pathlib import Path

from a3x.core.dynamic_replanner import dynamic_replan, detect_plan_stuck
from a3x.core.simulation import simulate_plan_execution, auto_evaluate_agent
from a3x.core.skill_autogen import detect_missing_skills, propose_and_generate_skill, save_skill_file, autogen_skills_from_heuristics
from a3x.core.monetization_loop import MonetizationLoop
from a3x.core.llm_interface import LLMInterface
from a3x.core.learning_logs import log_heuristic_with_traceability

# Mock LLM URL for testing
TEST_LLM_URL = "http://127.0.0.1:9999/completion" # Use a dummy or mock server URL

# Helper to setup logger if needed
import logging
logging.basicConfig(level=logging.DEBUG)

# Sample Heuristics Data (Simulating what might be logged)
SAMPLE_HEURISTICS = [
    {"type": "missing_skill_attempt", "skill_name": "calculate_fibonacci", "context": "User asked for 50th Fibonacci number"},
    {"type": "parsing_fallback", "action_inferred": "summarize_text", "context": "LLM provided text instead of action"},
    {"type": "missing_skill_attempt", "skill_name": "calculate_fibonacci", "context": "Need calculation for sequence"},
    {"type": "other_heuristic", "details": "..."}
]

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
async def test_skill_autogen_pipeline(tmp_path):
    """Tests the full skill autogeneration pipeline."""
    # 1. Setup: Instantiate LLMInterface (use a test URL)
    llm_interface = LLMInterface(llm_url=TEST_LLM_URL) # <-- INSTANTIATE
    
    # Use tmp_path provided by pytest for saving generated skills
    autogen_dir = tmp_path / "autogen_skills"
    
    # Mock save_skill_file to use tmp_path
    original_save = save_skill_file
    def mock_save(skill_name, code, skills_dir=None):
        # Ignore passed skills_dir, always use tmp_path
        return original_save(skill_name, code, skills_dir=autogen_dir)
    save_skill_file = mock_save

    # 2. Run the pipeline
    generated_skills = await autogen_skills_from_heuristics(
        SAMPLE_HEURISTICS, 
        llm_interface=llm_interface # <-- PASS INSTANCE
        # llm_url removed
    )

    # 3. Assertions (adjust based on expected mock LLM behavior)
    # For now, just check if it attempted to generate the detected missing skills
    # A real test would involve mocking the LLM response
    expected_missing = ["calculate_fibonacci", "summarize_text"]
    assert sorted(generated_skills) == sorted(expected_missing)
    
    # Check if files were created (assuming mock LLM returns some code)
    # This part depends heavily on mocking the LLM response correctly
    # For this example, we assume the mock LLM returned non-empty code
    # assert (autogen_dir / "calculate_fibonacci.py").is_file()
    # assert (autogen_dir / "summarize_text.py").is_file()
    
    # Restore original save function if necessary for other tests
    save_skill_file = original_save

# You might want separate tests for detect_missing_skills, propose_and_generate_skill (with mocked LLM), and save_skill_file

@pytest.mark.asyncio
async def test_propose_and_generate_skill_mocked(mocker):
     """Tests propose_and_generate_skill with a mocked LLM call."""
     # Setup LLMInterface
     llm_interface = LLMInterface(llm_url=TEST_LLM_URL)

     # Mock the llm_interface.call_llm method
     mock_code = "import math\n\n@skill(name='test_skill')\ndef test_skill():\n    return math.pi" 
     async def mock_llm_call(*args, **kwargs):
          yield mock_code # Simulate LLM returning code
     
     mocker.patch.object(llm_interface, 'call_llm', side_effect=mock_llm_call)
     
     generated_code = await propose_and_generate_skill(
          skill_name="test_skill", 
          llm_interface=llm_interface, # Pass instance
          context={"reason": "testing"}
     )
     
     assert generated_code.strip() == mock_code.strip()
     llm_interface.call_llm.assert_called_once() # Verify LLM was called

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