# tests/test_reflection_skill.py
import pytest
from unittest.mock import patch, AsyncMock
import sys
import os

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from skills.reflection import reflect_plan_step, _parse_reflection_output, REFLECT_STEP_PROMPT_TEMPLATE

# --- Tests for _parse_reflection_output (Synchronous Helper) ---

def test_parse_reflection_output_execute():
    response = "Decision: execute\nJustification: The step seems safe and necessary."
    expected = {"decision": "execute", "justification": "The step seems safe and necessary."}
    assert _parse_reflection_output(response) == expected

def test_parse_reflection_output_modify():
    response = "Decision: modify\nJustification: Need to specify the filename."
    expected = {"decision": "modify", "justification": "Need to specify the filename."}
    assert _parse_reflection_output(response) == expected

def test_parse_reflection_output_skip():
    response = "Decision: skip\nJustification: This step is redundant."
    expected = {"decision": "skip", "justification": "This step is redundant."}
    assert _parse_reflection_output(response) == expected

def test_parse_reflection_output_case_insensitive():
    response = "decision: EXECUTE\njustification: Looks good."
    expected = {"decision": "execute", "justification": "Looks good."}
    assert _parse_reflection_output(response) == expected

def test_parse_reflection_output_extra_whitespace():
    response = " Decision :  modify \n Justification :\tNeeds more detail. "
    expected = {"decision": "modify", "justification": "Needs more detail."}
    assert _parse_reflection_output(response) == expected

def test_parse_reflection_output_multiline_justification():
    response = "Decision: skip\nJustification: Step is too risky.\nIt might delete important files."
    expected = {"decision": "skip", "justification": "Step is too risky.\nIt might delete important files."}
    assert _parse_reflection_output(response) == expected

def test_parse_reflection_output_missing_justification():
    response = "Decision: execute"
    expected = {"decision": "execute", "justification": "No justification provided."}
    assert _parse_reflection_output(response) == expected

def test_parse_reflection_output_missing_decision():
    response = "Justification: Seems okay."
    expected = {"decision": "unknown", "justification": "Seems okay."}
    assert _parse_reflection_output(response) == expected

def test_parse_reflection_output_malformed():
    response = "Let's execute this. Justification: Go!"
    expected = {"decision": "execute", "justification": "Go!"} # Fallback guess
    assert _parse_reflection_output(response) == expected

def test_parse_reflection_output_empty():
    response = ""
    expected = {"decision": "unknown", "justification": "No justification provided."}
    assert _parse_reflection_output(response) == expected

# --- Tests for reflect_plan_step (Async Skill) ---

@pytest.mark.asyncio
@patch('skills.reflection.call_llm', new_callable=AsyncMock)
async def test_reflect_plan_step_success_execute(mock_call_llm):
    """Tests successful reflection deciding to execute."""
    step = "Read 'results.txt'"
    simulated_outcome = "Agent will successfully read the file content."
    context = {"previous_step": "generated results.txt"}
    llm_response = "Decision: execute\nJustification: The file exists and reading it is the logical next step."
    mock_call_llm.return_value = llm_response

    expected_prompt = REFLECT_STEP_PROMPT_TEMPLATE.format(step=step, simulated_outcome=simulated_outcome, context=context)

    result = await reflect_plan_step(step=step, simulated_outcome=simulated_outcome, context=context)

    mock_call_llm.assert_awaited_once_with(expected_prompt, stream=False)
    assert result['status'] == "success"
    assert result['decision'] == "execute"
    assert result['justification'] == "The file exists and reading it is the logical next step."
    assert result['confidence'] == "Média"
    assert result['error_message'] is None

@pytest.mark.asyncio
@patch('skills.reflection.call_llm', new_callable=AsyncMock)
async def test_reflect_plan_step_success_modify(mock_call_llm):
    """Tests successful reflection deciding to modify."""
    step = "Write analysis to file."
    simulated_outcome = "Agent will attempt to write analysis, but filename is missing."
    context = {"analysis_complete": True}
    llm_response = "Decision: modify\nJustification: The step is necessary but lacks a specific filename. Suggest adding 'analysis_report.md'."
    mock_call_llm.return_value = llm_response

    result = await reflect_plan_step(step=step, simulated_outcome=simulated_outcome, context=context)

    assert result['status'] == "success"
    assert result['decision'] == "modify"
    assert "lacks a specific filename" in result['justification']

@pytest.mark.asyncio
@patch('skills.reflection.call_llm', new_callable=AsyncMock)
async def test_reflect_plan_step_success_skip(mock_call_llm):
    """Tests successful reflection deciding to skip."""
    step = "List files in root directory."
    simulated_outcome = "Agent will list files, potentially many."
    context = {"files_already_listed": ["file1", "file2"]}
    llm_response = "Decision: skip\nJustification: Files were listed in a previous step, this is redundant."
    mock_call_llm.return_value = llm_response

    result = await reflect_plan_step(step=step, simulated_outcome=simulated_outcome, context=context)

    assert result['status'] == "success"
    assert result['decision'] == "skip"
    assert "redundant" in result['justification']

@pytest.mark.asyncio
@patch('skills.reflection.call_llm', new_callable=AsyncMock)
async def test_reflect_plan_step_llm_exception(mock_call_llm):
    """Tests when call_llm raises an exception during reflection."""
    step = "Finalize report."
    simulated_outcome = "Agent will compile final report."
    context = {}
    error_message = "LLM connection timeout"
    mock_call_llm.side_effect = Exception(error_message)

    result = await reflect_plan_step(step=step, simulated_outcome=simulated_outcome, context=context)

    assert result['status'] == "error"
    assert result['decision'] == "unknown"
    assert "Failed to reflect on step due to LLM error" in result['justification']
    assert error_message in result['error_message']

@pytest.mark.asyncio
@patch('skills.reflection.call_llm', new_callable=AsyncMock)
async def test_reflect_plan_step_llm_invalid_response(mock_call_llm):
    """Tests when the LLM response cannot be parsed correctly."""
    step = "Check status."
    simulated_outcome = "Agent checks system status."
    context = {}
    llm_response = "Everything looks okay, proceed."
    mock_call_llm.return_value = llm_response # Malformed, no Decision/Justification

    result = await reflect_plan_step(step=step, simulated_outcome=simulated_outcome, context=context)

    assert result['status'] == "success" # Skill itself succeeds
    assert result['decision'] == "unknown" # Parsing fails
    assert result['justification'] == "No justification provided." # Parsing fails 