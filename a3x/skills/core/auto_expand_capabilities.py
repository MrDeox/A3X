import logging
import json
import os
import inspect
import asyncio
from typing import Dict, Any, List, Optional
import re
from pathlib import Path

# Core framework imports
# Core imports
from a3x.core.skills import skill, get_skill_registry, get_skill_descriptions
# from a3x.core.prompt_builder import build_skill_generation_prompt # Function missing
from a3x.core.llm_interface import LLMInterface
from a3x.core.context import Context
from a3x.core.learning_logs import load_recent_reflection_logs
from a3x.skills.core.call_skill_by_name import call_skill_by_name
from a3x.core.agent import _ToolExecutionContext
from a3x.core.config import LEARNING_LOGS_DIR, HEURISTIC_LOG_FILE

logger = logging.getLogger(__name__)

# --- Constants ---

# LLM Prompt to analyze logs and suggest skills in JSON format
# Note: Using single quotes for the outer string to avoid excessive escaping inside the JSON example.
LLM_ANALYSIS_PROMPT_TEMPLATE = '''
Analyze the following A³X agent log entry to identify potential gaps in its autonomy.
Log Entry:
```json
{log_entry_json}
```

Based *only* on the provided log entry, is there a clear failure or limitation in the agent's autonomous capabilities that could be addressed by creating a *new, specific skill*?

If YES, describe the proposed new skill ONLY in the following JSON format. Ensure valid JSON.
If NO, or if the issue isn't solvable by a *new skill*, respond with an empty JSON object: {{}}.

JSON Format for New Skill Suggestion:
{{
  "skill_name": "suggested_snake_case_name",
  "reason": "Brief explanation of the autonomy gap observed in the log.",
  "suggestion": "Detailed description of the new skill's function and purpose.",
  "parameters": {{ "param1": "type", "param2": "type" }},
  "example_usage": "Action: {{ \\"tool_name\\": \\"suggested_snake_case_name\\", \\"tool_input\\": {{...}} }}",
  "auto_create": boolean
}}

Respond ONLY with the JSON object (either the suggestion or {{}}). Do not add explanations before or after.
'''

# --- Helper Functions ---

def _check_for_tests(skill_info: Dict[str, Any], test_dir_base: str = "tests/unit/skills") -> bool:
    """Checks if a test file exists for a given skill."""
    source_file = skill_info.get("source_file")
    if not source_file:
        return False # Cannot determine without source file

    # Attempt to construct a plausible test file path
    try:
        # Get path relative to the skills directory (e.g., core/list_skills.py)
        relative_path = os.path.relpath(source_file, start="a3x/skills") 
        base_name = os.path.splitext(relative_path)[0]
        # Construct potential test path: tests/unit/skills/core/test_list_skills.py
        test_path = Path(test_dir_base) / f"test_{base_name}.py"
        return test_path.exists()
    except ValueError:
        # Handle cases where relpath fails (e.g., different drives on Windows)
        logger.warning(f"Could not determine relative path for test check: {source_file}")
        return False # Assume no test if path fails
    return False

def _check_llm_usage(skill_info: Dict[str, Any]) -> bool:
    """Rudimentary check if a skill function *might* use llm_interface.call_llm or older patterns."""
    skill_func = skill_info.get("function")
    if not skill_func or not inspect.isfunction(skill_func):
        return False
    try:
        source = inspect.getsource(skill_func)
        # Check for new pattern and old patterns
        return "llm_interface.call_llm" in source or "ctx.llm_call" in source or "call_llm" in source
    except (OSError, TypeError):
        logger.warning(f"Could not inspect source for skill: {skill_info.get('name', 'unknown')}")
        return False

def _check_learning_mechanism(skill_info: Dict[str, Any]) -> bool:
    """Rudimentary check if a skill *might* save learning (e.g., uses memory or specific learning skills)."""
    skill_func = skill_info.get("function")
    if not skill_func or not inspect.isfunction(skill_func):
        return False
    try:
        source = inspect.getsource(skill_func)
        # Look for patterns like ctx.mem, ctx.memory, save_to_memory, learn_from...
        # This is very basic and needs refinement based on actual patterns.
        return "ctx.mem" in source or "ctx.memory" in source or "save_to_memory" in source or "learn_" in source
    except (OSError, TypeError):
        logger.warning(f"Could not inspect source for skill: {skill_info.get('name', 'unknown')}")
        return False

# --- Main Skill ---

@skill(
    name="auto_expand_capabilities",
    description="Analyzes agent performance and failure patterns to suggest or automatically generate new skills.",
    parameters={
        "context": {"type": Context, "description": "Execution context for LLM, memory, and skill registry access."},
        "analysis_depth": {"type": Optional[str], "default": "basic", "description": "Depth of analysis ('basic', 'detailed')."}
    }
)
async def auto_expand_capabilities(
    context: Context,
    analysis_depth: Optional[str] = "basic"
) -> Dict[str, Any]:
    """
    Analyzes A³X operational logs using the LLMInterface from context...
    """
    ctx.logger.info(f"Starting metacognitive analysis using last {num_logs_to_analyze} logs...")
    llm_interface = ctx.llm_interface # Get interface from context
    if not llm_interface:
        ctx.logger.error("LLMInterface not found in execution context.")
        return {"status": "error", "summary": "Internal error: LLMInterface missing.", "opportunities_found": [], "actions_taken": []}

    opportunities = []
    actions_taken = []
    logs_analyzed_count = 0
    suggestions_found = 0

    # --- Step 1: Load Relevant Logs ---
    ctx.logger.info("Loading recent reflection logs...")
    reflection_logs: List[Dict[str, Any]] = []
    try:
        reflection_logs = load_recent_reflection_logs(n=num_logs_to_analyze)
        logs_analyzed_count = len(reflection_logs)
        ctx.logger.info(f"Loaded {logs_analyzed_count} reflection logs.")
    except FileNotFoundError:
         ctx.logger.warning("Reflection log file not found. Skipping.")
    except Exception as e:
        ctx.logger.exception("Error loading reflection logs:")
    all_logs = reflection_logs # Use only reflection logs for now
    if not all_logs:
        ctx.logger.warning("No relevant logs found to analyze.")
        return {
            "status": "warning",
            "summary": "No logs found for analysis.",
            "opportunities_found": [],
            "actions_taken": []
        }

    # TODO: Implement loading for other relevant log types (e.g., execution errors, failed plans)
    #       and combine them into 'all_logs'.
    # error_logs = load_error_logs(n=num_logs_to_analyze)
    # all_logs = reflection_logs + error_logs

    # --- Step 2: Analyze Logs with LLM ---
    ctx.logger.info(f"Analyzing {len(all_logs)} log entries with LLM...")
    for i, log_entry in enumerate(all_logs):
        log_identifier = f"Log {i+1}/{len(all_logs)}"
        ctx.logger.debug(f"Analyzing {log_identifier}...")
        try:
            # Prepare the log entry string safely for the prompt
            try:
                 log_entry_str = json.dumps(log_entry, indent=2, ensure_ascii=False)
            except TypeError as te:
                 ctx.logger.warning(f"Could not serialize log entry {log_identifier}, using repr: {te}")
                 log_entry_str = repr(log_entry)
            prompt = LLM_ANALYSIS_PROMPT_TEMPLATE.format(log_entry_json=log_entry_str)
            messages = [{"role": "user", "content": prompt}]

            # --- Updated LLM Call --- Use llm_interface from context
            llm_response_str = ""
            response_stream = llm_interface.call_llm(messages=messages, stream=False) # Expect JSON, no stream needed
            async for chunk in response_stream:
                 llm_response_str += chunk
            # --- End LLM Call --- 

            # --- Step 2a: Parse LLM response ---
            try:
                cleaned_response = llm_response_str.strip()
                if cleaned_response.startswith("```json"): cleaned_response = cleaned_response[7:]
                if cleaned_response.startswith("```"): cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith("```"): cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()

                if not cleaned_response or cleaned_response == "{}": continue # Skip empty

                suggestion_data = json.loads(cleaned_response)

                if suggestion_data and isinstance(suggestion_data, dict) and "skill_name" in suggestion_data:
                    ctx.logger.info(f"LLM suggested skill from {log_identifier}: {suggestion_data.get('skill_name')}")
                    suggestions_found += 1
                    required_keys = ["skill_name", "reason", "suggestion", "auto_create"]
                    if not all(k in suggestion_data for k in required_keys): continue # Skip incomplete
                    suggestion_data["source_log_preview"] = dict(list(log_entry.items())[:3]) 
                    opportunities.append(suggestion_data)
                    # --- Step 3: Trigger Skill Proposal --- 
                    if suggestion_data.get("auto_create") is True:
                        ctx.logger.info(f"Triggering skill proposal for: {suggestion_data['skill_name']}")
                        try:
                             proposal_args = { # Arguments for propose_skill_from_gap
                                 "skill_name": suggestion_data['skill_name'],
                                 "reason": suggestion_data['reason'],
                                 "suggestion_description": suggestion_data['suggestion'],
                                 "parameters_json": json.dumps(suggestion_data.get('parameters', {})),
                                 "example_usage": suggestion_data.get('example_usage', ''),
                                 "source_analysis_log_preview": json.dumps(suggestion_data["source_log_preview"]), 
                                 "source_skill": "auto_expand_capabilities"
                             }
                             proposal_result = await call_skill_by_name(ctx, skill_name="propose_skill_from_gap", skill_args=proposal_args)
                             action_msg = f"Called 'propose_skill_from_gap' for '{suggestion_data['skill_name']}'. Result: {proposal_result}"
                             ctx.logger.info(action_msg)
                             actions_taken.append({"action": "call_propose_skill_from_gap", "target_skill": suggestion_data['skill_name'], "result": proposal_result})
                        except Exception as call_e:
                             error_msg = f"Error calling propose_skill_from_gap: {call_e}"
                             ctx.logger.exception(error_msg) 
                             actions_taken.append({"action": "call_propose_skill_from_gap", "target_skill": suggestion_data['skill_name'], "status": "error", "message": error_msg})
                    else:
                         ctx.logger.info(f"Suggestion '{suggestion_data['skill_name']}' found but auto_create is false.")
                elif not suggestion_data: # Parsed as empty dict
                     ctx.logger.debug(f"LLM analysis of {log_identifier} resulted in no suggestion.")
            except json.JSONDecodeError as json_e:
                ctx.logger.error(f"Failed to parse LLM JSON for {log_identifier}: {json_e}. Response: '{llm_response_str}'")
            except Exception as parse_e:
                 ctx.logger.error(f"Error processing LLM response for {log_identifier}: {parse_e}")
        except asyncio.TimeoutError: # Handle timeout from wait_for
            ctx.logger.error(f"LLM call timed out analyzing {log_identifier}.")
        except Exception as llm_e: # Catch other LLM call errors
            ctx.logger.exception(f"Error during LLM call for {log_identifier}: {llm_e}")

    # --- Step 4: Summarize and Return ---
    summary = f"Metacognitive analysis complete. Analyzed {logs_analyzed_count} logs. Found {suggestions_found} potential new skill opportunities. Triggered {len(actions_taken)} actions."
    ctx.logger.info(summary)
    return {
        "status": "success",
        "summary": summary,
        "opportunities_found": opportunities,
        "actions_taken": actions_taken
    }

# Example of how to potentially run this (requires an Agent setup)
# async def main():
#     # Minimal setup for testing (replace with your actual agent init)
#     from a3x.core.context import SkillContext
#     from a3x.core.log_config import setup_logging # Assuming you have this
#     setup_logging()
#     mock_ctx = SkillContext(logger=logging.getLogger("test_auto_expand"))
#     # Need to ensure skills are loaded before calling
#     from a3x.core.skills import load_skills
#     load_skills()
#
#     result = await auto_expand_capabilities(mock_ctx)
#     print(json.dumps(result, indent=2))
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main()) 