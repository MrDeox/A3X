import logging
import json
import os
import inspect
import asyncio
from typing import Dict, Any, List, Optional
import re

# Core framework imports
# Core imports
from a3x.core.skills import skill, get_skill_registry
# from a3x.core.prompt_builder import build_skill_generation_prompt # Function missing
from a3x.core.llm_interface import call_llm
# from a3x.core.context import SkillContext # Assuming SkillContext is defined here
from a3x.core.learning_logs import load_recent_reflection_logs
from a3x.skills.core.call_skill_by_name import call_skill_by_name

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

def _check_for_tests(skill_info: Dict[str, Any], test_dir_base: str = "tests/skills") -> bool:
    """Checks if a corresponding test file likely exists for a skill."""
    source_file = skill_info.get("source_file")
    if not source_file:
        return False # Cannot determine without source file

    # Try to construct a plausible test file path
    # Example: skills/web/search.py -> tests/skills/web/test_search.py
    relative_path = os.path.relpath(source_file, start="a3x/skills") # e.g., web/search.py
    test_filename = f"test_{os.path.splitext(os.path.basename(relative_path))[0]}.py" # e.g., test_search.py
    potential_test_path = os.path.join(test_dir_base, os.path.dirname(relative_path), test_filename)

    logger.debug(f"Checking for test file at: {potential_test_path}")
    return os.path.exists(potential_test_path)

def _check_llm_usage(skill_info: Dict[str, Any]) -> bool:
    """Rudimentary check if a skill function *might* use ctx.llm_call or call_llm."""
    skill_func = skill_info.get("function")
    if not skill_func or not inspect.isfunction(skill_func):
        return False
    try:
        source = inspect.getsource(skill_func)
        # Simple string check - might yield false positives/negatives
        return "ctx.llm_call" in source or "call_llm" in source
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
    description="Metacognitive Skill: Analisa logs do sistema A³X para identificar falhas de autonomia e propor novas skills.",
    parameters={"num_logs_to_analyze": (int, 20)} # Parameter to control analysis scope
)
async def auto_expand_capabilities(ctx: Any, num_logs_to_analyze: int = 20) -> Dict[str, Any]:
    """
    Analyzes A³X operational logs using an LLM to identify autonomy gaps
    and proposes new skills via the 'propose_skill_from_gap' skill.
    Acts as the metacognitive brain of the agent.
    """
    ctx.logger.info(f"Starting metacognitive analysis of autonomy gaps using last {num_logs_to_analyze} logs...")
    opportunities = []
    actions_taken = []
    logs_analyzed_count = 0
    suggestions_found = 0

    # --- Step 1: Load Relevant Logs ---
    ctx.logger.info("Loading recent reflection logs...")
    reflection_logs: List[Dict[str, Any]] = []
    try:
        # Using load_recent_reflection_logs as the primary source for now
        reflection_logs = load_recent_reflection_logs(n=num_logs_to_analyze)
        logs_analyzed_count = len(reflection_logs)
        ctx.logger.info(f"Loaded {logs_analyzed_count} reflection logs.")
    except FileNotFoundError:
         ctx.logger.warning("Reflection log file not found. Skipping reflection log analysis.")
    except Exception as e:
        ctx.logger.exception("Error loading reflection logs:")
        # Continue analysis with other log types if implemented later

    # TODO: Implement loading for other relevant log types (e.g., execution errors, failed plans)
    #       and combine them into 'all_logs'.
    # error_logs = load_error_logs(n=num_logs_to_analyze)
    # all_logs = reflection_logs + error_logs
    all_logs = reflection_logs # Use only reflection logs for this version

    if not all_logs:
        ctx.logger.warning("No relevant logs found to analyze.")
        return {
            "status": "warning",
            "summary": "No logs found for analysis.",
            "opportunities_found": [],
            "actions_taken": []
        }

    # --- Step 2: Analyze Logs with LLM ---
    ctx.logger.info(f"Analyzing {len(all_logs)} log entries with LLM...")
    for i, log_entry in enumerate(all_logs):
        log_identifier = f"Log {i+1}/{len(all_logs)}" # For clearer logging
        ctx.logger.debug(f"Analyzing {log_identifier}...")
        try:
            # Prepare the log entry string safely for the prompt
            try:
                 log_entry_str = json.dumps(log_entry, indent=2, ensure_ascii=False)
            except TypeError as te:
                 ctx.logger.warning(f"Could not serialize log entry {log_identifier} to JSON, using repr: {te}")
                 log_entry_str = repr(log_entry) # Fallback to repr if not serializable

            # Format the prompt using the template and the prepared log string
            prompt = LLM_ANALYSIS_PROMPT_TEMPLATE.format(log_entry_json=log_entry_str)

            # Call LLM using the context
            # Added explicit timeout handling example (adjust value as needed)
            llm_response_str = await asyncio.wait_for(ctx.llm_call(prompt), timeout=120.0) # 2 min timeout

            # --- Step 2a: Parse LLM response ---
            try:
                # Attempt to clean potential markdown code fences or leading/trailing whitespace
                cleaned_response = llm_response_str.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()

                # Handle empty response explicitly after cleaning
                if not cleaned_response or cleaned_response == "{}":
                     ctx.logger.debug(f"LLM analysis of {log_identifier} resulted in no new skill suggestion (empty JSON).")
                     continue # Skip to next log entry

                suggestion_data = json.loads(cleaned_response)

                # Check if it's a valid, non-empty suggestion dictionary
                if suggestion_data and isinstance(suggestion_data, dict) and "skill_name" in suggestion_data:
                    ctx.logger.info(f"LLM suggested a new skill based on {log_identifier}: {suggestion_data.get('skill_name')}")
                    suggestions_found += 1

                    # Basic validation of required keys
                    required_keys = ["skill_name", "reason", "suggestion", "auto_create"]
                    if not all(k in suggestion_data for k in required_keys):
                         ctx.logger.warning(f"LLM suggestion for {log_identifier} is missing required keys ({required_keys}). Skipping. Suggestion: {suggestion_data}")
                         continue

                    # Add source log for traceability
                    suggestion_data["source_log_preview"] = dict(list(log_entry.items())[:3]) # Add first few items as preview

                    opportunities.append(suggestion_data)

                    # --- Step 3: Trigger Skill Proposal if auto_create is True ---
                    if suggestion_data.get("auto_create") is True:
                        ctx.logger.info(f"Attempting to trigger skill proposal for: {suggestion_data['skill_name']}")
                        try:
                            # Prepare arguments for the propose_skill_from_gap skill
                            # Ensure these argument names match the parameters of 'propose_skill_from_gap'
                            proposal_args = {
                                "skill_name": suggestion_data['skill_name'],
                                "reason": suggestion_data['reason'],
                                "suggestion_description": suggestion_data['suggestion'],
                                # Pass parameters and example usage as JSON strings or dicts based on target skill
                                "parameters_json": json.dumps(suggestion_data.get('parameters', {})),
                                "example_usage": suggestion_data.get('example_usage', ''),
                                # Provide context about the analysis
                                "source_analysis_log_preview": json.dumps(suggestion_data["source_log_preview"]), # Ensure preview is JSON string
                                "source_skill": "auto_expand_capabilities"
                            }

                            # Call the 'propose_skill_from_gap' skill
                            # Make sure 'call_skill_by_name' is correctly imported and handles context passing
                            proposal_result = await call_skill_by_name(
                                ctx, # Pass the current context
                                skill_name="propose_skill_from_gap", # Target skill name
                                skill_args=proposal_args
                            )

                            action_msg = f"Called 'propose_skill_from_gap' for '{suggestion_data['skill_name']}'. Result: {proposal_result}"
                            ctx.logger.info(action_msg)
                            actions_taken.append({
                                "action": "call_propose_skill_from_gap",
                                "target_skill": suggestion_data['skill_name'],
                                "result": proposal_result
                            })

                        except ModuleNotFoundError:
                             error_msg = f"Failed to call 'propose_skill_from_gap': Skill 'call_skill_by_name' or 'propose_skill_from_gap' not found or import failed."
                             ctx.logger.error(error_msg)
                             actions_taken.append({"action": "call_propose_skill_from_gap", "target_skill": suggestion_data['skill_name'], "status": "error", "message": error_msg})
                        except Exception as call_e:
                            error_msg = f"Error calling 'propose_skill_from_gap' for '{suggestion_data['skill_name']}': {call_e}"
                            ctx.logger.exception(error_msg) # Log full traceback
                            actions_taken.append({"action": "call_propose_skill_from_gap", "target_skill": suggestion_data['skill_name'], "status": "error", "message": error_msg})
                    else:
                         ctx.logger.info(f"Skill suggestion '{suggestion_data['skill_name']}' found but auto_create is false. Adding to opportunities.")

                # Handles cases where JSON is valid but not a suggestion (e.g., empty dict parsed)
                elif not suggestion_data:
                     ctx.logger.debug(f"LLM analysis of {log_identifier} resulted in no new skill suggestion (parsed as empty).")

            except json.JSONDecodeError as json_e:
                ctx.logger.error(f"Failed to parse LLM JSON response for {log_identifier}: {json_e}. Response: '{llm_response_str}'")
            except Exception as parse_e:
                 ctx.logger.error(f"Error processing LLM response for {log_identifier}: {parse_e}")

        except Exception as llm_e:
            ctx.logger.exception(f"Error during LLM call for {log_identifier}:")
            # Optionally add this error to actions_taken or a separate error list
            actions_taken.append({"action": "llm_analysis", "log_id": log_identifier, "status": "error", "message": str(llm_e)})
            # Continue to the next log entry

    # --- Step 4: Return Summary Report ---
    final_summary = f"Metacognitive analysis complete. Analyzed ~{logs_analyzed_count} logs, found {suggestions_found} potential new skill opportunities."
    ctx.logger.info(final_summary)

    report = {
        "status": "success",
        "summary": final_summary,
        "opportunities_found": opportunities, # List of skill suggestions (dictionaries)
        "actions_taken": actions_taken       # List of actions performed (e.g., calls to propose_skill_from_gap)
    }
    ctx.logger.debug(f"Returning final report: {json.dumps(report, indent=2)}")
    return report

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