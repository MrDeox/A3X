# skills/planning.py
import logging
import json
import asyncio
from typing import Dict, Any, Optional, List

# from core.tools import skill
from a3x.core.skills import skill
from a3x.core.prompt_builder import (
    build_planning_prompt,
)  # Assuming this exists or will be created
from a3x.core.llm_interface import call_llm

# from core.skills_utils import create_skill_response

# from core.config import LLAMA_SERVER_URL, LLAMA_DEFAULT_HEADERS

# Need requests for HTTP call

# Define the system prompt for the hierarchical planner
DEFAULT_PLANNER_SYSTEM_PROMPT = """
You are a highly intelligent planning agent. Your goal is to break down a complex objective into a sequence of smaller, actionable steps that can be executed using a predefined set of tools.

Analyze the user's objective, the available tools, and the provided context.
Generate a plan as a JSON list of strings. Each string in the list represents a single step.
Each step should clearly describe an action that can likely be performed by one or more of the available tools, or represent a logical sub-goal.

**IMPORTANT OUTPUT FORMAT:**
*   Your response MUST be ONLY a valid JSON list of strings. Example: `["Step 1: Description...", "Step 2: Description..."]`
*   DO NOT include any introductory text, explanations, apologies, or other text outside the JSON list.
*   DO NOT wrap the JSON list in markdown code blocks (like ```json ... ```).
*   If the objective is impossible or cannot be broken down with the given tools, return an empty JSON list: `[]`.

**EXAMPLE RESPONSE (VALID):**
`["Use the 'read_file' tool to read the content of 'input.txt'.", "Analyze the content to find relevant keywords.", "Use the 'web_search' tool to search for information based on the keywords.", "Summarize the findings and provide a final answer using 'final_answer'."]`

**EXAMPLE RESPONSE FOR IMPOSSIBLE TASK (VALID):**
`[]`

**EXAMPLE RESPONSE (INVALID - DO NOT DO THIS):**
```json
["Step 1..."]
```

**EXAMPLE RESPONSE (INVALID - DO NOT DO THIS):**
Okay, here is the plan:
["Step 1..."]

**Focus entirely on providing ONLY the JSON list of plan steps.**
"""

# Logger specifically for the planning skill
planner_logger = logging.getLogger(__name__)


@skill(
    name="hierarchical_planner",
    description="Breaks down a complex objective into a sequence of smaller, actionable steps using available tools.",
    parameters={
        "objective": (str, ...),  # The main user objective
        "available_tools": (str, ...),  # Descriptions of tools the agent can use
        "context": (dict, {}),  # Optional context from memory or perception
    },
)
async def hierarchical_planner(
    objective: str, available_tools: str, context: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generates a multi-step plan (list of strings) to achieve the given objective.
    Considers learned heuristics (success and failure) from the context.
    Returns a dictionary: {'status': 'success', 'data': {'plan': [...]}} on success,
    or {'status': 'error', 'data': {'message': '...'}} on failure.
    """
    planner_logger.info(f"Generating plan for objective: '{objective[:100]}...'")
    context = context or {} # Ensure context is a dict
    if context:
        planner_logger.debug(f"Planning context keys: {list(context.keys())}")

    # --- MODIFIED: Extract Heuristics AND Generalized Rules --- 
    learned_heuristics_prompt_section = ""
    success_heuristics_text = context.get("learned_success_heuristics")
    failure_heuristics_text = context.get("learned_failure_heuristics")
    generalized_rules_text = context.get("learned_generalized_rules")

    # Format Generalized Rules Section
    if generalized_rules_text:
        learned_heuristics_prompt_section += f"\n\n## Conselhos Gerais (Aprendizado Consolidado)\n{generalized_rules_text}"
        planner_logger.info("Adding generalized rules (advice) section to planner prompt.")

    # Format existing heuristics sections
    if success_heuristics_text:
        learned_heuristics_prompt_section += f"\n\n## Recomendações (Sucessos Passados Recentes)\n{success_heuristics_text}"
        planner_logger.info("Adding learned success heuristics section to planner prompt.")
        
    if failure_heuristics_text:
        learned_heuristics_prompt_section += f"\n\n## Avisos (Falhas Passadas Recentes)\n{failure_heuristics_text}"
        planner_logger.info("Adding learned failure heuristics section to planner prompt.")
    # --- END MODIFIED SECTION ---

    # TODO: Load a potentially more specific system prompt for the planner if needed
    planner_system_prompt = DEFAULT_PLANNER_SYSTEM_PROMPT

    # Build the prompt for the planning phase - inject rules & heuristics
    prompt_messages = build_planning_prompt(
        objective,
        available_tools,
        planner_system_prompt + learned_heuristics_prompt_section, # Append combined section
        # context=context # Pass other context elements if build_planning_prompt accepts them
    )

    plan_json_str = ""
    plan_list = []
    try:
        plan_json_str = ""
        llm_url = getattr(context.get('ctx'), 'llm_url', None) # Try to get from ctx if passed
        async for chunk in call_llm(prompt_messages, llm_url=llm_url, stream=False):
            plan_json_str += chunk
        
        planner_logger.debug(f"Planner LLM raw response: {plan_json_str[:500]}...") # Log start of response
        parsed_plan = parse_llm_json_output(plan_json_str, expected_keys=["plan"], logger=planner_logger)

        if parsed_plan and "plan" in parsed_plan:
            plan_list = parsed_plan["plan"]
            if isinstance(plan_list, list):
                 # Basic validation: ensure it's a list of strings
                if all(isinstance(step, str) for step in plan_list):
                    planner_logger.info(f"Plan generated successfully with {len(plan_list)} steps.")
                    return {"status": "success", "data": {"plan": plan_list}}
                else:
                    planner_logger.error("Planner LLM returned 'plan' but it contains non-string elements.")
                    return {"status": "error", "data": {"message": "Plan generation failed: plan list contains invalid elements."}}
            else:
                planner_logger.error("Planner LLM returned 'plan' but it's not a list.")
                return {"status": "error", "data": {"message": "Plan generation failed: invalid plan format."}}
        else:
            planner_logger.error(f"Planner LLM response did not contain a valid 'plan' key after parsing. Raw: {plan_json_str[:200]}")
            return {"status": "error", "data": {"message": "Plan generation failed: Could not parse plan from LLM response."}}

    except Exception as e:
        planner_logger.exception(f"Error during planning LLM call or parsing: {e}")
        return {"status": "error", "data": {"message": f"Exception during planning: {e}"}}


# Example usage (for potential direct testing)
# async def main():
#     from core.tools import get_tool_descriptions, load_skills
#     load_skills()
#     tools_desc = get_tool_descriptions()
#     test_objective = "Read the file 'requirements.txt' and then write its content to 'requirements_copy.txt'"
#     result = await hierarchical_planner(objective=test_objective, available_tools=tools_desc)
#     print(json.dumps(result, indent=2))
#
# if __name__ == "__main__":
#     import asyncio
#     # Need to configure logging if run directly
#     logging.basicConfig(level=logging.DEBUG)
