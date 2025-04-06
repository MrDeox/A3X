# skills/planning.py
import logging
import json
from typing import Dict, Any, Optional, List

# from core.tools import skill
from a3x.core.tools import skill
from core.prompt_builder import (
    build_planning_prompt,
)  # Assuming this exists or will be created
from core.llm_interface import call_llm

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
) -> Optional[List[Dict[str, Any]]]:
    """
    Generates a multi-step plan to achieve the given objective using the available tools and context.
    Uses the LLM to generate the plan.
    """
    planner_logger.info(f"Generating plan for objective: '{objective[:100]}...'")
    if context:
        planner_logger.debug(f"Planning context keys: {list(context.keys())}")

    # TODO: Load a potentially more specific system prompt for the planner if needed
    planner_system_prompt = DEFAULT_PLANNER_SYSTEM_PROMPT

    # Build the prompt for the planning phase
    prompt_messages = build_planning_prompt(
        objective, available_tools, planner_system_prompt
    )

    plan_json_str = ""
    try:
        # <<< REVERTED: Use async for to consume the single item yielded by call_llm(stream=False) >>>
        plan_json_str = ""  # Initialize
        async for chunk in call_llm(prompt_messages, stream=False):
            plan_json_str += chunk  # Accumulate the single yielded string

        # Basic validation after accumulating
        if not isinstance(plan_json_str, str):
            planner_logger.error(
                f"[PlanningSkill] LLM call yielded unexpected type: {type(plan_json_str)}"
            )
            return None  # Or raise an error
        if not plan_json_str:
            planner_logger.warning(
                "[PlanningSkill] LLM call yielded an empty response string."
            )
            # Handle empty string (e.g., return empty plan or None)
            return []

        planner_logger.debug(f"[PlanningSkill] Raw LLM plan response: {plan_json_str}")

        # Clean and parse
        # Attempt to parse the entire JSON string after basic cleanup (e.g., stripping whitespace)
        try:
            plan_json_str_cleaned = plan_json_str.strip()
            # Remove potential markdown code block fences
            if plan_json_str_cleaned.startswith(
                "```json\n"
            ) and plan_json_str_cleaned.endswith("\n```"):
                plan_json_str_cleaned = plan_json_str_cleaned[
                    len("```json\n") : -len("\n```")
                ]
            elif plan_json_str_cleaned.startswith(
                "```"
            ) and plan_json_str_cleaned.endswith("```"):
                plan_json_str_cleaned = plan_json_str_cleaned[len("```") : -len("```")]
            plan_json_str_cleaned = (
                plan_json_str_cleaned.strip()
            )  # Strip again after removing fences

            plan_list = json.loads(plan_json_str_cleaned)
        except json.JSONDecodeError as e:
            planner_logger.error(
                f"Failed to decode JSON plan from LLM response: {e}. Raw: '{plan_json_str[:200]}...'"
            )
            # Try finding the first list if direct parsing fails (less reliable)
            start_index = plan_json_str.find("[")
            end_index = plan_json_str.rfind("]")
            if start_index != -1 and end_index != -1 and end_index > start_index:
                plan_json_str_extracted = plan_json_str[start_index : end_index + 1]
                planner_logger.warning(
                    f"Direct JSON parsing failed. Attempting to parse extracted list: {plan_json_str_extracted[:100]}..."
                )
                try:
                    plan_list = json.loads(plan_json_str_extracted)
                except json.JSONDecodeError as inner_e:
                    planner_logger.error(
                        f"Failed to decode extracted JSON list: {inner_e}. Original raw: '{plan_json_str[:200]}...'"
                    )
                    return None
            else:
                # If extraction also fails or no brackets found
                return None

        # --- Type Validation (Applied after successful parsing) ---
        if not isinstance(plan_list, list):
            planner_logger.error(
                f"Parsed JSON response is not a list, but type '{type(plan_list).__name__}'. Raw: '{plan_json_str[:200]}...'"
            )
            # Return error specifically for non-list type
            return None

        # Check for non-string elements *only if* it's a list
        if not all(isinstance(step, str) for step in plan_list):
            planner_logger.error(
                f"Parsed JSON list contains non-string elements. Raw: '{plan_json_str[:200]}...'"
            )
            return None
        # --- End Refactored Validation ---

        if not plan_list:  # Check if the list is empty
            planner_logger.warning("LLM generated an empty plan list.")
            return []

        planner_logger.info(f"Plan generated successfully with {len(plan_list)} steps.")
        return plan_list

    except TypeError as e:  # Catches the non-string element error
        planner_logger.error(
            f"Invalid plan structure (non-string elements): {e}. Raw: '{plan_json_str[:200]}...'",
            exc_info=True,
        )
        return None
    except Exception:
        planner_logger.exception("Unexpected error during plan generation:")
        return None


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
