# skills/planning.py
import logging
import json
import asyncio
import re
from typing import Dict, Any, Optional, List, Union
from pathlib import Path # <<< ADDED import >>>

# from core.tools import skill
from a3x.core.skills import skill
from a3x.core.prompt_builder import (
    build_planning_prompt,
)  # Assuming this exists or will be created
from a3x.core.llm_interface import LLMInterface # <-- IMPORT CLASS
from a3x.core.context import Context, _ToolExecutionContext
from a3x.fragments.base import FragmentContext 
from a3x.core.tool_registry import ToolRegistry # For type hint
from a3x.fragments.registry import FragmentRegistry # For type hint
from a3x.core.models import PlanStep # <<< KEEP PlanStep import >>>
from a3x.core.context import SharedTaskContext

# from core.skills_utils import create_skill_response

# from core.config import LLAMA_SERVER_URL, LLAMA_DEFAULT_HEADERS

# Need requests for HTTP call

# Define the system prompt for the hierarchical planner
STRUCTURED_PLANNER_SYSTEM_PROMPT = """
You are an expert planning agent (A3X Planner). Your goal is to create a robust, step-by-step plan to achieve the user's objective.

Analyze the user's objective, the available tools (skills), available components (fragments), and any relevant context from memory.
Generate a plan as a JSON list of objects. Each object represents a single step with the following mandatory keys:

1.  `step_id`: INTEGER - A unique sequential identifier for the step (e.g., 1, 2, 3, ...).
2.  `action_type`: STRING - Must be either "skill" or "fragment".
3.  `target_name`: STRING - The exact name of the skill (from the skills list) or fragment (from the fragments list) to execute.
4.  `arguments`: OBJECT - A JSON object containing the arguments needed for the `target_name`. Keys should be parameter names, values should be the extracted/inferred values.
    *   For skills, extract arguments based on the skill's parameters provided.
    *   For fragments, the primary argument is usually `sub_task` (a string describing what the fragment should achieve for this step). You may include other relevant arguments if the fragment description indicates they are useful.
    *   If a skill/fragment requires no arguments, provide an empty object: `{}`.
5.  `description`: STRING - A brief, human-readable description of what this step does (for logging/UI).

**CRITICAL: OUTPUT FORMAT RULES**
1.  Your response MUST be a single, valid JSON list `[...]` containing step objects.
2.  Each object in the list MUST have the keys `step_id`, `action_type`, `target_name`, `arguments`, and `description`.
3.  `step_id` MUST be a sequential integer starting from 1.
4.  Extract argument values directly from the objective or infer them logically. DO NOT just put placeholder strings like "..." as argument values.
5.  If the objective is impossible or cannot be broken down, return ONLY an empty JSON list: `[]`.
6.  Do NOT include ```json markdown fences or any other text outside the main JSON list.

**EXAMPLE RESPONSE (VALID STRUCTURED JSON):**
`[
  {
    "step_id": 1,
    "action_type": "skill", 
    "target_name": "read_file", 
    "arguments": {"path": "input.txt"}, 
    "description": "Read the content of input.txt"
  },
  {
    "step_id": 2,
    "action_type": "fragment", 
    "target_name": "DataAnalysisFragment", 
    "arguments": {"sub_task": "Analyze the text read from the file to find keywords."}, 
    "description": "Analyze file content for keywords using DataAnalysisFragment"
  },
  {
    "step_id": 3,
    "action_type": "skill", 
    "target_name": "web_search", 
    "arguments": {"query": "{step_2_result.keywords}"}, 
    "description": "Search web based on extracted keywords (placeholder for result reference)"
  },
  {
    "step_id": 4,
    "action_type": "skill", 
    "target_name": "final_answer", 
    "arguments": {"answer": "{step_3_result.summary}"}, 
    "description": "Provide final answer based on search summary (placeholder for result reference)"
  }
]`

**Focus entirely on providing ONLY the structured JSON list.**
"""

# Logger specifically for the planning skill
planner_logger = logging.getLogger(__name__)

# <<< Define context type alias for clarity (If needed, keep it simple) >>>
# PlannerSkillContext = Union[Context, FragmentContext] # Removed for now as ctx is SharedTaskContext

# <<< REMOVED Class Structure - Reverted to standalone function >>>

# <<< ADDED @skill decorator back >>>
@skill(
    name="hierarchical_planner",
    description="Generates a structured step-by-step plan for a complex task using available tools/fragments.",
    parameters={
        # <<< Use context object >>>
        "context": {"type": _ToolExecutionContext, "description": "Execution context providing access to LLM, registries, etc.", "optional": False},
        "task_description": {"type": str, "description": "The high-level description of the task to be planned.", "optional": False},
        "available_tools": {"type": List[str], "description": "List of tool/fragment names available for planning.", "optional": False},
        "max_steps": {"type": int, "description": "The maximum number of steps allowed in the plan.", "default": 10, "optional": True}
    }
)
async def hierarchical_planner(
        context: _ToolExecutionContext, # <<< Use context object
        task_description: str,
        available_tools: List[str], # Names of tools/fragments available for this step
        max_steps: int = 10
) -> Dict[str, Any]:
    """
    Generates a structured step-by-step plan for a complex task using available tools/fragments.

    Args:
        context: Execution context providing access to LLM, registries, etc.
        task_description: The high-level description of the task to be planned.
        available_tools: List of tool/fragment names available for planning.
        max_steps: The maximum number of steps allowed in the plan.

    Returns:
        A dictionary containing the structured plan: {"status": "success/failure", "data": {"plan": List[PlanStep]}}
        or {"status": "failure", "error": "Reason"}
    """
    # <<< Use logger and attributes from context >>>
    log_prefix = f"[PlannerSkill:{context.shared_task_context.task_id if context.shared_task_context else 'N/A'}]"
    logger = context.logger
    logger.info(f"{log_prefix} Hierarchical planner invoked for task: '{task_description}'")
    logger.debug(f"{log_prefix} Available tools/fragments: {available_tools}")

    # Access registries and LLM interface from context
    tool_registry = context.tools_dict
    fragment_registry = context.fragment_registry
    llm_interface = context.llm_interface

    if not tool_registry:
        logger.error(f"{log_prefix} ToolRegistry not found in context.")
        return {"status": "failure", "error": "ToolRegistry missing in context"}
    if not fragment_registry:
         logger.error(f"{log_prefix} FragmentRegistry not found in context.")
         return {"status": "failure", "error": "FragmentRegistry missing in context"}
    if not llm_interface:
        logger.error(f"{log_prefix} LLMInterface not found in context.")
        return {"status": "failure", "error": "LLMInterface missing in context"}

    # <<< ADDED: Load GBNF Grammar >>>
    gbnf_grammar = None
    try:
        # Construct path relative to this file (skills/planning.py -> grammars/)
        grammar_file_path = Path(__file__).parent.parent / 'grammars' / 'plan_step_grammar.gbnf'
        if grammar_file_path.is_file():
            with open(grammar_file_path, 'r') as f:
                gbnf_grammar = f.read()
            logger.info(f"{log_prefix} Loaded GBNF grammar for plan generation from {grammar_file_path}")
        else:
            logger.warning(f"{log_prefix} GBNF grammar file not found at {grammar_file_path}. Proceeding without grammar enforcement.")
    except Exception as e:
        logger.error(f"{log_prefix} Error loading GBNF grammar: {e}. Proceeding without grammar enforcement.")
    # <<< END ADDED >>>

    # 1. Get Schemas for Available Tools/Fragments
    # <<< REMOVE TEMPORARY FILTERING >>>
    logger.debug(f"{log_prefix} Full list of available tools/fragments for reference: {available_tools}") # Log original list
    # relevant_tool_names = available_tools # Use the full list provided
    # essential_tools = ["propose_skill_from_gap", "reload_generated_skills", "execute_code", "final_answer"]
    # relevant_tool_names = [tool for tool in available_tools if tool in essential_tools]
    # logger.info(f"{log_prefix} [TEMP TEST] Using filtered tool list for planning: {relevant_tool_names}")
    relevant_tool_names = available_tools # <<< USE FULL LIST >>>
    # <<< END REMOVAL >>>
    
    skills_prompt_section = ""
    fragments_prompt_section = ""
    # <<< Use relevant_tool_names (which is now the full list) >>>
    for tool_name in relevant_tool_names:
        # Try getting fragment definition first
        all_fragment_defs = fragment_registry.get_all_definitions()
        fragment_def = all_fragment_defs.get(tool_name)
        if fragment_def:
            # Format fragment info
            fragments_prompt_section += f"### Fragment: {fragment_def.name}\n"
            fragments_prompt_section += f"* Description: {fragment_def.description}\n"
            # List managed skills if any
            if fragment_def.managed_skills:
                 fragments_prompt_section += f"* Handles Skills: {', '.join(fragment_def.managed_skills)}\n"
            fragments_prompt_section += "\n"
            continue # Move to next tool name
            
        # If not a fragment, try getting skill schema
        schema = tool_registry.get_tool_details(tool_name)
        if schema:
            # Format skill info
            skills_prompt_section += f"### Skill: {schema.get('name', tool_name)}\n"
            skills_prompt_section += f"* Description: {schema.get('description', 'No description provided.')}\n"
            
            # Format parameters
            schema_parameters_object = schema.get('parameters', {})
            skill_defined_params = schema_parameters_object.get('properties', {})
            required_params = set(schema_parameters_object.get('required', []))
            llm_visible_params = {
                k: v for k, v in skill_defined_params.items()
                if k not in ('self', 'cls', 'ctx', 'context', 'resolved_path', 'original_path_str') # Exclude injected/internal params
                and k[0].islower()
            }
            
            if llm_visible_params:
                params_str_parts = []
                for param_name, param_info in llm_visible_params.items():
                    if isinstance(param_info, dict):
                        param_type = param_info.get('type', 'any')
                        param_desc = param_info.get('description', '')
                        is_required = param_name in required_params
                        required_indicator = " (required)" if is_required else ""
                        params_str_parts.append(f"  - {param_name} ({param_type}){required_indicator}: {param_desc}")
                    else:
                        params_str_parts.append(f"  - {param_name}")
                if params_str_parts:
                    skills_prompt_section += f"* Parameters:\n" + "\n".join(params_str_parts) + "\n"
                else:
                    skills_prompt_section += f"* Parameters: None\n"
            else:
                skills_prompt_section += f"* Parameters: None\n"
            skills_prompt_section += "\n"
        else:
         logger.warning(f"{log_prefix} Schema not found for relevant tool: {tool_name}")
    
    # Skip fragments for this simplified test
    # <<< Log the simplified tools being sent >>>
    logger.debug(f"{log_prefix} Simplified tools/fragments sent to LLM for planning:\n-------\n{skills_prompt_section}\n-------")
    # <<< END MODIFICATION >>>

    # 2. Construct the Prompt
    # Based on the PlanStep TypedDict structure
    plan_step_format = """
    {
        "step_id": <int>, // Sequential step number starting from 1
        "description": "<Clear description of what this step achieves>",
        "action_type": "<'skill' or 'fragment'>", // Type of action
        "target_name": "<Name of the skill or fragment to call (e.g., 'write_file', 'FileOpsManager')>", // <<< CORRECTED EXAMPLE >>>
        "arguments": { // Arguments for the skill/fragment
            "<arg_name_1>": "<value_1>",
            "<arg_name_2>": "<value_2>",
            ...
        }
    }
    """

    prompt = f"""
Objective: Create a detailed, step-by-step plan to accomplish the following task, using *only* the provided tools (skills and fragments).

Task: {task_description}

Available Tools (Skills and Fragments):
--- Available Tools Start ---
{skills_prompt_section if skills_prompt_section else "No specific tools/fragments provided. Assume standard capabilities if necessary, but prioritize listed tools."}
--- Available Tools End ---

Important Instructions & Constraints:
- Distinguish carefully between 'skill' and 'fragment'. Use the correct `action_type`.
- Skills are specific actions (e.g., `write_file`, `read_file`).
- Fragments coordinate actions or other fragments (e.g., `FileOpsManager`).
- The plan must be a sequence of steps, numbered starting from 1 using `step_id`.
- Each step must use exactly one available Skill or Fragment listed above.
- For each step, specify the 'action_type' ('skill' or 'fragment'), the 'target_name' (the exact name from the list), and the 'arguments' needed.
- **Crucially, you MUST generate the `arguments` field as a valid JSON object containing the actual values needed for the chosen skill/fragment, extracted or inferred from the Task description and the tool's parameters.** Do not use placeholders.
- Ensure file paths in arguments are constructed correctly, often relative to the workspace root or based on directories created in previous steps. Use forward slashes ('/') for paths.
- Ensure the plan is logical and achieves the overall task.
- The maximum number of steps allowed is {max_steps}.
- The final output MUST be a valid JSON list containing the plan steps, following this exact format for each step:
{plan_step_format}

High-Level Plan Consideration (Optional):
Think step-by-step how to break down the task. What needs to happen first? What depends on previous steps?

Generate the JSON plan as a list of steps. If the task cannot be achieved with the provided tools, return an empty list [].
"""

    # <<< ADDED: Log the prompt (or a summary) >>>
    prompt_summary = prompt[:500] + "...(truncated)" if len(prompt) > 500 else prompt # Log summary if too long
    logger.debug(f"{log_prefix} Prompt sent to LLM for planning:\n-------\n{prompt_summary}\n-------")
    # <<< END ADDED >>>

    messages = [
        {"role": "system", "content": STRUCTURED_PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    # 3. Call LLM
    logger.info(f"{log_prefix} Calling LLM for planning...")
    llm_response_raw = ""
    try:
        # <<< Use context.llm_interface >>>
        async for chunk in llm_interface.call_llm( 
            messages=messages,
            stream=False, # Expecting a single JSON response
            temperature=0.1, # Lower temperature for more deterministic planning
            grammar=gbnf_grammar # Pass grammar if loaded
        ):
            llm_response_raw += chunk

        # <<< Log raw response at INFO level >>>
        logger.info(f"{log_prefix} Raw LLM response received: {llm_response_raw}")

        if not llm_response_raw:
            logger.error(f"{log_prefix} LLM returned an empty response.")
            # <<< Ensure consistent error format >>>
            return {"status": "failure", "error": "LLM returned empty response"}

        # 4. Parse the Response
        # <<< Improve JSON extraction: Look for list within optional markdown fences >>>
        json_str = None
        # Try finding ```json ... ``` block first
        # Using raw strings r"..." to simplify escaping
        json_block_match = re.search(r"```json\\s*(\\[.*?]\\s*)```", llm_response_raw, re.DOTALL)
        if json_block_match:
            json_str = json_block_match.group(1)
        else:
            # Fallback to finding any list `[...]`
            # Using raw string r"..."
            list_match = re.search(r"(\\[.*?]\s*)", llm_response_raw, re.DOTALL)
            if list_match:
                json_str = list_match.group(1)
        # <<< END improved extraction >>>

        if not json_str:
            logger.error(f"{log_prefix} Could not extract JSON list `[...]` from LLM response. Response: {llm_response_raw}")
            # <<< Include raw response in error message >>>
            return {"status": "failure", "error": f"LLM response did not contain a JSON list. Raw Response: {llm_response_raw[:1000]}..."} # Limit length
        
        logger.debug(f"{log_prefix} Extracted JSON string: {json_str}") # Keep this debug

        try:
            plan = json.loads(json_str)
            if not isinstance(plan, list):
                 raise ValueError("Parsed JSON is not a list.")
            
            # TODO: Add validation for PlanStep structure here?
            # for step in plan:
            #    PlanStep(**step) # Validate against TypedDict/Pydantic model

            logger.info(f"{log_prefix} Successfully generated plan with {len(plan)} steps.")
            return {"status": "success", "data": {"plan": plan}}
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"{log_prefix} Failed to parse LLM response as valid plan JSON list: {e}. Response: {json_str}")
            # <<< Include raw response and extracted string in error message >>>
            return {"status": "failure", "error": f"Failed to parse LLM plan response: {e}. Extracted: {json_str[:500]}... Raw: {llm_response_raw[:500]}..."} # Limit length

    except Exception as e:
        logger.exception(f"{log_prefix} Exception during planning LLM call: {e}")
        # <<< Include raw response in error message if available >>>
        raw_resp_snippet = llm_response_raw[:500] + "..." if 'llm_response_raw' in locals() and llm_response_raw else "(Raw response not available)"
        return {"status": "failure", "error": f"Exception during LLM call: {e}. Raw Response: {raw_resp_snippet}"}

# Example of how to potentially register this skill's methods if not done automatically
# Needs access to the ToolRegistry instance
# def register_planning_skill(registry: ToolRegistry, context_id: str):
#     instance = PlanningSkill(context_id) # Assuming context_id is available
#     # Find methods with _skill_info
#     for attr_name in dir(instance):
#         attr = getattr(instance, attr_name)
#         if callable(attr) and hasattr(attr, '_skill_info'):
#             skill_info = getattr(attr, '_skill_info')
#             registry.register_tool(
#                 name=skill_info['name'],
#                 instance=instance,
#                 tool=attr, # The bound method
#                 schema=skill_info['schema']
#             )
#             planner_logger.info(f"Registered skill: {skill_info['name']}")

# This registration part would typically happen during application setup.

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
