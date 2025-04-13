# skills/reflection.py
import json
import logging
import re
from typing import Dict, Any, List, Optional, Union
from a3x.core.skills import skill
from a3x.core.llm_interface import LLMInterface
from a3x.core.agent import _ToolExecutionContext
from a3x.core.context import Context
# from a3x.core.prompt_builder import build_reflection_prompt # Function missing
# from a3x.core.config import LLM_DEFAULT_MODEL # Removed, var not defined

logger = logging.getLogger(__name__)

# Prompt template for the reflection skill
REFLECT_STEP_PROMPT_TEMPLATE = """
Objective Context:
{context}

Plan Step: {step}

Simulated Outcome of Step: {simulated_outcome}

Based on the plan step, the simulated outcome, and the overall context, please evaluate this step.
Consider:
- Usefulness: Does this step contribute effectively towards the main objective?
- Risk: Are there potential negative consequences or errors likely to occur?
- Efficiency: Is this the most direct way to achieve the sub-goal?
- Feasibility: Is the step actually possible with the available tools/context?

Decision:
Based on your evaluation, decide whether to 'execute', 'modify', or 'skip' this step.
- execute: The step is good as is.
- modify: The step needs adjustment before execution (provide brief suggestion if possible).
- skip: The step is unnecessary, harmful, or redundant.

Justification:
Briefly explain the reasoning behind your decision.

Output Format:
Decision: [execute|modify|skip]
Justification: [Your reasoning here]
"""

DEFAULT_REFLECTION_SYSTEM_PROMPT = """
You are an expert AI assistant specializing in analyzing plan execution steps.
Your task is to reflect on a proposed plan step and its simulated outcome.
Based on the simulation, decide if the step should be executed as planned, modified (requiring replanning), or skipped altogether.
Provide a clear justification for your decision.

Respond ONLY with the following format:
Decision: <execute|modify|skip>
Justification: <Your detailed reasoning>

Example 1:
Decision: execute
Justification: The simulation shows the step achieves the intended sub-goal without issues.

Example 2:
Decision: modify
Justification: The simulation revealed that the file path is incorrect. The plan needs to be modified to first list files to find the correct path before writing.

Example 3:
Decision: skip
Justification: The simulation indicates this step is redundant as the required information was already obtained in a previous step.
"""


def _parse_reflection_output(response_text: str) -> Dict[str, Any]:
    """Parses the LLM response to extract decision and justification."""
    decision = "unknown"
    justification = "No justification provided."

    # Allow spaces around the colon for Decision
    decision_match = re.search(
        r"Decision\s*:\s*(execute|modify|skip)", response_text, re.IGNORECASE
    )
    if decision_match:
        decision = decision_match.group(1).lower()

    # Allow spaces around the colon for Justification
    justification_match = re.search(
        r"Justification\s*:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL
    )
    if justification_match:
        justification = justification_match.group(1).strip()

    # Fallbacks
    if decision == "unknown":
        if "execute" in response_text.lower():
            decision = "execute"
        elif "skip" in response_text.lower():
            decision = "skip"
        elif "modify" in response_text.lower():
            decision = "modify"

    return {"decision": decision, "justification": justification}


@skill(
    name="reflect_plan_step",
    description="Reflects on the outcome of a plan step, extracts insights, and updates memory.",
    parameters={
        "context": {"type": Context, "description": "Execution context providing access to logger, LLM, and memory."},
        "step_index": {"type": int, "description": "The index of the plan step being reflected upon."},
        "plan": {"type": List[str], "description": "The overall plan as a list of steps."},
        "action_taken": {"type": Optional[str], "description": "The specific action performed for this step."},
        "observation": {"type": str, "description": "The result or observation obtained after executing the step."},
        "success": {"type": bool, "description": "Whether the step was considered successful."}
    }
)
async def reflect_plan_step(
    context: Context,
    step_index: int,
    plan: List[str],
    action_taken: Optional[str],
    observation: str,
    success: bool
) -> Dict[str, Any]:
    """
    Reflects on a plan step and its simulated outcome...
    Uses the LLMInterface from the execution context.
    """
    logger.debug(
        f"Reflecting on step: '{step_index}' with simulated outcome: '{observation}'"
    )
    
    llm_interface = context.llm_interface
    if not llm_interface:
        logger.error("LLMInterface not found in execution context for reflection.")
        return {"status": "error", "decision": "unknown", "justification": "Internal error: LLMInterface missing.", "confidence": "N/A", "error_message": "LLMInterface missing."}

    # Build prompt using context from ctx if needed (e.g., memory)
    # For now, just pass the direct inputs
    prompt_content = REFLECT_STEP_PROMPT_TEMPLATE.format(
        step=step_index,
        simulated_outcome=observation,
        context=context.memory.get_memory() if context.memory else "No memory context available.", # Example: Get context from memory
    )
    
    # Prepare messages for the LLM call
    messages = [
        {"role": "system", "content": DEFAULT_REFLECTION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_content}
    ]

    try:
        llm_response_text = ""
        # Updated call site
        async for chunk in llm_interface.call_llm( # <-- Use instance method
            messages=messages, 
            stream=False
        ):
            llm_response_text += chunk

        if not llm_response_text:
            logger.error("Reflection LLM call returned empty response.")
            return {"status": "error", "decision": "unknown", "justification": "LLM response was empty.", "confidence": "N/A", "error_message": "LLM response empty."}

        parsed_output = _parse_reflection_output(llm_response_text)
        decision = parsed_output["decision"]
        justification = parsed_output["justification"]

        logger.info(f"Reflection complete for step '{step_index}'. Decision: {decision}.")
        logger.debug(f"Justification: {justification}")

        confidence = "Média" # Placeholder confidence

        return {
            "status": "success",
            "decision": decision,
            "justification": justification,
            "confidence": confidence,
            "error_message": None,
        }

    except Exception as e:
        logger.exception(f"Error during step reflection LLM call for step '{step_index}':")
        return {
            "status": "error",
            "decision": "unknown",
            "justification": f"Failed to reflect on step due to LLM error: {e}",
            "confidence": "N/A",
            "error_message": f"Failed to reflect on step: {e}",
        }


@skill(
    name="reflect_on_execution",
    description="Reflects on the overall execution outcome, identifies issues, and suggests improvements.",
    parameters={
        "context": {"type": Context, "description": "Execution context providing access to LLM, memory, and full history."},
        "original_objective": {"type": str, "description": "The initial objective given to the agent."},
        "action_taken": {"type": List[Dict[str, Any]], "description": "Full history of actions taken and observations received."},
        "observation": {"type": str, "description": "The final outcome or observation of the entire task execution."},
        "success": {"type": bool, "description": "Whether the overall task was considered successful."}
    }
)
async def reflect_on_execution(
    context: Context,
    original_objective: str,
    action_taken: List[Dict[str, Any]],
    observation: str,
    success: bool
) -> Dict[str, Any]:
    """Analyzes the overall execution. (Placeholder implementation)"""
    logger.info(f"Reflecting on overall execution for objective: {original_objective}")
    # TODO: Implement full reflection logic using LLM if needed
    # Access llm_interface via ctx.llm_interface
    # Access memory via ctx.memory
    
    # Example placeholder logic:
    success_count = sum(1 for r in action_taken if r.get('status') == 'success')
    failure_count = len(action_taken) - success_count
    summary = f"Execution Summary: Objective='{original_objective}', Steps={len(action_taken)}, Success={success_count}, Failures={failure_count}."
    
    logger.debug(f"Execution Results: {action_taken}")
    
    # Potentially call LLM to generate insights or heuristics based on summary/results
    
    return {
        "status": "success",
        "summary": summary,
        "learned_heuristics": [] # Placeholder for learned insights
    }


# Ensure the skill is registered (assuming automatic loading from skills directory)
