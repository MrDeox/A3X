# skills/reflection.py
import json
import logging
import re
from typing import Dict, Any, List, Optional, Union
from a3x.core.skills import skill
from a3x.core.llm_interface import LLMInterface
from a3x.core.agent import _ToolExecutionContext
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
    description="Analyzes a planned step and its simulated outcome to decide whether to execute, skip, or modify the plan.",
    parameters={
        "step": (str, ...),  # The planned step objective
        "simulated_outcome": (str, ...),  # The description of the simulated result
        # Context is implicitly passed by the agent
    },
)
async def reflect_plan_step(
    step: str, 
    simulated_outcome: str, 
    ctx: _ToolExecutionContext # <-- Accept context object
) -> Dict[str, Any]:
    """
    Reflects on a plan step and its simulated outcome...
    Uses the LLMInterface from the execution context.
    """
    logger.debug(
        f"Reflecting on step: '{step}' with simulated outcome: '{simulated_outcome}'"
    )
    
    llm_interface = ctx.llm_interface
    if not llm_interface:
        logger.error("LLMInterface not found in execution context for reflection.")
        return {"status": "error", "decision": "unknown", "justification": "Internal error: LLMInterface missing.", "confidence": "N/A", "error_message": "LLMInterface missing."}

    # Build prompt using context from ctx if needed (e.g., memory)
    # For now, just pass the direct inputs
    prompt_content = REFLECT_STEP_PROMPT_TEMPLATE.format(
        step=step,
        simulated_outcome=simulated_outcome,
        context=ctx.memory.get_memory() if ctx.memory else "No memory context available.", # Example: Get context from memory
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

        logger.info(f"Reflection complete for step '{step}'. Decision: {decision}.")
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
        logger.exception(f"Error during step reflection LLM call for step '{step}':")
        return {
            "status": "error",
            "decision": "unknown",
            "justification": f"Failed to reflect on step due to LLM error: {e}",
            "confidence": "N/A",
            "error_message": f"Failed to reflect on step: {e}",
        }


@skill(
    name="reflect_on_execution",
    description="Analyzes the overall execution of a plan based on the objective, the plan itself, and the results of each step.",
    parameters={
        "objective": (str, ...),
        "plan": (list, ...),
        "execution_results": (list, ...),
        # Context implicitly passed
    },
)
async def reflect_on_execution(
    objective: str,
    plan: List[str],
    execution_results: List[Dict[str, Any]],
    ctx: _ToolExecutionContext # <-- Accept context object
) -> Dict[str, Any]:
    """Analyzes the overall execution. (Placeholder implementation)"""
    logger.info(f"Reflecting on overall execution for objective: {objective}")
    # TODO: Implement full reflection logic using LLM if needed
    # Access llm_interface via ctx.llm_interface
    # Access memory via ctx.memory
    
    # Example placeholder logic:
    success_count = sum(1 for r in execution_results if r.get('status') == 'success')
    failure_count = len(execution_results) - success_count
    summary = f"Execution Summary: Objective='{objective}', Steps={len(plan)}, Success={success_count}, Failures={failure_count}."
    
    logger.debug(f"Execution Results: {execution_results}")
    
    # Potentially call LLM to generate insights or heuristics based on summary/results
    
    return {
        "status": "success",
        "summary": summary,
        "learned_heuristics": [] # Placeholder for learned insights
    }


# Ensure the skill is registered (assuming automatic loading from skills directory)
