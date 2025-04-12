# skills/reflection.py
import json
import logging
import re
from typing import Dict, Any, List, Optional, Union
from a3x.core.skills import skill
from a3x.core.llm_interface import call_llm
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

    if decision == "unknown" and "execute" in response_text.lower():
        decision = "execute"  # Fallback guess
    if decision == "unknown" and "skip" in response_text.lower():
        decision = "skip"
    if decision == "unknown" and "modify" in response_text.lower():
        decision = "modify"

    return {"decision": decision, "justification": justification}


@skill(
    name="reflect_plan_step",
    description="Analyzes a planned step and its simulated outcome to decide whether to execute, skip, or modify the plan.",
    parameters={
        "step": (str, ...),  # The planned step objective
        "simulated_outcome": (str, ...),  # The description of the simulated result
        "context": (
            dict,
            None,
        ),  # REVERTED for @skill compatibility (keep func signature)
    },
)
async def reflect_plan_step(
    step: str, simulated_outcome: str, context: Optional[Union[Dict, str]] = None
) -> Dict[str, Any]:
    """
    Reflects on a plan step and its simulated outcome to decide the next course of action.

    Args:
        step (str): The plan step under evaluation.
        simulated_outcome (str): The predicted outcome from the simulation skill.
        context (Dict[str, Any], optional): The current context available to the agent. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing the reflection result.
            - status (str): 'success' or 'error'.
            - decision (str): 'execute', 'modify', 'skip', or 'unknown'.
            - justification (str): Explanation for the decision.
            - confidence (str): Confidence level ('Alta', 'Média', 'Baixa') - Placeholder.
            - error_message (str | None): Error details if reflection failed.
    """
    logger.debug(
        f"Reflecting on step: '{step}' with simulated outcome: '{simulated_outcome}'"
    )

    prompt = REFLECT_STEP_PROMPT_TEMPLATE.format(
        step=step,
        simulated_outcome=simulated_outcome,
        context=context if context else "Nenhum contexto fornecido.",
    )

    try:
        # <<< REVERTED: Use async for >>>
        llm_response_text = ""
        # <<< ADDED: Wrap prompt in message list structure >>>
        messages = [{"role": "user", "content": prompt}]
        async for chunk in call_llm(messages, stream=False):  # Pass messages list
            llm_response_text += chunk

        if not llm_response_text or not isinstance(llm_response_text, str):
            logger.error(
                f"Reflection LLM call returned invalid data type: {type(llm_response_text)}"
            )
            # <<< MODIFIED: Return error dict matching test expectations >>>
            return {
                "status": "error",
                "decision": "unknown",
                "justification": "LLM response was empty or not a string.",
                "confidence": "N/A",
                "error_message": "LLM response was empty or not a string.",
            }

        parsed_output = _parse_reflection_output(llm_response_text)
        decision = parsed_output["decision"]
        justification = parsed_output["justification"]

        logger.info(
            f"Reflection complete for step '{step}'. Decision: {decision}. Justification: {justification}"
        )

        confidence = "Média"

        # <<< MODIFIED: Return success dict matching test expectations >>>
        return {
            "status": "success",
            "decision": decision,
            "justification": justification,
            "confidence": confidence,
            "error_message": None,
        }

    except Exception as e:
        logger.exception(f"Error during step reflection LLM call for step '{step}':")
        # <<< MODIFIED: Return error dict matching test expectations >>>
        return {
            "status": "error",
            "decision": "unknown",
            "justification": "Failed to reflect on step due to LLM error.",
            "confidence": "N/A",
            "error_message": f"Failed to reflect on step: {e}",
        }


@skill(
    name="reflect_on_execution",
    description="Analyzes the overall execution of a plan based on the objective, the plan itself, and the results of each step.",
    parameters={
        "objective": (str, ...),  # The original objective
        "plan": (list, ...),  # The final executed plan (list of strings)
        "execution_results": (
            list,
            ...,
        ),  # List of result dictionaries for each executed/skipped step
        "context": (
            dict,
            None,
        ),  # REVERTED for @skill compatibility (keep func signature)
    },
)
def reflect_on_execution(
    objective: str,
    plan: List[str],
    execution_results: List[Dict[str, Any]],
    context: Optional[Union[dict, str]] = None,
) -> Dict[str, Any]:
    return {}  # ADDED placeholder return
    # pass # REMOVED pass
    # ... existing code ...


# Ensure the skill is registered (assuming automatic loading from skills directory)
