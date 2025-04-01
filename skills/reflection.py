# skills/reflection.py
import logging
import re
from typing import Dict, Any
from core.tools import skill
from core.llm_interface import call_llm # Corrected import

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

def _parse_reflection_output(response_text: str) -> Dict[str, Any]:
    """Parses the LLM response to extract decision and justification."""
    decision = "unknown"
    justification = "No justification provided."

    # Allow spaces around the colon for Decision
    decision_match = re.search(r"Decision\s*:\s*(execute|modify|skip)", response_text, re.IGNORECASE)
    if decision_match:
        decision = decision_match.group(1).lower()

    # Allow spaces around the colon for Justification
    justification_match = re.search(r"Justification\s*:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
    if justification_match:
        justification = justification_match.group(1).strip()

    if decision == "unknown" and "execute" in response_text.lower(): decision = "execute" # Fallback guess
    if decision == "unknown" and "skip" in response_text.lower(): decision = "skip"
    if decision == "unknown" and "modify" in response_text.lower(): decision = "modify"

    return {"decision": decision, "justification": justification}

@skill(
    name="reflect_plan_step",
    description="Avalia o resultado simulado de um passo do plano e decide se ele deve ser executado, modificado ou descartado.",
    parameters={
        "step": (str, ...),                # Passo do plano a ser avaliado (obrigatório)
        "simulated_outcome": (str, ...), # Resultado simulado pela skill 'simulate_step' (obrigatório)
        "context": (Dict[str, Any], {}),   # Contexto atual do agente (opcional)
    }
)
async def reflect_plan_step(step: str, simulated_outcome: str, context: Dict[str, Any] = {}) -> Dict[str, Any]:
    """
    Reflects on a plan step and its simulated outcome to decide the next course of action.

    Args:
        step (str): The plan step under evaluation.
        simulated_outcome (str): The predicted outcome from the simulation skill.
        context (Dict[str, Any], optional): The current context available to the agent. Defaults to {}.

    Returns:
        Dict[str, Any]: A dictionary containing the reflection result.
            - status (str): 'success' or 'error'.
            - decision (str): 'execute', 'modify', 'skip', or 'unknown'.
            - justification (str): Explanation for the decision.
            - confidence (str): Confidence level ('Alta', 'Média', 'Baixa') - Placeholder.
            - error_message (str | None): Error details if reflection failed.
    """
    logger.debug(f"Reflecting on step: '{step}' with simulated outcome: '{simulated_outcome}'")

    prompt = REFLECT_STEP_PROMPT_TEMPLATE.format(
        step=step,
        simulated_outcome=simulated_outcome,
        context=context if context else "Nenhum contexto fornecido."
    )

    try:
        # Correctly consume the async generator even for stream=False
        llm_response_text = ""
        async for chunk in call_llm(prompt, stream=False):
            llm_response_text += chunk

        if not llm_response_text or not isinstance(llm_response_text, str):
            logger.error(f"Reflection LLM call returned invalid data type: {type(llm_response_text)}")
            raise ValueError("LLM reflection response was empty or not a string.")

        parsed_output = _parse_reflection_output(llm_response_text)
        decision = parsed_output["decision"]
        justification = parsed_output["justification"]

        logger.info(f"Reflection complete for step '{step}'. Decision: {decision}. Justification: {justification}")

        # Placeholder for confidence - could be based on LLM certainty or keyword analysis
        confidence = "Média" # Default confidence

        return {
            "status": "success",
            "decision": decision,
            "justification": justification,
            "confidence": confidence,
            "error_message": None
        }

    except Exception as e:
        logger.exception(f"Error during step reflection LLM call for step '{step}':")
        return {
            "status": "error",
            "decision": "unknown",
            "justification": "Failed to reflect on step due to LLM error.",
            "confidence": "N/A",
            "error_message": f"Failed to reflect on step: {e}"
        }

# Ensure the skill is registered (assuming automatic loading from skills directory) 