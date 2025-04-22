import logging
import random
from typing import Dict, Any, Optional, List

from a3x.core.skills import skill
from a3x.core.context import Context # Basic context type

logger = logging.getLogger(__name__)

# Placeholder lists of potential outcomes
SUCCESS_OUTCOMES = [
    "Action completed successfully.",
    "Task finished as expected.",
    "Operation successful.",
    "Expected output generated."
]
ERROR_OUTCOMES = [
    "File not found.",
    "Permission denied.",
    "Invalid input parameters.",
    "Network connection failed.",
    "Timeout during execution.",
    "Unexpected internal error."
]

@skill(
    name="simulate_execution",
    description="Simula a execução de uma ação/skill, retornando um resultado de sucesso ou erro plausível.",
    parameters={
        "action_name": {"type": str, "description": "O nome da ação/skill sendo simulada."},
        "action_input": {"type": Dict[str, Any], "description": "Os parâmetros de entrada fornecidos para a ação.", "default": {}},
        "force_outcome": {"type": Optional[str], "enum": ["success", "error"], "description": "Força um resultado específico (success/error) para fins de teste (opcional).", "default": None}
    }
)
async def simulate_execution(
    ctx: Context, # Or SkillContext
    action_name: str,
    action_input: Optional[Dict[str, Any]] = None,
    force_outcome: Optional[str] = None
) -> Dict[str, Any]:
    """Simulates the execution of a skill, returning a plausible success or error."""
    action_input = action_input or {}
    logger.info(f"Simulating execution for action: '{action_name}' with input keys: {list(action_input.keys())}")

    outcome = "success"
    message = ""

    if force_outcome:
        outcome = force_outcome
        logger.info(f"Forcing outcome to: {outcome}")
    else:
        # Simple random simulation: 80% chance of success
        if random.random() < 0.8:
            outcome = "success"
        else:
            outcome = "error"

    if outcome == "success":
        message = random.choice(SUCCESS_OUTCOMES)
    else:
        message = random.choice(ERROR_OUTCOMES)

    # Structure the output similar to a real skill execution
    result = {
        "status": outcome,
        # Add more plausible output fields based on action_name if needed
        "output": message if outcome == "success" else None,
        "error": message if outcome == "error" else None,
        "action": action_name # Include the action name for clarity
    }

    logger.info(f"Simulation result for '{action_name}': {result['status']} - {message}")
    return result 