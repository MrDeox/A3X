# skills/simulation.py
import logging
import json
from typing import Dict, Any, Optional

from a3x.core.skills import skill
# Correct import
from a3x.core.llm_interface import LLMInterface # <-- IMPORT CLASS
# Import context type for hinting
from a3x.core.agent import _ToolExecutionContext 
from a3x.core.context import Context # Added import

logger = logging.getLogger(__name__)

SIMULATE_STEP_PROMPT_TEMPLATE = """
Contexto Atual:
{context}

Passo do Plano a Simular: {step}

Com base no contexto e no passo do plano, simule mentalmente o resultado mais provável da execução deste passo.
Descreva o resultado esperado de forma concisa e objetiva. Inclua qual ferramenta provavelmente seria usada e qual seria o desfecho principal.
Se o passo parecer inviável ou propenso a erro, descreva o problema esperado.

Resultado Simulado:
"""


@skill(
    name="simulate_step",
    description="Simulates the outcome of a planned step without actually executing it.",
    parameters={
        "context": {"type": Context, "description": "Execution context for LLM access and state info."},
        "step": {"type": str, "description": "The planned step to simulate."},
        "current_state": {"type": Optional[Dict[str, Any]], "default": None, "description": "Optional dictionary representing the current world/system state."}
    }
)
async def simulate_step(context: Context, step: str, current_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Simulates the likely outcome of executing a plan step given the current context.
    Uses the LLMInterface from the execution context.
    Args:
        step (str): The plan step to simulate.
        ctx (_ToolExecutionContext): The execution context.
    Returns:
        Dict[str, Any]: A dictionary containing the simulation result.
    """
    # Get components from context
    logger = context.logger
    llm_interface = context.llm_interface
    memory_context = context.memory.get_memory() # Get current memory context

    if not llm_interface:
        logger.error("LLMInterface not found in execution context for simulation.")
        return {"status": "error", "simulated_outcome": None, "confidence": "N/A", "error_message": "Internal error: LLMInterface missing."}

    logger.debug(f"Simulating step: '{step}' with context: {memory_context}")

    prompt = SIMULATE_STEP_PROMPT_TEMPLATE.format(
        step=step, 
        context=memory_context if memory_context else "Nenhum contexto fornecido."
    )

    try:
        llm_response_text = ""
        messages = [{"role": "user", "content": prompt}]
        # Updated call site
        async for chunk in llm_interface.call_llm( # <-- USE INSTANCE METHOD
            messages=messages, 
            stream=False
        ):
            llm_response_text += chunk

        if not llm_response_text or not isinstance(llm_response_text, str):
            logger.error(f"Simulation LLM call returned invalid data type: {type(llm_response_text)}")
            return {"status": "error", "simulated_outcome": None, "confidence": "N/A", "error_message": "LLM response was empty or not a string."}
        
        # Check for LLM error string
        if llm_response_text.startswith("[LLM Error:"):
             logger.error(f"LLM call failed during simulation: {llm_response_text}")
             return {"status": "error", "simulated_outcome": None, "confidence": "N/A", "error_message": llm_response_text}

        simulated_outcome = llm_response_text.strip()
        logger.info(f"Simulation successful for step '{step}'. Outcome: {simulated_outcome}")

        confidence = "Média" # Placeholder confidence

        return {
            "status": "success",
            "simulated_outcome": simulated_outcome,
            "confidence": confidence,
            "error_message": None,
        }

    except Exception as e:
        logger.exception(f"Error during step simulation LLM call for step '{step}':")
        return {
            "status": "error",
            "simulated_outcome": None,
            "confidence": "N/A",
            "error_message": f"Failed to simulate step due to LLM error: {e}",
        }


# Ensure the skill is registered by importing it in skills/__init__.py if needed
# (Assuming the current setup automatically loads modules in the skills directory)
