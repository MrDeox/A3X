# skills/simulation.py
import logging
from typing import Dict, Any
from core.tools import skill
from core.llm_interface import call_llm # <-- CORRECT IMPORT

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
    description="Simula mentalmente o resultado provável de um passo do plano com base no contexto atual.",
    parameters={
        "step": (str, ...),      # Passo do plano a ser simulado (obrigatório)
        "context": (Dict[str, Any], {}), # Contexto atual do agente (opcional)
    }
)
async def simulate_step(step: str, context: Dict[str, Any] = {}) -> Dict[str, Any]:
    """
    Simulates the likely outcome of executing a plan step given the current context.

    Args:
        step (str): The plan step to simulate.
        context (Dict[str, Any], optional): The current context available to the agent. Defaults to {}.

    Returns:
        Dict[str, Any]: A dictionary containing the simulation result.
            - status (str): 'success' or 'error'.
            - simulated_outcome (str | None): A description of the likely outcome, or None if simulation failed.
            - confidence (str): Confidence level ('Alta', 'Média', 'Baixa') - Placeholder for now.
            - error_message (str | None): Error details if simulation failed.
    """
    logger.debug(f"Simulating step: '{step}' with context: {context}")

    prompt = SIMULATE_STEP_PROMPT_TEMPLATE.format(step=step, context=context if context else "Nenhum contexto fornecido.")

    try:
        # Correctly consume the async generator even for stream=False
        llm_response_text = ""
        async for chunk in call_llm(prompt, stream=False):
            llm_response_text += chunk

        if not llm_response_text or not isinstance(llm_response_text, str):
             logger.error(f"Simulation LLM call returned invalid data type: {type(llm_response_text)}")
             raise ValueError("LLM simulation response was empty or not a string.")

        simulated_outcome = llm_response_text.strip()
        logger.info(f"Simulation successful for step '{step}'. Outcome: {simulated_outcome}")

        # Placeholder for confidence assessment - could involve analyzing the LLM response text
        confidence = "Média" # Default confidence for now

        return {
            "status": "success",
            "simulated_outcome": simulated_outcome,
            "confidence": confidence,
            "error_message": None
        }

    except Exception as e:
        logger.exception(f"Error during step simulation LLM call for step '{step}':")
        return {
            "status": "error",
            "simulated_outcome": None,
            "confidence": "N/A",
            "error_message": f"Failed to simulate step due to LLM error: {e}"
        }

# Ensure the skill is registered by importing it in skills/__init__.py if needed
# (Assuming the current setup automatically loads modules in the skills directory) 