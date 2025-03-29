import logging
import json
from typing import TYPE_CHECKING

# Avoid circular import for type hinting
if TYPE_CHECKING:
    from .agent import ReactAgent

# Get the agent logger instance directly from log_config
from .log_config import agent_logger

def handle_parsing_error(agent_instance: 'ReactAgent', parse_error: Exception, current_history: list, current_iteration: int, max_iterations: int, is_meta_objective: bool, meta_depth: int) -> tuple[bool, str | None]:
    """
    Handles JSON parsing errors from the LLM response.

    Logs the error, appends an observation to the history, and determines
    if the main loop should continue.

    Args:
        agent_instance: The ReactAgent instance (used for logging context).
        parse_error: The exception caught (JSONDecodeError or ValueError).
        current_history: The current conversation history list.
        current_iteration: The current iteration number.
        max_iterations: The maximum allowed iterations.
        is_meta_objective: Flag indicating if this is a meta-cycle.
        meta_depth: The current depth of the meta-cycle.

    Returns:
        tuple[bool, str | None]: (should_continue, final_error_message)
                                 - True if the loop should continue.
                                 - False if the loop should break, along with the error message.
    """
    log_prefix = f"[ReactAgent META-{meta_depth}]" if is_meta_objective else "[ReactAgent]"
    log_prefix += f"[Iter {current_iteration}/{max_iterations}]"

    error_message = f"Erro: Falha ao parsear a resposta do LLM como JSON válido. Erro: {parse_error}"
    agent_logger.error(f"{log_prefix} {error_message}")
    observation = f"Observation: {error_message}"
    current_history.append(observation)
    agent_logger.debug(f"{log_prefix} Histórico atualizado após erro de parsing: {current_history}")

    if current_iteration >= max_iterations:
        agent_logger.warning(f"{log_prefix} Limite máximo de iterações ({max_iterations}) atingido após erro de parsing.")
        final_msg = f"Erro: Atingido limite de iterações ({max_iterations}) após falha em parsear resposta do LLM."
        # No need to save state here, it will be saved in the main loop's finally block
        return False, final_msg
    else:
        return True, None # Continue loop


def handle_llm_call_error(agent_instance: 'ReactAgent', llm_error_msg: str, current_history: list, current_iteration: int, max_iterations: int, is_meta_objective: bool, meta_depth: int) -> tuple[bool, str | None]:
    """
    Handles errors during the LLM call (e.g., network issues, API errors reported by call_llm).

    Logs the error, appends an observation to the history, and determines
    if the main loop should continue.

    Args:
        agent_instance: The ReactAgent instance (used for logging context).
        llm_error_msg: The error message string received (prefixed with "Erro:").
        current_history: The current conversation history list.
        current_iteration: The current iteration number.
        max_iterations: The maximum allowed iterations.
        is_meta_objective: Flag indicating if this is a meta-cycle.
        meta_depth: The current depth of the meta-cycle.

    Returns:
        tuple[bool, str | None]: (should_continue, final_error_message)
                                 - True if the loop should continue.
                                 - False if the loop should break, along with the error message.
    """
    log_prefix = f"[ReactAgent META-{meta_depth}]" if is_meta_objective else "[ReactAgent]"
    log_prefix += f"[Iter {current_iteration}/{max_iterations}]"

    agent_logger.error(f"{log_prefix} Erro na chamada LLM: {llm_error_msg}")
    # Ensure the observation starts with "Observation: "
    if llm_error_msg.startswith("Observation: "):
        observation = llm_error_msg
    elif llm_error_msg.startswith("Erro: "):
         observation = f"Observation: {llm_error_msg}" # Keep "Erro:" prefix if present
    else:
        observation = f"Observation: Erro: {llm_error_msg}" # Add standard error prefix

    current_history.append(observation)
    agent_logger.debug(f"{log_prefix} Histórico atualizado após erro LLM: {current_history}")

    if current_iteration >= max_iterations:
        agent_logger.warning(f"{log_prefix} Limite máximo de iterações ({max_iterations}) atingido após erro na chamada LLM.")
        final_msg = f"Erro: Atingido limite de iterações ({max_iterations}) após falha na comunicação com LLM."
        # No need to save state here, it will be saved in the main loop's finally block
        return False, final_msg
    else:
        return True, None # Continue loop

