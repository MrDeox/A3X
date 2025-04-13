# core/agent_reflector.py
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING, Literal

# from a3x.core.llm_interface import call_llm # Removed unused import

# Avoid circular import for type hinting
if TYPE_CHECKING:
    # from .agent import ReactAgent
    from a3x.core.agent import ReactAgent

# Logger for this module
logger = logging.getLogger(__name__)

# Placeholder for types related to memory or agent state if needed later
MemoryType = Dict[str, Any]

# Decision types
Decision = Literal[
    "continue_plan",
    "retry_step",
    "replace_step_and_retry",
    "stop_plan",
    "plan_complete",
    "ask_user",
]


async def reflect_on_observation(
    objective: str,
    plan: List[str],
    current_step_index: int,
    action_name: str,
    action_input: dict,
    observation_dict: dict,
    history: list,
    memory: MemoryType,
    agent_logger: logging.Logger,
    agent_instance: "ReactAgent",
) -> Tuple[Decision, Optional[List[str]]]:
    """Analyzes the observation from the last action and decides the next course of action.

    Args:
        objective: The overall goal.
        plan: The current plan (list of steps).
        current_step_index: The index of the step just executed.
        action_name: The name of the action executed.
        action_input: The input given to the action.
        observation_dict: The result of the action execution (parsed JSON observation).
        history: The agent's conversation history.
        memory: The agent's memory state.
        agent_logger: The logger instance for agent activities.
        agent_instance: The instance of the ReactAgent for potential recursive calls.

    Returns:
        A tuple containing:
            - decision (Decision): The suggested next action for the agent loop.
            - new_plan (Optional[List[str]]): A revised plan, if applicable (currently unused).
    """
    agent_logger.info("[Reflector] Reflecting on observation...")
    agent_logger.debug(f"[Reflector] Action: {action_name}, Input: {action_input}")
    agent_logger.debug(
        f"[Reflector] Observation: {json.dumps(observation_dict, indent=2, ensure_ascii=False)}"
    )

    status = observation_dict.get("status", "unknown")
    observed_action = observation_dict.get("action", "unknown")
    new_plan: Optional[List[str]] = None

    # --- Decision Logic ---

    if status == "success":
        agent_logger.info(f"[Reflector] Action '{action_name}' completed successfully.")

        # Check if the successful action was the final one needed
        if (
            action_name == "final_answer" or observed_action == "final_answer"
        ):  # Check both intended and observed
            agent_logger.info("[Reflector] Final Answer provided. Plan complete.")
            return "plan_complete", new_plan
        else:
            # Successful step, continue the plan
            return "continue_plan", new_plan

    elif status == "no_change":
        agent_logger.info(
            f"[Reflector] Action '{action_name}' resulted in no change. Continuing plan."
        )
        return "continue_plan", new_plan

    elif status == "error":
        error_message = observation_dict.get("data", {}).get("message", "Unknown error")
        agent_logger.error(
            f"[Reflector] Error detected during action '{action_name}'. Status: {status}, Action: {observed_action}, Message: {error_message[:500]}..."
        )

        # --- Error Handling Logic ---
        if observed_action == "tool_not_found":
            agent_logger.warning(
                f"[Reflector] Tool '{action_name}' not found. Suggesting step retry."
            )
            return "retry_step", new_plan

        elif observed_action == "execution_failed":
            agent_logger.warning(
                f"[Reflector] Code execution failed for action '{action_name}'. Attempting auto-correction."
            )
            original_code = action_input.get("code")
            if not original_code:
                agent_logger.error(
                    "[Reflector] Cannot attempt correction: Original code not found in action_input."
                )
                return "stop_plan", new_plan

            # 1. Build Meta-Objective
            error_detail = error_message  # Default to full message
            if isinstance(error_message, str):
                lines = error_message.strip().split("\n")
                # Try to get the last non-empty line as error detail
                for line in reversed(lines):
                    if line.strip():
                        error_detail = line.strip()
                        break
            meta_objective = (
                f"The following {action_input.get('language', 'python')} code failed execution:\n"
                f"```\n{original_code}\n```\n"
                f"The error was: {error_detail}\n"
                f"Please analyze the error and the code, then use the 'modify_code' tool to provide a corrected version."
            )
            agent_logger.info(
                f"[Reflector] Generated Meta-Objective for correction: {meta_objective[:200]}..."
            )

            # 2. Call agent.run recursively
            try:
                agent_logger.info(
                    "[Reflector] --- Starting Meta-Cycle for Auto-Correction ---"
                )
                meta_result_str = await agent_instance.run(meta_objective)
                agent_logger.info("[Reflector] --- Meta-Cycle Completed --- ")
                agent_logger.debug(
                    f"[Reflector] Meta-Cycle Result (raw): {meta_result_str}"
                )
            except Exception:
                agent_logger.exception(
                    "[Reflector] Exception during recursive agent run for auto-correction:"
                )
                return "stop_plan", new_plan

            # 3. Process Meta-Result
            try:
                parsed_meta_result = json.loads(meta_result_str)
                if parsed_meta_result.get(
                    "status"
                ) == "success" and "modified_code" in parsed_meta_result.get(
                    "data", {}
                ):
                    modified_code = parsed_meta_result["data"]["modified_code"]
                    agent_logger.info(
                        "[Reflector] Auto-correction successful. Modified code received."
                    )
                    # Save corrected code to memory
                    agent_instance._memory["last_code"] = modified_code
                    agent_logger.info(
                        "[Reflector] Corrected code saved to memory['last_code']. Continuing plan."
                    )
                    # Decide to continue the main plan. The LLM in the next regular cycle
                    # might choose to use the corrected code from memory.
                    return "continue_plan", new_plan
                else:
                    fail_reason = parsed_meta_result.get("data", {}).get(
                        "message", "Unknown reason"
                    )
                    agent_logger.error(
                        f"[Reflector] Auto-correction meta-cycle did not return successful modified code. Status: {parsed_meta_result.get('status')}, Reason: {fail_reason}. Stopping plan."
                    )
                    return "stop_plan", new_plan
            except json.JSONDecodeError as json_err:
                agent_logger.error(
                    f"[Reflector] Failed to parse meta-cycle result as JSON: {json_err}. Result: {meta_result_str[:500]}... Stopping plan."
                )
                return "stop_plan", new_plan
            except Exception:
                agent_logger.exception(
                    "[Reflector] Unexpected error processing meta-cycle result:"
                )
                return "stop_plan", new_plan

        elif observed_action in ["parsing_failed", "llm_call_failed", "internal_error"]:
            agent_logger.error(
                f"[Reflector] Internal agent error detected ({observed_action}). Suggesting step retry."
            )
            return "retry_step", new_plan

        else:
            # Catch-all for other unspecified errors
            agent_logger.error(
                f"[Reflector] Unhandled error type ({observed_action}). Stopping plan."
            )
            return "stop_plan", new_plan

    else:  # Unknown status
        agent_logger.warning(
            f"[Reflector] Unknown status '{status}' in observation. Stopping plan as a precaution."
        )
        return "stop_plan", new_plan

    # Verificar se a observação indica uma falha no LLM (resposta vazia ou erro)
    observation_text = observation_dict.get('content', '')
    if not observation_text or '[LLM Error:' in observation_text:
        agent_logger.info("[Reflector] Falha detectada no LLM. Chamando skill adjust_llm_parameters para ajustar configurações.")
        from a3x.skills.adjust_llm_parameters import adjust_llm_parameters
        adjustment_result = adjust_llm_parameters(context={'mem': memory})
        agent_logger.info(f"[Reflector] Resultado do ajuste do LLM: {adjustment_result.get('message', 'Ajuste falhou')}")
        # Retornar decisão para retry após ajuste
        return 'retry_step', None


# Note: We need to import Literal for the Decision type hint
