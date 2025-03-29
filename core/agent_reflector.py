# core/agent_reflector.py
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Literal

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
    "ask_user"
]

def reflect_on_observation(
    objective: str,
    plan: List[str],
    current_step_index: int,
    action_name: str,
    action_input: dict,
    observation_dict: dict,
    history: list,
    memory: MemoryType,
    agent_logger: logging.Logger
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

    Returns:
        A tuple containing:
            - decision (Decision): The suggested next action for the agent loop.
            - new_plan (Optional[List[str]]): A revised plan, if applicable (currently unused).
    """
    agent_logger.info("[Reflector] Reflecting on observation...")
    agent_logger.debug(f"[Reflector] Action: {action_name}, Input: {action_input}")
    agent_logger.debug(f"[Reflector] Observation: {json.dumps(observation_dict, indent=2, ensure_ascii=False)}")

    status = observation_dict.get("status", "unknown")
    observed_action = observation_dict.get("action", "unknown")
    new_plan: Optional[List[str]] = None

    # --- Decision Logic --- 

    if status == "success":
        agent_logger.info(f"[Reflector] Action '{action_name}' completed successfully.")
        
        # Check if the successful action was the final one needed
        if action_name == "final_answer" or observed_action == "final_answer": # Check both intended and observed
            agent_logger.info("[Reflector] Final Answer provided. Plan complete.")
            return "plan_complete", new_plan 
        else:
            # Successful step, continue the plan
            return "continue_plan", new_plan
            
    elif status == "no_change":
         agent_logger.info(f"[Reflector] Action '{action_name}' resulted in no change. Continuing plan.")
         return "continue_plan", new_plan

    elif status == "error":
        error_message = observation_dict.get("data", {}).get("message", "Unknown error")
        agent_logger.error(f"[Reflector] Error detected during action '{action_name}'. Status: {status}, Action: {observed_action}, Message: {error_message}")

        # --- Error Handling Logic --- 
        if observed_action == "tool_not_found":
            agent_logger.warning(f"[Reflector] Tool '{action_name}' not found. Stopping plan.")
            # Future: Could try asking user or replanning here.
            return "stop_plan", new_plan
            
        elif observed_action == "execution_failed":
            agent_logger.warning(f"[Reflector] Code execution failed for action '{action_name}'.")
            # <<< PLACEHOLDER for Auto-Correction Logic >>>
            # Here, we would call the logic moved from agent.py:
            # 1. Build meta-objective (e.g., "Correct the code that caused this error: {error_message}")
            # 2. Call agent.run(meta_objective) recursively (needs agent instance access or refactoring)
            # 3. Process result: 
            #    - If successful modification -> return "replace_step_and_retry", [new_step]
            #    - If failed modification -> return "stop_plan", None
            agent_logger.info("[Reflector] Auto-correction for execution_failed not implemented yet. Stopping plan.")
            return "stop_plan", new_plan # Stop plan for now
            
        elif observed_action in ["parsing_failed", "llm_call_failed", "internal_error"]:
            # These errors typically indicate issues before tool execution or internal agent problems
            agent_logger.error(f"[Reflector] Internal agent error detected ({observed_action}). Stopping plan.")
            return "stop_plan", new_plan

        else:
            # Catch-all for other unspecified errors
            agent_logger.error(f"[Reflector] Unhandled error type ({observed_action}). Stopping plan.")
            return "stop_plan", new_plan

    else: # Unknown status
        agent_logger.warning(f"[Reflector] Unknown status '{status}' in observation. Stopping plan as a precaution.")
        return "stop_plan", new_plan


# Note: We need to import Literal for the Decision type hint
from typing import Literal 