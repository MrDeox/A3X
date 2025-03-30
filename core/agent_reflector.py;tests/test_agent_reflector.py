from typing import List, Dict, Any, Optional, Tuple, Literal, TYPE_CHECKING

async def reflect_on_observation(
    # ... (arguments) ...
):
    # ... (logging)

    # --- Error Handling Logic --- 
    if observed_action == "tool_not_found":
        agent_logger.warning(f"[Reflector] Tool '{action_name}' not found. Suggesting step retry.")
        # Future: Could try asking user or replanning here.
        return "retry_step", new_plan
        
    elif observed_action == "execution_failed":
        # ... (existing auto-correct setup) ...
        # 1. Build Meta-Objective
        error_detail = error_message # Default to full message
        if isinstance(error_message, str):
            lines = error_message.strip().split('\n')
            for line in reversed(lines):
                if line.strip():
                    error_detail = line.strip()
                    break 
        meta_objective = (
            f"The following {action_input.get('language', 'python')} code failed execution:\n"
            # ... (rest of meta objective) ...

    elif observed_action in ["parsing_failed", "llm_call_failed", "internal_error"]:
        # These errors typically indicate issues before tool execution or internal agent problems
        # Suggest retrying the step (which usually involves the LLM call again)
        agent_logger.error(f"[Reflector] Internal agent error detected ({observed_action}). Suggesting step retry.")
        return "retry_step", new_plan

    else:
        # Catch-all for other unspecified errors
        agent_logger.error(f"[Reflector] Unhandled error type ({observed_action}). Stopping plan.")
        return "stop_plan", new_plan
# ... (rest of file) 