from typing import Dict, Any, Optional, List
import logging

# Centralized imports for state and logging
from a3x.api.state import AGENT_STATE
from a3x.api.log_buffer import log_event

hooks_logger = logging.getLogger(__name__)

# --- Task Lifecycle Hooks --- #

def notify_new_task(task_description: str):
    """Called when a new task is received and initiated."""
    try:
        AGENT_STATE.set_current_task(task_description) # Resets state and sets status to 'planning'
        log_event("INFO", f"Task received: '{task_description}'", source="Orchestrator")
    except Exception as e:
        hooks_logger.error(f"Error in notify_new_task: {e}", exc_info=True)

def notify_task_planning_complete(plan: List[str]):
    """Called after the initial plan has been generated."""
    try:
        AGENT_STATE.current_plan = plan
        AGENT_STATE.set_status("executing")
        log_event("INFO", f"Plan generated: {plan}", source="Planner", extra_data={"plan": plan})
    except Exception as e:
        hooks_logger.error(f"Error in notify_task_planning_complete: {e}", exc_info=True)

async def notify_task_completion(task_id: str, final_answer: str = ""):
    """Hook to notify when a task completes successfully."""
    try:
        hooks_logger.info(f"Task {task_id} completed successfully.")
        # Log completion event with final answer if provided
        log_event(
            task_id=task_id,
            event_type=EVENT_TASK_COMPLETION,
            message=f"Task completed successfully.",
            extra_data={"final_answer": final_answer} if final_answer else None
        )
        # Here you could add logic to update a database, send notifications, etc.
    except Exception as e:
        hooks_logger.error(f"Error in notify_task_completion: {e}", exc_info=True)

async def notify_task_error(error_message: str, source: str = "Unknown", error_details: Optional[Dict] = None):
    """Called when the agent encounters an unrecoverable error during the task."""
    try:
        AGENT_STATE.set_error(error_message)
        log_event("ERROR", f"Task failed: {error_message}", source=source, extra_data=error_details if isinstance(error_details, dict) else {"error_details_raw": str(error_details)})
    except Exception as e:
        hooks_logger.error(f"Error in notify_task_error hook itself: {e}", exc_info=True)

# --- Fragment/Step Lifecycle Hooks --- #

async def notify_fragment_selection(fragment_name: str):
    """Called when a specific fragment is chosen to handle a step/sub-task."""
    try:
        AGENT_STATE.set_active_fragment(fragment_name)
        # Don't change overall agent status here, maybe fragment has its own status?
        log_event("DEBUG", f"Fragment selected: {fragment_name}", source="Orchestrator")
    except Exception as e:
        hooks_logger.error(f"Error in notify_fragment_selection: {e}", exc_info=True)

def notify_react_step(step_info: Dict[str, Any], source: str = "AgentLoop"):
    """Called after each ReAct step (thought, action, input, observation)."""
    try:
        AGENT_STATE.update_step(step_info)
        AGENT_STATE.set_status("thinking" if step_info.get("thought") else "executing_tool")
        # Create a compact version for logging to avoid overwhelming logs
        log_data = {
            "thought": step_info.get("thought", "N/A")[:100] + "..." if step_info.get("thought") and len(step_info.get("thought", "")) > 100 else step_info.get("thought"),
            "action": step_info.get("action"),
            # Potentially truncate input/observation too if they can be large
            "input": str(step_info.get("action_input"))[:100] + "..." if step_info.get("action_input") and len(str(step_info.get("action_input"))) > 100 else step_info.get("action_input"),
            "observation_status": step_info.get("observation", {}).get("result", {}).get("status", "N/A")
        }
        log_event("DEBUG", f"ReAct Step: Action='{log_data['action']}'", source=source, extra_data=log_data)
    except Exception as e:
        hooks_logger.error(f"Error in notify_react_step: {e}", exc_info=True)

def notify_tool_execution_start(tool_name: str, tool_input: Dict[str, Any], source: str = "ToolExecutor"):
    """Called just before a tool/skill is executed."""
    try:
        AGENT_STATE.set_status("executing_tool")
        log_event("DEBUG", f"Executing tool: {tool_name}", source=source, extra_data={"tool_input": tool_input})
    except Exception as e:
        hooks_logger.error(f"Error in notify_tool_execution_start: {e}", exc_info=True)

def notify_tool_execution_end(tool_name: str, result: Dict[str, Any], source: str = "ToolExecutor"):
    """Called after a tool/skill has finished execution."""
    try:
        AGENT_STATE.set_status("processing_observation") # Or back to 'thinking'?
        # Check result status for log level
        status = result.get("status", "unknown").lower()
        log_level = "ERROR" if status == "error" else "DEBUG"
        log_event(log_level, f"Tool execution finished: {tool_name} (Status: {status})", source=source, extra_data={"tool_result": result})
    except Exception as e:
        hooks_logger.error(f"Error in notify_tool_execution_end: {e}", exc_info=True)

# --- Heuristic/Memory Hooks (Example) --- #

def notify_heuristic_applied(heuristic_id: str, description: str):
    """Called when a stored heuristic is successfully applied."""
    try:
        AGENT_STATE.last_heuristic_used = heuristic_id
        log_event("INFO", f"Heuristic applied: {heuristic_id} - {description}", source="HeuristicEngine")
    except Exception as e:
        hooks_logger.error(f"Error in notify_heuristic_applied: {e}", exc_info=True) 