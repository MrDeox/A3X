# a3x/fragments/debugger.py
import logging
import json
from typing import Dict, Any, List, Optional

from .base import BaseFragment, FragmentContext # Import FragmentContext
from .registry import fragment # Import the decorator
from a3x.core.llm_interface import LLMInterface # Assuming it needs LLM
from a3x.core.tool_executor import _ToolExecutionContext, execute_tool # Import execute_tool

logger = logging.getLogger(__name__)

@fragment(
    name="DebuggerFragment",
    description="Analyzes persistent task failures and suggests diagnostic or corrective actions.",
    category="Execution", # Or maybe a new 'Diagnosis' category?
    skills=["llm_error_diagnosis", "read_file", "web_search"] # Example skills it might use
)
class DebuggerFragment(BaseFragment): # Inherit from BaseFragment
    """
    A specialized fragment designed to analyze persistent failures during task execution.

    When the orchestrator detects that a specific sub-task has failed multiple times
    consecutively, it invokes this fragment. The DebuggerFragment uses the
    `llm_error_diagnosis` skill, providing it with the context of the failures
    (objective, failed sub-task, error messages, previous actions/observations).
    Its goal is to diagnose the root cause of the failure and suggest potential
    corrective actions or alternative approaches.
    """
    def get_purpose(self) -> str:
        """Returns a string describing the primary goal of this fragment."""
        return "Analyze persistent task failures and suggest diagnostic or corrective actions."

    async def execute(
        self,
        context: FragmentContext, # Changed from ctx: _ToolExecutionContext
        sub_task: Optional[str] = None, # Added sub_task as optional
        objective: Optional[str] = None, # Made optional
        failed_sub_task: Optional[str] = None, # Made optional
        failure_context: Optional[List[Dict[str, Any]]] = None, # Made optional
        max_retries_hit: Optional[int] = None # Made optional
    ) -> Dict[str, Any]:
        """
        Analyzes repeated failures or handles direct diagnostic sub-tasks.

        If called with failure context (failed_sub_task, failure_context, max_retries_hit),
        it performs diagnosis using llm_error_diagnosis.
        If called with only a sub_task (e.g., from orchestrator LLM), it attempts to
        execute the sub-task, potentially using its available skills if appropriate.

        Args:
            context: The execution context (FragmentContext).
            sub_task: The general sub-task assigned by the orchestrator (optional).
            objective: The original objective (optional, used for diagnosis context).
            failed_sub_task: The specific sub-task that failed (optional, for diagnosis).
            failure_context: History of failures for the failed_sub_task (optional, for diagnosis).
            max_retries_hit: Number of retries hit (optional, for diagnosis).

        Returns:
            A dictionary with status and data/message.
        """
        log_prefix = "[DebuggerFragment]"
        logger = context.logger # Use logger from context

        # --- Check if called in Diagnosis Mode --- 
        is_diagnosis_mode = failed_sub_task and failure_context and max_retries_hit is not None

        if is_diagnosis_mode:
            logger.info(f"{log_prefix} Initiating diagnosis for failed sub-task: '{failed_sub_task}' (failed {max_retries_hit} times)")

            # --- Prepare Context for llm_error_diagnosis --- 
            last_failure = failure_context[-1] if failure_context else {}
            last_action = last_failure.get('action', 'N/A')
            last_observation = last_failure.get('observation', {})
            error_message = "Unknown error (Observation format varies)"
            if isinstance(last_observation, dict):
                error_message = (
                    last_observation.get('message')
                    or last_observation.get('data', {}).get('message')
                    or last_observation.get('result', {}).get('data', {}).get('message')
                    or json.dumps(last_observation)
                )
            elif isinstance(last_observation, str):
                error_message = last_observation
            
            formatted_failure_context = [
                f"Attempt {i+1}: Action='{f.get('action','N/A')}', Observation={str(f.get('observation',{}))[:200]}..."
                for i, f in enumerate(failure_context[-3:])
            ]

            diagnosis_input = {
                "error_message": str(error_message)[:500],
                "traceback": None, 
                "execution_context": {
                    "original_objective": objective or "Unknown Objective",
                    "failed_sub_task": failed_sub_task,
                    "consecutive_failures": max_retries_hit,
                    "last_attempted_action": last_action,
                    "recent_failure_history": formatted_failure_context
                }
            }

            # --- Call llm_error_diagnosis Skill --- 
            try:
                logger.info(f"{log_prefix} Calling llm_error_diagnosis skill...")
                diagnosis_result_wrapped = await execute_tool(
                    tool_name="llm_error_diagnosis",
                    action_input=diagnosis_input,
                    tools_dict=context.tool_registry, # Use context.tool_registry
                    context=context # Pass FragmentContext
                )
                
                diagnosis_result = diagnosis_result_wrapped.get("result", {})
                metrics = diagnosis_result_wrapped.get("metrics", {})
                logger.info(f"{log_prefix} llm_error_diagnosis completed. Status: {diagnosis_result.get('status')}, Metrics: {metrics}")

                if diagnosis_result.get("status") == "success":
                    diagnosis_data = diagnosis_result.get("data", {})
                    diagnosis = diagnosis_data.get("diagnosis", "Diagnosis not provided by LLM.")
                    suggested_actions = diagnosis_data.get("suggested_actions", [])
                    suggestion = {
                        "type": "diagnosis_provided",
                        "raw_diagnosis": diagnosis,
                        "raw_suggestions": suggested_actions,
                        "message": "Further action based on diagnosis needs implementation."
                    }
                    logger.info(f"{log_prefix} Diagnosis: {diagnosis}")
                    logger.info(f"{log_prefix} Suggested Actions: {suggested_actions}")
                    return {
                        "status": "success",
                        "data": {
                            "diagnosis": diagnosis,
                            "suggestion": suggestion
                        }
                    }
                else:
                    error_msg = diagnosis_result.get("data", {}).get("message", "llm_error_diagnosis skill failed.")
                    logger.error(f"{log_prefix} llm_error_diagnosis skill failed: {error_msg}")
                    return {"status": "error", "data": {"message": f"Failed to get diagnosis: {error_msg}"}}

            except Exception as e:
                logger.exception(f"{log_prefix} Error occurred while trying to run diagnosis:")
                return {"status": "error", "data": {"message": f"Unexpected error during diagnosis: {e}"}}
        
        elif sub_task:
            # --- Handle Direct Sub-task Mode (e.g., called by Orchestrator LLM) ---
            logger.warning(f"{log_prefix} Received direct sub-task: '{sub_task}'. This fragment primarily handles diagnosis. Attempting generic execution or returning error.")
            # Option 1: Try to execute sub-task using available skills? Risky.
            # Option 2: Return an error indicating wrong delegation.
            return {
                "status": "error",
                "message": f"DebuggerFragment received unexpected direct sub-task '{sub_task}'. It should typically be invoked for diagnosis.",
                "data": {"sub_task": sub_task}
            }
        
        else:
            # --- Invalid Call --- 
             logger.error(f"{log_prefix} Execute called without sufficient arguments for diagnosis or a direct sub-task.")
             return {"status": "error", "message": "DebuggerFragment called with invalid arguments."} 