# a3x/fragments/debugger.py
import logging
import json
from typing import Dict, Any, List, Optional

from .base import BaseFragment # Assuming BaseFragment is defined here or imported
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
    async def execute(
        self,
        ctx: _ToolExecutionContext,
        objective: str,
        failed_sub_task: str,
        failure_context: List[Dict[str, Any]], # e.g., list of {'action': ..., 'observation': ..., 'error': ...}
        max_retries_hit: int
    ) -> Dict[str, Any]:
        """
        Analyzes repeated failures for a specific sub-task and proposes next steps.

        This method orchestrates the diagnosis process:
        1. Extracts relevant information (last action, error message) from the failure context.
        2. Formats the recent failure history for concise presentation to the LLM.
        3. Constructs the input for the `llm_error_diagnosis` skill.
        4. Executes the `llm_error_diagnosis` skill using `execute_tool`.
        5. Parses the diagnosis result, extracting the diagnosis text and suggested actions.
        6. Returns a structured result containing the diagnosis and suggestions.

        Args:
            ctx: The execution context, providing access to tools and other resources.
            objective: The original, high-level objective the agent is trying to achieve.
            failed_sub_task: The specific sub-task that has been failing repeatedly.
            failure_context: A list of dictionaries, where each dictionary represents a
                             failed attempt for the `failed_sub_task`. It should ideally
                             contain keys like 'action', 'observation', and potentially 'error'.
            max_retries_hit: The number of consecutive times the `failed_sub_task` has failed.

        Returns:
            A dictionary containing:
            - `status`: 'success' if diagnosis was obtained, 'error' otherwise.
            - `data`: A dictionary with:
                - `diagnosis`: (str) The textual diagnosis provided by the LLM skill.
                - `suggestion`: (Dict) A structured suggestion based on the diagnosis
                                  (currently includes raw diagnosis/suggestions, needs refinement).
                - `message`: (str) An error message if the status is 'error'.
        """
        log_prefix = "[DebuggerFragment]"
        logger.info(f"{log_prefix} Initiating diagnosis for failed sub-task: '{failed_sub_task}' (failed {max_retries_hit} times)")

        # --- Prepare Context for llm_error_diagnosis --- 
        # Extract relevant info from the failure context
        last_failure = failure_context[-1] if failure_context else {}
        last_action = last_failure.get('action', 'N/A')
        last_observation = last_failure.get('observation', {})
        # Try to get a specific error message from the observation
        error_message = "Unknown error (Observation format varies)"
        if isinstance(last_observation, dict):
            # Look in common places for error messages
            error_message = (
                last_observation.get('message')
                or last_observation.get('data', {}).get('message')
                or last_observation.get('result', {}).get('data', {}).get('message')
                or json.dumps(last_observation) # Fallback to full observation
            )
        elif isinstance(last_observation, str):
            error_message = last_observation
        
        # Limit context size for the prompt
        formatted_failure_context = [
            f"Attempt {i+1}: Action='{f.get('action','N/A')}', Observation={str(f.get('observation',{}))[:200]}..."
            for i, f in enumerate(failure_context[-3:]) # Limit to last 3 attempts for prompt clarity
        ]

        diagnosis_input = {
            "error_message": str(error_message)[:500], # Limit length
            "traceback": None, # TODO: How to get traceback if available?
            "execution_context": {
                "original_objective": objective,
                "failed_sub_task": failed_sub_task,
                "consecutive_failures": max_retries_hit,
                "last_attempted_action": last_action,
                "recent_failure_history": formatted_failure_context
            }
        }

        # --- Call llm_error_diagnosis Skill --- 
        try:
            logger.info(f"{log_prefix} Calling llm_error_diagnosis skill...")
            # Use the execute_tool function, passing the current fragment's context
            diagnosis_result_wrapped = await execute_tool(
                tool_name="llm_error_diagnosis",
                action_input=diagnosis_input,
                tools_dict=ctx.tools_dict, # Use tools from the passed context
                context=ctx # Pass the full context
            )
            
            # Extract the actual result payload
            diagnosis_result = diagnosis_result_wrapped.get("result", {})
            metrics = diagnosis_result_wrapped.get("metrics", {})
            logger.info(f"{log_prefix} llm_error_diagnosis completed. Status: {diagnosis_result.get('status')}, Metrics: {metrics}")

            if diagnosis_result.get("status") == "success":
                diagnosis_data = diagnosis_result.get("data", {})
                diagnosis = diagnosis_data.get("diagnosis", "Diagnosis not provided by LLM.")
                suggested_actions = diagnosis_data.get("suggested_actions", [])

                # TODO: Implement more sophisticated suggestion parsing/generation based on suggested_actions
                # For now, just return the raw diagnosis and a placeholder suggestion
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
                # Diagnosis skill itself failed
                error_msg = diagnosis_result.get("data", {}).get("message", "llm_error_diagnosis skill failed.")
                logger.error(f"{log_prefix} llm_error_diagnosis skill failed: {error_msg}")
                return {"status": "error", "data": {"message": f"Failed to get diagnosis: {error_msg}"}}

        except Exception as e:
            logger.exception(f"{log_prefix} Error occurred while trying to run diagnosis:")
            return {"status": "error", "data": {"message": f"Unexpected error during diagnosis: {e}"}} 