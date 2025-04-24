import logging
import json
from typing import Dict, Any, Optional
import inspect
from pathlib import Path

# <<< Import the new sandbox function >>>
from ..core.sandbox import execute_code_in_sandbox

# <<< Import base and decorator >>>
from .base import BaseFragment, FragmentContext
from .manager_fragment import ManagerFragment
from .registry import fragment

logger = logging.getLogger(__name__)

# --- Define Skills Managed by this Manager --- 
# This fragment might not *manage* the skill anymore, but rather *uses* the sandbox.
# Let's rename the managed skill if this fragment now acts more like a coordinator.
# Or, if it just executes code directly via the sandbox, it might not manage any skill.
# For now, let's assume it still conceptually represents code execution capability.
MANAGED_SKILLS = ["execute_python_code"] # Keep name consistent for now, but consider if this fragment should expose a skill.

# <<< Apply the decorator >>>
@fragment(
    name="CodeExecutionManager", # Consider renaming if role changes
    description="Executes code snippets using the centralized sandbox.", # Updated description
    category="Execution", # Changed category?
    managed_skills=MANAGED_SKILLS
)
class CodeExecutionManager(ManagerFragment):
    """Manager Fragment responsible for handling code execution tasks via the sandbox."""

    def get_purpose(self) -> str:
        """Returns the purpose of this fragment."""
        return "To execute provided code snippets safely via the core sandbox and return the results."

    # <<< REVISED execute method >>>
    async def execute(self, sub_task: str, context: FragmentContext) -> Dict[str, Any]:
        """
        Executes the code specified in the sub_task using the centralized sandbox function.
        Assumes the sub_task IS the code snippet.
        """
        logger = context.logger
        logger.info(f"[CodeExecutionManager] Received sub-task for execution: {sub_task[:100]}...")
        
        # --- Basic Code Extraction (Placeholder - keep as is for now) ---
        code_to_execute = sub_task
        language = "python" # Default assumption
        if sub_task.strip().startswith("```python"):
            code_to_execute = sub_task.strip()[9:-3].strip()
        elif sub_task.strip().startswith("```bash") or sub_task.strip().startswith("```sh"):
            # Note: sandbox currently only supports python
            language = "bash"
            code_to_execute = sub_task.strip()[7:-3].strip()
        elif sub_task.strip().startswith("```"):
             code_to_execute = sub_task.strip()[3:-3].strip()

        # --- Call the centralized sandbox function --- 
        logger.info(f"Delegating execution of {language} code to core sandbox.")
        try:
            # Prepare skill_context for logging within the sandbox function
            sandbox_skill_context = {
                "skill_name": self.fragment_id # Identify the caller
                # Add other relevant context if needed
            }
            
            # Call the sandbox function
            # Assuming DEFAULT_TIMEOUT_SECONDS is defined in sandbox or use None
            execution_result = execute_code_in_sandbox(
                code=code_to_execute,
                language=language,
                timeout=None, # Or pass a specific timeout if needed
                shared_context=context.shared_task_context,
                skill_context=sandbox_skill_context 
            )
            
            logger.info(f"Sandbox execution finished with status: {execution_result.get('status')}")
            
            # --- Process result and update shared context --- 
            final_status = execution_result.get("status", "error")
            # Construct a message based on the result
            if final_status == "success":
                 final_message = f"Execution successful. Output:\n{execution_result.get('stdout')}"
            elif final_status == "timeout":
                 final_message = f"Execution timed out. Error:\n{execution_result.get('stderr')}"
            elif final_status == "ast_block":
                 final_message = f"Execution blocked by safety checks: {execution_result.get('stderr')}"
            else: # Generic error
                 final_message = f"Execution failed. Error:\n{execution_result.get('stderr')}"
                 
            # Update shared context with the raw result if needed
            if context and hasattr(context, 'shared_task_context'):
                 await context.shared_task_context.update_data("last_execution_result", execution_result)
            
            # Return a result structure suitable for the fragment's caller
            return {
                 "status": final_status, 
                 "message": final_message, 
                 "data": execution_result # Return the raw sandbox result in 'data'
            }

        except Exception as e:
            logger.exception(f"Unexpected error calling sandbox from {self.fragment_id}: {e}")
            error_details = f"Fragment: {self.fragment_id}, Error calling sandbox: {e}"
            return {"status": "error", "message": f"Unexpected error during execution: {error_details}"}

    # Remove old execution logic / helper methods if they existed here
    # e.g., remove _execute_generated_code if it existed