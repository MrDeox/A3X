# a3x/core/executors/action_executor.py
import logging
from typing import Dict, Any, Optional

from a3x.core.types import ActionIntent
from a3x.core.context import SharedTaskContext # Needed for context passing?
from a3x.core.tool_executor import ToolExecutor # Import the existing ToolExecutor
from a3x.core.constants import STATUS_SUCCESS, STATUS_ERROR, REASON_ACTION_EXECUTION_FAILED, REASON_SKILL_NOT_FOUND

logger = logging.getLogger(__name__)

class ActionExecutor:
    """
    Responsible for receiving structured ActionIntents and executing the corresponding
    symbolic skill using the central ToolExecutor.
    """
    def __init__(self, tool_executor: ToolExecutor):
        """
        Initializes the ActionExecutor.

        Args:
            tool_executor: An instance of the ToolExecutor which handles skill execution.
        """
        self.tool_executor = tool_executor
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("ActionExecutor initialized.")

    async def execute_intent(
        self, 
        intent: ActionIntent, 
        shared_task_context: SharedTaskContext # Pass the full context
    ) -> Dict[str, Any]:
        """
        Executes a skill based on the provided ActionIntent.

        Args:
            intent: The ActionIntent object describing the action to perform.
            shared_task_context: The context of the current task, potentially needed 
                                 by the ToolExecutor or the skill itself.

        Returns:
            A dictionary containing the execution status and result.
        """
        skill_name = intent.skill_target
        parameters = intent.parameters
        requested_by = intent.requested_by or "UnknownFragment"
        reasoning = intent.reasoning or "No reasoning provided."
        
        log_prefix = f"[ActionExecutor Task {shared_task_context.task_id}]"
        self.logger.info(f"{log_prefix} Executing intent from '{requested_by}': Skill='{skill_name}', Params={parameters}. Reasoning: {reasoning}")

        # --- Use ToolExecutor to run the skill --- 
        try:
            # ToolExecutor needs tool_name, tool_input, and the execution context
            # We pass the SharedTaskContext here; ToolExecutor should internally 
            # construct the specific context needed for the skill.
            result_dict = await self.tool_executor.execute_tool(
                tool_name=skill_name,
                tool_input=parameters,
                context=shared_task_context # Pass the shared context 
            )

            # Check the result structure from ToolExecutor
            if not isinstance(result_dict, dict):
                self.logger.error(f"{log_prefix} ToolExecutor returned non-dict result for skill '{skill_name}': {result_dict}")
                return {
                    "status": STATUS_ERROR,
                    "reason": REASON_ACTION_EXECUTION_FAILED,
                    "message": f"Internal error: ToolExecutor returned invalid format for {skill_name}.",
                    "skill": skill_name
                }
            
            exec_status = result_dict.get("status", STATUS_ERROR) # Assume error if status missing
            if exec_status == STATUS_SUCCESS:
                self.logger.info(f"{log_prefix} Successfully executed skill '{skill_name}'. Result: {result_dict}")
            else:
                self.logger.error(f"{log_prefix} Skill '{skill_name}' execution failed. Result: {result_dict}")
            
            # Return the dictionary received from ToolExecutor
            return result_dict

        except Exception as e:
            # Catch potential exceptions from ToolExecutor.execute_tool itself
            # ToolExecutor might already catch and format errors, but this is a failsafe.
            self.logger.exception(f"{log_prefix} Unexpected error during execution of skill '{skill_name}' via ToolExecutor:")
            
            # Check if the error indicates the skill wasn't found
            # (This might depend on specific exceptions raised by ToolExecutor or ToolRegistry)
            if "not found" in str(e).lower() or "unknown tool" in str(e).lower(): # Heuristic check
                 reason = REASON_SKILL_NOT_FOUND
            else:
                 reason = REASON_ACTION_EXECUTION_FAILED
                 
            return {
                "status": STATUS_ERROR,
                "reason": reason,
                "message": f"Failed to execute action '{skill_name}': {e}",
                "skill": skill_name
            } 