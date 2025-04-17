import logging
import json
from typing import Dict, Any, Optional
import inspect
from pathlib import Path

# <<< Import base and decorator >>>
from .base import ManagerFragment, FragmentContext
from ..core.context import _ToolExecutionContext
from .registry import fragment

logger = logging.getLogger(__name__)

# --- Define Skills Managed by this Manager --- 
CODE_EXEC_SKILLS = ["execute_code"]

# <<< Apply the decorator >>>
@fragment(
    name="CodeExecutionManager",
    description="Manages code execution, including validation and running code blocks.",
    category="Management",
    managed_skills=CODE_EXEC_SKILLS
)
class CodeExecutionManager(ManagerFragment):
    """Manager Fragment responsible for handling code execution tasks."""

    def get_purpose(self) -> str:
        """Returns the purpose of this fragment."""
        return "To execute provided code snippets safely and return the results."

    # <<< ADDED execute method >>>
    async def execute(self, sub_task: str, context: FragmentContext) -> Dict[str, Any]:
        """
        Executes the code specified in the sub_task using the managed 'execute_code' skill.
        Currently assumes the sub_task IS the code, possibly with language hints.
        """
        logger = context.logger # Use logger from context
        logger.info(f"[CodeExecutionManager] Received sub-task: {sub_task[:100]}...")
        tool_name = "execute_code" # Only manages this skill
        
        # --- Basic Code Extraction (Placeholder) ---
        # TODO: Implement more robust extraction if sub_task contains more than just code
        # For now, assume sub_task is the code. Determine language (default python)
        code_to_execute = sub_task
        language = "python" # Default assumption
        # Simple check for shebang or language hints?
        if sub_task.strip().startswith("```python"):
            code_to_execute = sub_task.strip()[9:-3].strip()
            language = "python"
        elif sub_task.strip().startswith("```bash") or sub_task.strip().startswith("```sh"):
            code_to_execute = sub_task.strip()[7:-3].strip()
            language = "bash"
        elif sub_task.strip().startswith("```"):
            # Try to remove generic backticks
             code_to_execute = sub_task.strip()[3:-3].strip()
             # Could try to guess language here if needed
        
        action_input = {
            "code": code_to_execute,
            "language": language
        }
        logger.info(f"Attempting to execute {language} code using '{tool_name}'")

        # --- Tool Execution Logic (similar to FileOpsManager) ---
        try:
            # Retrieve the tool function/callable from the registry
            tool_info = context.tool_registry.get_tool(tool_name)
            
            # Check if tool_info itself is callable and get instance safely
            if not tool_info or not callable(tool_info):
                raise ValueError(f"Skill '{tool_name}' not found or is not callable.")
            
            # <<< CORRECTED: tool_info *is* the callable, instance is separate >>>
            tool_callable = tool_info # The function itself
            # Attempt to get instance if it was registered (might be None)
            instance_maybe, _ = context.tool_registry.get_instance_and_tool(tool_name)
            skill_instance = instance_maybe 
            # <<< END CORRECTION >>>

            logger.info(f"Found tool '{tool_name}'. Instance: {'Yes' if skill_instance else 'No'}.")

            # 1. Retrieve Necessary Context
            llm_interface = context.llm_interface
            workspace_root = context.workspace_root
            tool_registry = context.tools_dict # The ToolRegistry instance

            # 2. Prepare Execution Context for the execute_code tool itself
            #    The execute_code tool needs access to the *other* tools/skills.
            # Ensure llm_url is accessed correctly from llm_interface
            execute_code_context = _ToolExecutionContext(
                logger=self.logger,
                workspace_root=workspace_root,
                llm_url=context.llm_interface.llm_url,  # Correct attribute
                tools_dict=tool_registry,
                llm_interface=llm_interface,
                fragment_registry=context.fragment_registry,
                shared_task_context=context.shared_task_context,
                allowed_skills=None, # execute_code should have access to all tools
                memory_manager=context.memory_manager, # Pass memory manager
                skill_instance=None # execute_code is likely a standalone function
            )

            tool_sig = inspect.signature(tool_callable)
            valid_param_names = set(tool_sig.parameters.keys()) - {'self', 'ctx'}
            filtered_action_input = {
                k: v for k, v in action_input.items() if k in valid_param_names
            }
            logger.debug(f"Filtered action input for {tool_name}: {filtered_action_input}")

            if skill_instance:
                logger.debug(f"Calling method {tool_name} on instance {type(skill_instance)} with context {type(execute_code_context)}")
                execution_result_payload = await tool_callable(skill_instance, execute_code_context, **filtered_action_input)
            else:
                if 'ctx' in tool_sig.parameters:
                    logger.debug(f"Calling standalone function {tool_name} with context {type(execute_code_context)}")
                    execution_result_payload = await tool_callable(execute_code_context, **filtered_action_input)
                else:
                    logger.debug(f"Calling standalone function {tool_name} without context")
                    execution_result_payload = await tool_callable(**filtered_action_input)

            # Process result
            final_status = "unknown"
            final_message = "No message provided"
            tool_data = None
            if isinstance(execution_result_payload, dict):
                final_status = execution_result_payload.get("status", "unknown")
                final_message = execution_result_payload.get("message", execution_result_payload.get("data", {}).get("message", "No message provided"))
                tool_data = execution_result_payload.get("data")
            else:
                logger.warning(f"Tool '{tool_name}' returned non-dict result: {type(execution_result_payload)}")
                final_message = f"Tool returned unexpected type: {type(execution_result_payload)}"
                final_status = "error"

            logger.info(f"Tool '{tool_name}' execution finished with status: {final_status}")

            # Update shared context
            if context and hasattr(context, 'shared_task_context'):
                await context.shared_task_context.update_data("last_execution_result", tool_data or {})
                # Add specific context updates for code execution if needed

            return {"status": final_status, "message": final_message, "data": tool_data}

        except Exception as e:
            logger.exception(f"Unexpected error during tool execution '{tool_name}': {e}")
            error_details = f"Tool: {tool_name}, Input: {action_input}, Error: {e}"
            return {"status": "error", "message": f"Unexpected error during execution: {error_details}"}
    # <<< END ADDED execute method >>>

    # (Comment out or remove old methods if they existed and are replaced)