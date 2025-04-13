import logging
from typing import Dict, Any, Optional

# <<< Import base and decorator >>>
from .base import BaseFragment
from .registry import fragment
from a3x.core.tool_executor import _ToolExecutionContext, execute_tool

logger = logging.getLogger(__name__)

# --- Planner Fragment --- 
@fragment(
    name="PlannerFragment",
    description="Generates a step-by-step plan to achieve an objective.",
    category="Execution",
    skills=["generate_plan"] # Assumes a skill named generate_plan
)
class PlannerFragment(BaseFragment):
    """Fragment responsible for generating plans."""
    async def execute(
        self, 
        ctx: _ToolExecutionContext, 
        objective: str, 
        # Add other planner-specific args if needed
    ) -> Dict[str, Any]:
        log_prefix = "[PlannerFragment]"
        logger.info(f"{log_prefix} Generating plan for objective: {objective}")
        
        planner_input = {"objective": objective}
        # Use execute_tool to call the underlying generate_plan skill
        try:
            result_wrapped = await execute_tool(
                tool_name="generate_plan", # Make sure skill name matches
                action_input=planner_input,
                tools_dict=ctx.tools_dict,
                context=ctx
            )
            logger.info(f"{log_prefix} Plan generation skill completed.")
            return result_wrapped # Return the full wrapped result (result + metrics)
        except Exception as e:
            logger.exception(f"{log_prefix} Error executing generate_plan skill:")
            return {"status": "error", "data": {"message": f"Error generating plan: {e}"}}

# --- Final Answer Fragment --- 
@fragment(
    name="FinalAnswerProvider",
    description="Provides the final answer or summary to the user.",
    category="Execution",
    skills=["final_answer"] # Assumes a skill named final_answer
)
class FinalAnswerProvider(BaseFragment):
    """Fragment responsible for delivering the final answer."""
    async def execute(
        self, 
        ctx: _ToolExecutionContext, 
        answer: str, 
        # Add other final-answer-specific args if needed
    ) -> Dict[str, Any]:
        log_prefix = "[FinalAnswerProvider]"
        logger.info(f"{log_prefix} Providing final answer: {answer[:100]}...")

        final_answer_input = {"answer": answer}
        # Use execute_tool to call the underlying final_answer skill
        try:
            result_wrapped = await execute_tool(
                tool_name="final_answer", # Make sure skill name matches
                action_input=final_answer_input,
                tools_dict=ctx.tools_dict,
                context=ctx
            )
            logger.info(f"{log_prefix} Final answer skill completed.")
            return result_wrapped # Return the full wrapped result (result + metrics)
        except Exception as e:
            logger.exception(f"{log_prefix} Error executing final_answer skill:")
            return {"status": "error", "data": {"message": f"Error providing final answer: {e}"}} 