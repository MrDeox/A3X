import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable

# <<< Import base and decorator >>>
from .base import BaseFragment, ManagerFragment, FragmentDef
from .registry import fragment
from a3x.core.tool_executor import _ToolExecutionContext, execute_tool
from a3x.core.tool_registry import ToolRegistry
from a3x.core.context_accessor import ContextAccessor

logger = logging.getLogger(__name__)

# Define Fragment Definitions for Basic Fragments
PLANNER_FRAGMENT_DEF = FragmentDef(
    name="Planner",
    fragment_class="PlannerFragment",
    description="Plans the approach to solve a given task or problem.",
    category="Planning",
    skills=["plan_task", "break_down_problem"]
)

FINAL_ANSWER_FRAGMENT_DEF = FragmentDef(
    name="FinalAnswerProvider",
    fragment_class="FinalAnswerProvider",
    description="Formats and delivers the final answer to the user's query.",
    category="Execution",
    skills=["format_answer", "summarize_solution"]
)

# --- Planner Fragment --- 
@fragment(
    name="PlannerFragment",
    description="Generates a step-by-step plan to achieve an objective.",
    category="Execution",
    skills=["generate_plan"] # Assumes a skill named generate_plan
)
class PlannerFragment(BaseFragment):
    """Fragment responsible for generating plans."""
    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(PLANNER_FRAGMENT_DEF, tool_registry)

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        return "Break down complex tasks into manageable steps and create a plan."

    async def execute_task(
        self,
        objective: str,
        tools: List[Callable[[str, Dict[str, Any]], Awaitable[str]]] = None,
        context: Optional[Dict] = None
    ) -> str:
        if context is None:
            context = {}
        if tools is None:
            tools = []
            for skill in self.state.skills:
                try:
                    tool = self._tool_registry.get_tool(skill)
                    tools.append(tool)
                except KeyError:
                    self._logger.warning(f"Tool for skill '{skill}' not found in registry.")
        self._logger.info(f"PlannerFragment creating plan for objective: {objective}")
        # Use context accessor to get task objective if available
        task_objective = self._context_accessor.get_task_objective()
        if task_objective:
            self._logger.info(f"Using task objective from context: {task_objective}")
        else:
            task_objective = objective
        # Execute planning logic using tools
        return await self._default_execute(task_objective, tools, context)

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
    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(FINAL_ANSWER_FRAGMENT_DEF, tool_registry)

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        return "Provide the final formatted answer to the user's request."

    async def execute_task(
        self,
        objective: str,
        tools: List[Callable[[str, Dict[str, Any]], Awaitable[str]]] = None,
        context: Optional[Dict] = None
    ) -> str:
        if context is None:
            context = {}
        if tools is None:
            tools = []
            for skill in self.state.skills:
                try:
                    tool = self._tool_registry.get_tool(skill)
                    tools.append(tool)
                except KeyError:
                    self._logger.warning(f"Tool for skill '{skill}' not found in registry.")
        self._logger.info(f"FinalAnswerProvider formatting final answer for: {objective}")
        # Use context accessor to retrieve relevant data if available
        results = self._context_accessor.get_data_by_tag("task_result")
        if results:
            self._logger.info(f"Using task results from context: {list(results.keys())}")
            context.update({"previous_results": results})
        return await self._default_execute(objective, tools, context)

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