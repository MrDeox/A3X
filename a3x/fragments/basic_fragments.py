import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable

# <<< Import base and decorator >>>
from .base import BaseFragment, ManagerFragment, FragmentDef, FragmentContext
from .registry import fragment
from a3x.core.context import SharedTaskContext, _ToolExecutionContext
from a3x.core.tool_registry import ToolRegistry
from a3x.core.context_accessor import ContextAccessor
from a3x.core.llm_interface import LLMInterface
from a3x.core.tool_executor import execute_tool
from a3x.core.skills import skill

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
    skills=["hierarchical_planner"]
)
class PlannerFragment(BaseFragment):
    """Fragment responsible for generating plans."""
    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(PLANNER_FRAGMENT_DEF, tool_registry)
        self._internal_replan_request = False

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

    async def execute(self, objective: str, context: FragmentContext) -> Dict[str, Any]:
        """
        Uses the 'hierarchical_planner' skill to generate a plan.
        Returns the plan or an error message.
        """
        logger = context.logger

        # <<< ADDED: Check internal flag for replan request >>>
        if self._internal_replan_request:
            logger.info(f"[{self.get_name()}] Detected internal replan request. Returning status 'request_replan'.")
            self._internal_replan_request = False # Reset flag
            return { "status": "request_replan", "message": "Replanning requested due to prior event (e.g., chat message)." }
        # <<< END ADDED >>>

        logger.info(f"[PlannerFragment] Generating plan for objective: {objective[:100]}...")
        try:
            # Call the hierarchical_planner skill using the execute_tool function
            plan_result_wrapped = await execute_tool(
                tool_name="hierarchical_planner", 
                action_input={"goal": objective}, # Pass objective as 'goal'
                tools_dict=context.tool_registry,    # Use context.tool_registry
                context=context                 # Pass FragmentContext directly
            )

            logger.info("[PlannerFragment] Plan generation skill completed.")
            
            # Extract result and status from the wrapped response
            # execute_tool returns {"result": {...}, "metrics": {...}}
            plan_result = plan_result_wrapped.get("result", {})
            metrics = plan_result_wrapped.get("metrics", {})
            status = metrics.get("status", "error") # Status is in metrics

            if status == "success":
                plan = plan_result.get("data", {}).get("plan")
                if plan is not None:
                    # Use context.shared_task_context to set data
                    await context.shared_task_context.update_data("current_plan", plan)
                    await context.shared_task_context.update_data("current_plan_step", 0)
                    logger.info(f"[PlannerFragment] Stored plan with {len(plan)} steps in shared context.")
                    # Return a standard fragment result structure
                    return {"status": "success", "data": {"plan": plan}, "message": "Plan generated successfully."}
                else:
                    logger.warning("[PlannerFragment] Planner skill succeeded but returned no plan data.")
                    return {"status": "error", "message": "Planner succeeded but returned no plan."}
            else:
                error_message = metrics.get("message", "Unknown planning error from tool executor metrics")
                logger.error(f"[PlannerFragment] Plan generation failed: {error_message}")
                return {"status": "error", "message": f"Plan generation failed: {error_message}"}

        except Exception as e:
            logger.exception(f"[PlannerFragment] Unexpected error during planning: {e}")
            return {"status": "error", "message": f"Unexpected error in PlannerFragment: {e}"}

    # <<< ADDED: Real-time chat handler implementation >>>
    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """
        Handles incoming chat messages dispatched by the ChatMonitor.
        Example implementation for PlannerFragment.
        """
        sender = message.get("sender")
        msg_type = message.get("type")
        content = message.get("content")
        subject = content.get('subject', 'N/A') if isinstance(content, dict) else 'N/A'
        message_id = message.get("message_id")

        log_prefix = f"[{self.get_name()} REALTIME]"
        context.logger.info(f"{log_prefix} Received chat message (ID: {message_id}) from {sender} (Type: {msg_type}, Subject: {subject})")

        # --- Example: Reacting to specific message types --- 
        # Protect access to shared state with the lock
        async with self._internal_state_lock:
            # Example: If another fragment invalidates our current plan
            if msg_type == "PLAN_INVALIDATED":
                invalidating_reason = content.get('reason', 'Unknown')
                context.logger.warning(f"{log_prefix} Plan invalidated by {sender}! Reason: {invalidating_reason}. Flagging for replan.")
                # Set internal flag instead of modifying shared context directly
                self._internal_replan_request = True
                context.logger.info(f"{log_prefix} Set internal replan request flag to True.")
                
            # Example: Reacting to an informational message
            elif msg_type == "INFO" and sender == "FileOpsManager":
                # Maybe update internal context or state based on file operation
                file_op_details = content.get('details')
                context.logger.info(f"{log_prefix} Noted successful FileOp: {file_op_details}")
                # self.state.last_known_file_op = file_op_details # Hypothetical
                pass # Placeholder
                
            # Example: Responding to a direct request for help (if Planner can help)
            elif msg_type == "HELP_REQUEST":
                requested_help = content.get('details')
                context.logger.info(f"{log_prefix} Received help request from {sender}: {requested_help}")
                # Potentially post a response back via chat?
                # await self.post_chat_message(context, "INFO", {"subject": "Re: Help Request", "details": "Analyzing your request..."}, target_fragment=sender)
                pass # Placeholder
            
            else:
                 context.logger.debug(f"{log_prefix} No specific handler for message type '{msg_type}' from {sender}.")

        # Note: Logic here should be relatively quick or spawn background tasks 
        # to avoid blocking the ChatMonitor for too long.

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
        context: FragmentContext,
        answer: str = None,
        sub_task: str = None,
        # Add other final-answer-specific args if needed
    ) -> Dict[str, Any]:
        log_prefix = "[FinalAnswerProvider]"
        # Use sub_task as answer if answer is not provided (compatibilidade com orquestrador)
        if answer is None and sub_task is not None:
            answer = sub_task
        context.logger.info(f"{log_prefix} Providing final answer: {answer[:100]}...")

        final_answer_input = {"answer": answer}

        # <<< ADDED: Create _ToolExecutionContext from FragmentContext >>>
        tool_execution_context = _ToolExecutionContext(
            logger=context.logger,
            workspace_root=context.workspace_root, # Get from FragmentContext
            llm_url=context.llm_interface.llm_url, # Get from LLMInterface
            tools_dict=context.tool_registry,
            llm_interface=context.llm_interface,
            fragment_registry=context.fragment_registry,
            shared_task_context=context.shared_task_context,
            allowed_skills=self.get_skills(), # Use fragment's own skills
            skill_instance=None, # final_answer is standalone
            memory_manager=context.memory_manager # <<< ADDED >>>
        )
        # <<< END ADDED >>>

        # Use execute_tool to call the underlying final_answer skill
        try:
            result_wrapped = await execute_tool(
                tool_name="final_answer", # Make sure skill name matches
                action_input=final_answer_input,
                tools_dict=context.tool_registry,
                context=tool_execution_context # <<< CHANGED: Pass the correct context type >>>
            )
            context.logger.info(f"{log_prefix} Final answer skill completed.")
            
            # Check execute_tool status; assume success if no exception
            tool_status = result_wrapped.get("metrics", {}).get("status", "success")
            
            if tool_status == "success":
                # Return the simplest structure the orchestrator expects on success
                return {
                    "status": "success",
                    "final_answer": answer, # Use the input answer directly
                    "message": f"Final answer provided: {answer[:50]}..."
                    # Metrics are handled by the orchestrator wrapper
                }
            else:
                 # If execute_tool reported an error in its metrics
                 error_message = result_wrapped.get("metrics", {}).get("message", "Unknown error in final_answer skill")
                 context.logger.warning(f"{log_prefix} final_answer skill reported non-success status: {tool_status} - {error_message}")
                 return {"status": "error", "message": error_message}

        except Exception as e:
            context.logger.exception(f"{log_prefix} Error executing final_answer skill:")
            return {"status": "error", "message": f"Error providing final answer: {e}"} 