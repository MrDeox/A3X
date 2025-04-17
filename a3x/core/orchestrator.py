import logging
import json
import re
import uuid
from typing import Dict, Any, List, AsyncGenerator, Optional, Union, Set, Tuple, Type
import traceback
import time
import inspect  # Add this import at the top of the file
from pathlib import Path # Added import for Path
import asyncio # <<< ADDED import >>>
from dataclasses import dataclass # <<< ADDED DATACLASS IMPORT AGAIN >>>
from a3x.core.memory.memory_manager import MemoryManager # <<< ADDED import >>>
from a3x.core.models import PlanStep # <<< ADDED import >>>

# Package imports (copy necessary ones from agent.py)
from a3x.core.config import (
    MAX_REACT_ITERATIONS, # Maybe define orchestrator-specific limits?
    MAX_HISTORY_TURNS,
)
from a3x.core.skills import get_skill_descriptions # Needed? Maybe just pass descriptions
from a3x.core.db_utils import add_episodic_record
from a3x.core.prompt_builder import (
    build_orchestrator_messages, 
    build_worker_messages,
)
from a3x.core.agent_parser import parse_llm_response
from a3x.core.history_manager import trim_history
# Import the module, not specific names that might cause cycles
# import a3x.core.tool_executor 
from a3x.core.llm_interface import LLMInterface
from a3x.fragments.registry import FragmentRegistry
from a3x.core.context import SharedTaskContext, _ToolExecutionContext, Context
from a3x.fragments.base import FragmentContext, BaseFragment
from a3x.core.utils.param_normalizer import normalize_action_input
from a3x.core.tool_registry import ToolRegistry # Import ToolRegistry
from a3x.core.tool_executor import execute_tool # <<< ENSURE THIS IMPORT IS PRESENT
# Remove unused import causing circular dependency
# from a3x.core.agent import ReactAgent, AgentState, AgentMetadata, AgentOutput
# from .persistence import PersistenceManager # Removed unused import
# from .task_manager import TaskManager # Removed unused import
# from .state_manager import AgentStateManager # Removed unused import
# from .memory import AgentMemory # Removed unused import

# --- Import State Update Hooks ---
from a3x.core.hooks.state_updates import (
    notify_new_task,
    notify_fragment_selection,
    notify_task_completion,
    notify_task_error,
    # notify_task_end, # <<< REMOVED >>>
    # notify_task_planning_complete, # Not used directly here yet
    notify_react_step,
    notify_tool_execution_start,
    notify_tool_execution_end
)
# --- End Hook Imports ---

# Placeholder for now:
def get_skills_for_fragment(fragment_name: str, tool_registry: ToolRegistry) -> List[str]:
    """Determines the list of allowed skills for a given fragment.
       Placeholder logic - replace with actual dynamic logic.
    """
    # Example: Allow all registered tools for now, except maybe internal/meta ones
    all_skills = list(tool_registry.get_tool_names())
    # A more robust approach would involve fragment definitions specifying required/allowed skills
    # or using tags/categories in the tool registry.
    logging.warning(f"Using placeholder get_skills_for_fragment for {fragment_name}. Allowing: {all_skills}")
    # Simple filtering example (adjust as needed)
    # if fragment_name == "PlannerFragment":
    #     return [s for s in all_skills if s in ["plan_task", "final_answer"]] # Only allow planning/answering
    # else: # Default: allow most skills
    #     return [s for s in all_skills if not s.startswith("internal_")]
    return all_skills # Allow all for now during refactor

# <<< ADDED import for chat monitor >>>
from .chat_monitor import chat_monitor_task

# --- Required for direct skill execution in _execute_plan_step --- # CORRECTED IMPORTS
from a3x.core.utils.argument_parser import ArgumentParser # <<< RE-ENABLED IMPORT
# --- End required imports ---

# --- Import Fragment Executor ---
from a3x.core.executors.fragment_executor import FragmentExecutor, FragmentExecutionError # Import error class too

# --- Import Status Constants ---
from a3x.core.constants import (
    STATUS_SUCCESS, STATUS_ERROR, STATUS_MAX_ITERATIONS, STATUS_TIMEOUT,
    REASON_EXECUTOR_CALL_FAILED, REASON_DELEGATION_FAILED, REASON_FRAGMENT_FAILED,
    REASON_MAX_STEPS_REACHED, REASON_ORCHESTRATION_CRITICAL_ERROR, REASON_NO_ALLOWED_SKILLS,
    REASON_UNKNOWN, STATUS_NOT_FOUND
)

class TaskOrchestrator:
    """Handles the step-by-step orchestration of a task using Fragments and Skills."""

    def __init__(
        self,
        llm_interface: LLMInterface,
        fragment_registry: FragmentRegistry,
        tool_registry: ToolRegistry, # Use ToolRegistry
        memory_manager: 'MemoryManager', # <<< ADD memory_manager parameter >>>
        workspace_root: Path,
        agent_logger: logging.Logger,
        # Potentially add agent_id if needed for state saving?
    ):
        self.llm_interface = llm_interface
        self.fragment_registry = fragment_registry
        self.tool_registry = tool_registry # Store ToolRegistry
        self.memory_manager = memory_manager # <<< Store memory_manager >>>
        self.workspace_root = workspace_root
        self.logger = agent_logger # Use the passed logger
        self.fragment_descriptions = self.fragment_registry.get_available_fragments_description()
        if "Error" in self.fragment_descriptions:
             self.logger.error(f"[Orchestrator INIT] Failed to get fragment descriptions: {self.fragment_descriptions}")
             self.fragment_descriptions = "- FinalAnswerProvider: Provides final answer." # Fallback
        # <<< ADDED: Initialize forced step attributes >>>
        self._next_forced_component = None
        self._next_forced_sub_task = None
        # <<< END ADDED >>>

        # --- Instantiate Fragment Executor ---
        self.fragment_executor = FragmentExecutor(
            llm_interface=self.llm_interface,
            tool_registry=self.tool_registry,
            fragment_registry=self.fragment_registry, # Pass registry
            memory_manager=self.memory_manager, # <<< Pass memory_manager >>>
            workspace_root=self.workspace_root,
            # max_iterations can be configured here if needed, defaults to MAX_REACT_ITERATIONS
        )
        # --- End Executor Instantiation ---

    # --- LLM Interaction Helper (Moved to FragmentExecutor) ---
    # async def _process_llm_response(self, messages: List[Dict[str, str]], log_prefix: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    #     """Calls LLM and parses Thought/Action/Input format (ReAct cycle).
    #        Returns (thought, action_name, action_input_dict) or (None, None, None)."""
    #     llm_response_raw = ""
    #     try:
    #         self.logger.debug(f"{log_prefix} Calling llm_interface.call_llm (stream=False expected for ReAct cycle)")
    #         async for chunk in self.llm_interface.call_llm(
    #             messages=messages,
    #             stream=False,
    #             stop=["\nObservation:"] # Standard ReAct stop sequence
    #         ):
    #             llm_response_raw += chunk
    #             
    #         if not llm_response_raw:
    #              self.logger.warning(f"{log_prefix} LLM call returned empty response for ReAct cycle.")
    #              return None, None, None

    #         thought, action_name, action_input_str = parse_llm_response(llm_response_raw, self.logger)
    #         
    #         if not action_name:
    #              # Handle cases where only Final Answer is given
    #              if thought and thought.startswith("Final Answer:"):
    #                   final_answer_content = thought.split("Final Answer:", 1)[1].strip()
    #                   self.logger.info(f"{log_prefix} LLM provided Final Answer directly.")
    #                   return None, "final_answer", {"answer": final_answer_content} # Synthesize action
    #               
    #              self.logger.warning(f"{log_prefix} Could not parse Action from LLM response: {llm_response_raw[:500]}")
    #              # Return thought if available, even without action
    #              return (thought if thought and not thought.startswith("Final Answer:") else None), None, None 
    #         try:
    #              action_input_dict = json.loads(action_input_str) if action_input_str else {}
    #         except json.JSONDecodeError:
    #              self.logger.warning(f"{log_prefix} Failed to parse Action Input JSON: {action_input_str}")
    #              # Return action name even if input fails parsing, let execution handle it
    #              return thought, action_name, None 
    #         return thought, action_name, action_input_dict

    #     except Exception as e:
    #         self.logger.exception(f"{log_prefix} Error processing LLM response for ReAct: {e}")
    #         return None, None, None
            
    # --- Action Execution Helper (Moved to FragmentExecutor) ---
    # async def _execute_action(
    #     self, action_name: str, action_input: Optional[Dict[str, Any]], 
    #     log_prefix: str, shared_task_context: SharedTaskContext,
    #     allowed_skills: List[str] # Allowed skills for this step
    # ) -> Dict[str, Any]:
    #     """Executes the chosen action (skill) and returns the observation."""
    #     if action_input is None:
    #          action_input = {} # Ensure action_input is a dict for normalization
    #          
    #     self.logger.info(
    #         f"{log_prefix} Executing Action: {action_name} with input: {action_input}"
    #     )

    #     if action_name not in self.tool_registry:
    #         # Handle missing skill (maybe invoke learning later?)
    #         self.logger.error(f"{log_prefix} Skill '{action_name}' not found in tool registry.")
    #         return {
    #             "status": "error", "action": f"{action_name}_not_found",
    #             "data": {"message": f"Skill '{action_name}' not found."}
    #         }
    #     try:
    #         normalized_action_input = normalize_action_input(action_name, action_input)
    #         if normalized_action_input != action_input:
    #             self.logger.info(f"{log_prefix} Action input normalized to: {normalized_action_input}")
    #         
    #         # Use self.tool_registry here
    #         if not isinstance(self.tool_registry, ToolRegistry):
    #             self.logger.error(f"{log_prefix} Invalid tool_registry type: {type(self.tool_registry)}")
    #             return {"status": "error", "action": "internal_error", "data": {"message": "Invalid tool registry configuration."}}
    #             
    #         # Create execution context
    #         exec_context = _ToolExecutionContext(
    #             logger=self.logger,
    #             workspace_root=self.workspace_root,
    #             llm_url=self.llm_interface.llm_url if self.llm_interface else None,
    #             tools_dict=self.tool_registry,
    #             llm_interface=self.llm_interface,
    #             fragment_registry=self.fragment_registry,
    #             shared_task_context=shared_task_context,
    #             allowed_skills=allowed_skills, # Pass allowed_skills here
    #             skill_instance=None # execute_tool will fill this in
    #         )
    #         
    #         # <<< ADDED LINE TO PROVIDE TOOL REGISTRY TO SKILL CONTEXT >>>
    #         exec_context.tool_registry = self.tool_registry # Make registry accessible via ctx.tool_registry
    #         
    #         tool_result = await execute_tool(
    #             tool_name=action_name,
    #             action_input=normalized_action_input,
    #             tools_dict=self.tool_registry, # Pass ToolRegistry instance
    #             context=exec_context
    #         )
    #         self.logger.info(
    #             f"{log_prefix} Tool Result Status: {tool_result.get('status', 'N/A')}"
    #         )
    #         return tool_result
    #     except Exception as tool_err:
    #         self.logger.exception(f"{log_prefix} Error executing tool '{action_name}':\"")
    #         return {
    #             "status": "error", "action": f"{action_name}_failed",
    #             "data": {"message": f"Error during tool execution: {tool_err}"},
    #         }

    # --- Orchestration Step Helper (Delegation Logic - Stays Here) ---
    async def _get_next_step_delegation(
        self, objective: str, history: List[Tuple[str, str]], shared_task_context: SharedTaskContext
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Determines the next component (Fragment) and sub-task using an LLM call."""
        log_prefix = "[Orchestrator LLM]"
        self.logger.info(f"{log_prefix} Determining next step delegation for objective: {objective}")

        # <<< ADDED: Initial Keyword-Based Routing >>>
        objective_lower = objective.lower()
        routed_component = None
        sub_task_for_manager = objective # Default sub-task if routed

        # Define keywords indicating complex actions that should bypass simple managers
        complex_action_keywords = ["research", "search", "analyze", "summarise", "summarize", "generate", "plan", "write code", "develop"] # Added more keywords

        # File Operations Keywords
        file_keywords = ["file", "read", "write", "save", "list", "directory", "delete", "append", "create directory", "folder", ".txt", ".json", ".csv", ".md"] # Added extensions
        has_file_keywords = any(keyword in objective_lower for keyword in file_keywords)
        has_complex_keywords = any(keyword in objective_lower for keyword in complex_action_keywords)

        # <<< REFINED Logic: Only route to FileOps if file keywords exist AND complex keywords are ABSENT >>>
        if has_file_keywords and not has_complex_keywords:
             # Double check code execution isn't the primary goal despite file keywords
             code_keywords_check = ["execute", "run code", "python", "bash", "script"]
             if not any(keyword in objective_lower for keyword in code_keywords_check):
                 self.logger.info(f"{log_prefix} [Router] Keyword match found for FileOpsManager (File keywords present, complex/code keywords absent).")
                 routed_component = "FileOpsManager"
                 # FileOpsManager can usually handle the full objective as its sub-task
                 sub_task_for_manager = objective

        # Code Execution Keywords (check only if not already routed to FileOps)
        if not routed_component:
             code_keywords = ["execute", "run code", "python", "bash", "script"]
             # <<< REFINED Logic: Ensure it's primarily about code, not complex tasks also involving code >>>
             if any(keyword in objective_lower for keyword in code_keywords) and not has_complex_keywords:
                 self.logger.info(f"{log_prefix} [Router] Keyword match found for CodeExecutionManager (Code keywords present, complex keywords absent).")
                 routed_component = "CodeExecutionManager"
                 # CodeExecutionManager usually expects the code itself, but pass objective for now
                 sub_task_for_manager = objective

        # If routed by keywords, return directly
        if routed_component:
             self.logger.info(f"{log_prefix} [Router] Directly routing to '{routed_component}' based on keywords.")
             # Return the selected component and the determined sub-task
             return routed_component, sub_task_for_manager, 'keyword'
        # <<< END: Initial Keyword-Based Routing >>>

        # --- If not routed by keywords, proceed with LLM Delegation ---
        self.logger.info(f"{log_prefix} No keyword match found. Using LLM delegation.")
        # 1. Build the Orchestrator Prompt Messages
        # Update fragment descriptions in case they changed (e.g., dynamic loading)
        # --- BEGIN DEBUG LOGGING ---
        all_defs = self.fragment_registry.get_all_definitions()
        self.logger.debug(f"{log_prefix} DEBUG: Available fragment def keys for prompt: {list(all_defs.keys())}")
        # --- END DEBUG LOGGING ---
        self.fragment_descriptions = self.fragment_registry.get_available_fragments_description()
        if "Error" in self.fragment_descriptions:
             self.logger.error(f"{log_prefix} Failed to get valid fragment descriptions for prompt: {self.fragment_descriptions}")
             # Use last known good or fallback? For now, error out.
             return None, None, None
             
        messages = await build_orchestrator_messages(
            objective=objective, 
            history=history, 
            fragment_descriptions=self.fragment_descriptions, 
            shared_task_context=shared_task_context
        )
        if not messages:
            self.logger.error(f"{log_prefix} Failed to build orchestrator prompt messages.")
            return None, None, None

        # 2. Call the LLM
        self.logger.debug(f"{log_prefix} Calling LLM with Orchestrator prompt...")
        llm_response_raw = ""
        try:
            # Await the call to get the async generator, then iterate
            llm_response_generator = self.llm_interface.call_llm(
                messages=messages,
                stream=False, # Expecting a single JSON response for delegation
                temperature=0.2,
                # response_format={"type": "json_object"} # If supported
            )
            async for chunk in llm_response_generator:
                 llm_response_raw += chunk
            
            if not llm_response_raw:
                self.logger.warning(f"{log_prefix} LLM call returned empty response for delegation.")
                return None, None, None
            
            self.logger.debug(f"{log_prefix} LLM Raw Response: {llm_response_raw}")

            # 3. Interpret the JSON Response
            try:
                json_match = re.search(r'{\s*.*\s*}', llm_response_raw, re.DOTALL)
                if not json_match:
                    self.logger.error(f"{log_prefix} Could not find valid JSON object {{...}} in LLM response. Response: {llm_response_raw}")
                    return None, None, None
                json_str = json_match.group(0)
                delegation_decision = json.loads(json_str)
                component_name = delegation_decision.get("component")
                sub_task = delegation_decision.get("sub_task")

                if not component_name or not sub_task:
                    self.logger.warning(f"{log_prefix} LLM response missing 'component' or 'sub_task'. Response: {json_str}")
                    return None, None, None
                
                # Validate component existence (optional but good)
                if component_name not in self.fragment_registry.get_all_definitions():
                    self.logger.warning(f"{log_prefix} LLM suggested unknown component '{component_name}'. Available: {list(self.fragment_registry.get_all_definitions().keys())}")
                    return None, None, None 

                self.logger.info(f"{log_prefix} [LLM Router] Determined next fragment: '{component_name}', sub-task: '{sub_task}'")
                return component_name, sub_task, 'llm'

            except json.JSONDecodeError as e:
                self.logger.error(f"{log_prefix} Failed to parse LLM JSON response: {e}. Response: {llm_response_raw}")
                return None, None, None
            except Exception as e:
                 self.logger.error(f"{log_prefix} [LLM Router] Error during LLM delegation: {e}")
                 return None, None, None

        except TypeError as e:
            # Catch the specific error we saw
            self.logger.error(f"{log_prefix} TypeError calling LLM (check async generator handling?): {e}", exc_info=True)
            return None, None, None
        except Exception as e:
            self.logger.exception(f"{log_prefix} Error calling LLM for delegation: {e}")
            return None, None, None

    # --- Fragment Execution (NOW DELEGATES TO FragmentExecutor) ---
    async def _execute_fragment_task(
        self,
        component_name: str,
        sub_task: str,
        objective: str,
        shared_task_context: SharedTaskContext,
        current_fragment_history: List[Tuple[str, str]] = [] # History specific to this fragment's attempts
    ) -> Dict[str, Any]: # Return the result dict from FragmentExecutor
        """Delegates the execution of a fragment's sub-task to the FragmentExecutor."""
        log_prefix = f"[Orchestrator->Executor|{component_name}]"
        self.logger.info(f"{log_prefix} Delegating sub-task '{sub_task}' to FragmentExecutor.")

        # Determine allowed skills for this fragment
        allowed_skills = self._get_allowed_skills(component_name, self.tool_registry, self.fragment_registry)
        if not allowed_skills:
            self.logger.error(f"{log_prefix} No allowed skills determined for fragment '{component_name}'. Cannot execute.")
            return {"status": STATUS_ERROR, "reason": REASON_NO_ALLOWED_SKILLS, "history": current_fragment_history}

        try:
            # Call the FragmentExecutor's execute method
            execution_result = await self.fragment_executor.execute(
                fragment_name=component_name,
                sub_task_objective=sub_task,
                overall_objective=objective,
                fragment_history=current_fragment_history,
                shared_task_context=shared_task_context,
                allowed_skills=allowed_skills,
                logger=self.logger.getChild(f"FragmentExecutor.{component_name}") # Pass child logger
            )
            self.logger.info(f"{log_prefix} FragmentExecutor finished with status: {execution_result.get('status')}")
            return execution_result

        except Exception as e:
            self.logger.exception(f"{log_prefix} Unexpected error occurred while calling FragmentExecutor:")
            return {
                "status": STATUS_ERROR,
                "reason": REASON_EXECUTOR_CALL_FAILED,
                "details": str(e),
                "history": current_fragment_history
            }

    # --- Learning Cycle Helper (Moved from Agent) ---
    async def _invoke_learning_cycle(self, objective: str, main_history: List, final_status: str, shared_context: SharedTaskContext):
        """Invokes a separate learning/reflection skill if available."""
        log_prefix = "[Learning Cycle Invoker]"
        self.logger.info(f"{log_prefix} Invoking learning cycle skill for objective: {objective[:100]}... Status: {final_status}")
        # Check if a specific learning skill exists using ToolRegistry method
        if "learning_cycle_skill" in self.tool_registry.list_tools(): # CORRECT: Check keys of the dict returned by list_tools
            try:
                learning_input = {
                    "objective": objective,
                    "final_status": final_status,
                    "full_history": main_history, # Pass the full history
                    # Pass necessary context data
                    "context_data": shared_context.get_all_data()
                }
                # Execute learning skill (fire and forget for now?)
                exec_context = _ToolExecutionContext(
                    logger=self.logger,
                    workspace_root=self.workspace_root,
                    llm_url=self.llm_interface.llm_url,
                    tools_dict=self.tool_registry,
                    llm_interface=self.llm_interface,
                    fragment_registry=self.fragment_registry,
                    shared_task_context=shared_context,
                    allowed_skills=list(self.tool_registry.keys()) # Allow all skills for learning?
                )
                await execute_tool(
                    tool_name="learning_cycle_skill", 
                    action_input=learning_input, 
                    tools_dict=self.tool_registry,
                    context=exec_context
                )
                self.logger.info(f"{log_prefix} Learning cycle skill executed.")
            except Exception as learn_err:
                    self.logger.error(f"{log_prefix} Error executing learning cycle skill: {learn_err}")
        # else:
            # self.logger.warning(f"{log_prefix} learning_cycle_skill not found in tools. Skipping.")

    # --- Main Orchestration Method (Logic from Agent.run_task) ---
    async def orchestrate(self, objective: str, max_steps: Optional[int] = None) -> Dict:
        """Orchestrates task execution by delegating to Fragments."""
        log_prefix = "[TaskOrchestrator]"
        self.logger.info(f"{log_prefix} Starting task orchestration: {objective}")

        # --- Initialization ---
        task_id = str(uuid.uuid4())
        shared_task_context = SharedTaskContext(task_id=task_id, initial_objective=objective)
        self.logger.info(f"{log_prefix} Initialized SharedTaskContext with ID: {task_id}")
        notify_new_task(objective)

        # <<< ADDED: Start Chat Monitor Task >>>
        monitor_task = asyncio.create_task(
            chat_monitor_task(
                task_id=task_id,
                shared_task_context=shared_task_context,
                # Pass dependencies needed by the monitor and handler context
                llm_interface=self.llm_interface,
                tool_registry=self.tool_registry,
                fragment_registry=self.fragment_registry,
                memory_manager=self.memory_manager,
                workspace_root=self.workspace_root
            ),
            name=f"ChatMonitor-{task_id[:8]}" # Name the task for debugging
        )
        self.monitor_task = monitor_task # Store the task reference
        self.logger.info(f"{log_prefix} Started Chat Monitor task.")
        # <<< END ADDED >>>

        orchestration_history: List[Tuple[str, str]] = [] # High-level orchestrator decisions
        current_step = 0
        max_orchestration_steps = max_steps or float('inf')
        self.logger.info(f"{log_prefix} Maximum orchestration steps allowed: {max_orchestration_steps}")
        
        final_status = "in_progress" # Initialize status
        final_answer = None # Initialize final answer

        self._next_forced_component = None
        self._next_forced_sub_task = None

        try:
            # --- Main Orchestration Loop ---
            while current_step < max_orchestration_steps:
                current_step += 1
                self.logger.info(f"{log_prefix} Orchestration step {current_step}/{max_orchestration_steps if max_orchestration_steps != float('inf') else 'N/A'}")

                # Check if history needs trimming before adding the new assistant message
                if len(orchestration_history) // 2 > MAX_HISTORY_TURNS:
                    self.logger.debug(f"[Orchestrator] History length ({len(orchestration_history) // 2} turns) exceeds max ({MAX_HISTORY_TURNS}). Trimming.")
                    # Pass the logger to trim_history
                    trimmed_history = trim_history(orchestration_history, MAX_HISTORY_TURNS, self.logger)
                    if len(trimmed_history) < len(orchestration_history):
                        self.logger.info(f"[Orchestrator] History trimmed from {len(orchestration_history)} entries to {len(trimmed_history)}.")
                        orchestration_history = trimmed_history

                # 1. Determine Next Step (Fragment & Sub-task) via Orchestrator LLM
                if self._next_forced_component and self._next_forced_sub_task:
                    component_name = self._next_forced_component
                    sub_task = self._next_forced_sub_task
                    self.logger.info(f"{log_prefix} Using forced next step: {component_name} -> {sub_task}")
                    self._next_forced_component = None
                    self._next_forced_sub_task = None
                else:
                    component_name, sub_task, routing_method = await self._get_next_step_delegation(
                        objective=objective,
                        history=orchestration_history,
                        shared_task_context=shared_task_context
                    )

                if not component_name or not sub_task:
                    self.logger.error("[Orchestrator] Failed to determine next step. Ending task.")
                    final_status = STATUS_ERROR
                    final_reason = REASON_DELEGATION_FAILED
                    final_answer = "Could not determine next fragment/sub-task."
                    await notify_task_error(task_id, final_reason, {"details": final_answer})
                    break # Exit loop

                # Delegate to the chosen component (Fragment or Manager)
                self.logger.info(f"{log_prefix} Delegating to Component: {component_name}, Sub-task: {sub_task}")

                # Notify subscribers about the selected fragment/manager
                try:
                    # Pass only the component name
                    await notify_fragment_selection(component_name)
                except Exception as notify_err:
                    self.logger.error(f"{log_prefix} Error notifying fragment selection: {notify_err}")

                # <<< CORRECTED: Instantiate fragment using the registry >>>
                fragment_instance = self.fragment_registry.get_fragment(component_name)
                if not fragment_instance:
                    final_reason = f"Failed to load or instantiate fragment: {component_name}"
                    self.logger.error(f"{log_prefix} {final_reason}")
                    # Notify error hook if needed
                    # await notify_task_error(shared_task_context.task_id, final_reason, component_name)
                    break # Stop orchestration if fragment cannot be loaded

                # Prepare context for the fragment execution
                fragment_context = FragmentContext(
                    logger=self.logger,
                    llm_interface=self.llm_interface,
                    tool_registry=self.tool_registry,
                    fragment_registry=self.fragment_registry,
                    shared_task_context=shared_task_context,
                    workspace_root=self.workspace_root,
                    memory_manager=self.memory_manager # Pass memory manager
                )

                # Execute the fragment using FragmentExecutor
                self.logger.info(f"{log_prefix} Executing fragment '{component_name}' with sub-task: '{sub_task}'")
                fragment_start_time = time.monotonic()
                fragment_result = None

                current_fragment_history = [] # Start fresh for this delegation

                # --- Execute Fragment ---
                fragment_success = False
                start_time = time.monotonic()

                try:
                    # <<< MODIFIED: Check for direct execution using the flag >>>
                    fragment_instance = self.fragment_registry.get_fragment(component_name) # Get instance
                    if not fragment_instance:
                        raise FragmentExecutionError(f"Fragment instance '{component_name}' not found.", status=STATUS_NOT_FOUND, reason=REASON_FRAGMENT_NOT_FOUND)

                    is_direct_executable = getattr(fragment_instance, 'IS_DIRECT_EXECUTABLE', False)

                    if is_direct_executable:
                        self.logger.info(f"{log_prefix} Executing fragment '{component_name}' directly.")
                        # Determine arguments based on fragment type
                        if component_name == "PlannerFragment":
                            # PlannerFragment needs task_description and available_tools
                            # Get available tools (schema format) - assumed available from orchestrator LLM call context
                            # We need to ensure 'available_tools_schema' is accessible here
                            # Let's assume it was prepared earlier in the loop or function scope
                            # If not, we need to adjust the surrounding code to fetch/prepare it.
                            # For now, assuming 'llm_delegation_result' contains it or it's accessible.
                            # Placeholder: Need to ensure available_tools_schema is correctly sourced.
                            # It's likely needed for the orchestrator LLM call anyway.
                            available_tools_schema = self._prepare_tool_schemas(
                                self._get_allowed_skills(None, self.tool_registry, self.fragment_registry) # Get all skills for planning
                            )
                            if not available_tools_schema:
                                self.logger.warning(f"{log_prefix} Could not determine available tools for PlannerFragment direct execution.")
                                # Handle error case - perhaps raise or log and continue?
                                # Raising seems safer to indicate a setup issue.
                                raise FragmentExecutionError(f"Could not determine available tools for PlannerFragment.", status=STATUS_ERROR, reason=REASON_SETUP_ERROR)

                            fragment_result = await fragment_instance.execute(
                                task_description=sub_task, # Pass sub_task as task_description
                                available_tools=available_tools_schema # Pass the tool schemas
                            )
                        elif component_name == "CodeExecutionManager":
                             # Assuming CodeExecutionManager still expects context and sub_task
                             # Verify this assumption if issues arise.
                             fragment_result = await fragment_instance.execute(
                                 context=fragment_context,
                                 sub_task=sub_task
                             )
                        else:
                            # Handle other potential direct executables if needed, or raise error
                            self.logger.error(f"{log_prefix} Direct execution requested for unhandled fragment '{component_name}'. Falling back or erroring?")
                            # For now, let's treat this as an error
                            raise NotImplementedError(f"Direct execution logic not implemented for fragment: {component_name}")

                        # Determine success based on the direct result
                        if isinstance(fragment_result, dict) and fragment_result.get("status") == STATUS_SUCCESS:
                            fragment_success = True
                        else:
                            fragment_success = False
                            self.logger.error(f"{log_prefix} Fragment '{component_name}' direct execution failed or returned non-success status. Result: {fragment_result}")
                    # <<< CORRECTED INDENTATION for else block >>>
                    else:
                        # <<< Original logic: Use FragmentExecutor for other fragments >>>
                        self.logger.info(f"{log_prefix} Executing fragment '{component_name}' via FragmentExecutor (ReAct).")
                        # Determine allowed skills for this fragment
                        allowed_skills_names = self._get_allowed_skills(component_name, self.tool_registry, self.fragment_registry)
                        if not allowed_skills_names:
                             self.logger.warning(f"{log_prefix} No specific allowed skills determined for fragment '{component_name}'. Executor will use its default behavior (likely allowing all registered tools).")

                        fragment_result_dict = await self.fragment_executor.execute(
                            fragment_name=component_name,
                            sub_task_objective=sub_task,
                            overall_objective=objective,
                            fragment_history=current_fragment_history,
                            shared_task_context=shared_task_context,
                            allowed_skills=allowed_skills_names,
                            logger=fragment_context.logger
                        )
                        # Process executor result
                        if isinstance(fragment_result_dict, dict) and fragment_result_dict.get("status") == STATUS_SUCCESS:
                            fragment_success = True
                            fragment_result = fragment_result_dict
                        else:
                            fragment_success = False
                            fragment_result = fragment_result_dict
                            self.logger.error(f"{log_prefix} FragmentExecutor execution failed for '{component_name}'. Result Status: {fragment_result.get('status', 'N/A')}, Reason: {fragment_result.get('reason', 'N/A')}")
                # <<< CORRECTED INDENTATION for except blocks >>>
                except FragmentExecutionError as fee:
                    self.logger.error(f"{log_prefix} FragmentExecutionError for '{component_name}': {fee.message} (Status: {fee.status}, Reason: {fee.reason})")
                    fragment_success = False
                    fragment_result = {"status": fee.status, "reason": fee.reason, "message": fee.message}
                except Exception as e:
                    self.logger.exception(f"{log_prefix} Unexpected error during fragment execution for '{component_name}':")
                    fragment_success = False
                    fragment_result = {"status": STATUS_ERROR, "reason": REASON_ORCHESTRATION_CRITICAL_ERROR, "message": f"Unexpected orchestrator error: {e}"}

                # --- Process Fragment Result --- (This block and onwards are outside the try...except)
                # <<< Ensure this block starts at the correct indentation level (same as the 'try' above) >>>
                final_answer = None
                plan_generated = False
                request_replan = False
                result_data = None
                status_from_result = STATUS_ERROR # Default status if result is invalid
                reason_from_result = REASON_UNKNOWN # Default reason
                message_from_result = "Invalid or missing fragment result."

                if isinstance(fragment_result, dict):
                    status_from_result = fragment_result.get("status", STATUS_ERROR)
                    reason_from_result = fragment_result.get("reason", REASON_UNKNOWN)
                    message_from_result = fragment_result.get("message", "No message provided.")

                    # Check for final answer (common key expected from executor or potentially direct fragment)
                    final_answer = fragment_result.get("final_answer")
                    # Check for replan request (standardized status)
                    request_replan = status_from_result == "request_replan" # Use constant if available
                    # Extract data payload (might contain plan from planner or history from executor)
                    result_data = fragment_result.get("data")

                    # Specific handling for PlannerFragment direct call result
                    # <<< MODIFIED: Check for IS_DIRECT_EXECUTABLE to be more general >>>
                    if is_direct_executable and component_name == "PlannerFragment":
                        if fragment_success: # Planner call succeeded
                            # Planner's execute should return {"status": STATUS_SUCCESS, "data": {"plan": [...]}}
                            if isinstance(result_data, dict) and result_data.get("plan"):
                                plan_generated = True
                                self.logger.info(f"{log_prefix} PlannerFragment returned a plan.")
                            else:
                                self.logger.warning(f"{log_prefix} PlannerFragment succeeded but result data did not contain a 'plan'. Result: {fragment_result}")
                        # If planner failed, fragment_success is False, handled by main status check below.

                    # <<< MODIFIED: Moved history processing to always occur if data contains history >>>
                    if isinstance(result_data, dict):
                        fragment_history = result_data.get("history", [])
                        if fragment_history:
                            self.logger.info(f"{log_prefix} Received {len(fragment_history)} history entries from {component_name} execution.")
                            for entry_index, entry in enumerate(fragment_history):
                                if isinstance(entry, tuple) and len(entry) == 2:
                                    log_message = f"ReAct Step {entry_index + 1}:\\nThought/Action:\\n{entry[0]}\\nObservation:\\n{entry[1]}"
                                    await shared_task_context.add_history("Orchestrator", log_message)
                                else:
                                     await shared_task_context.add_history("Orchestrator", f"History Entry {entry_index + 1} (raw from {component_name}): {str(entry)[:500]}")

                # --- Update Orchestration History ---
                orchestrator_observation = f"Fragment '{component_name}' executed. Status: {status_from_result}."
                if message_from_result and status_from_result != STATUS_SUCCESS:
                    orchestrator_observation += f" Message: {message_from_result[:150]}{'...' if len(message_from_result)>150 else ''}"
                if plan_generated:
                    orchestrator_observation += " Plan generated."
                if request_replan:
                    orchestrator_observation += " Replanning requested."
                if final_answer:
                    orchestrator_observation += f" Final Answer provided: {final_answer[:100]}{'...' if len(final_answer)>100 else ''}"

                orchestrator_thought = f"Completed execution delegation to '{component_name}' for sub-task '{sub_task}'."
                # Avoid adding duplicate history if executor already added similar entries
                # Append this high-level summary to the orchestrator's own history
                orchestration_history.append((orchestrator_thought, orchestrator_observation))
                self.logger.info(f"{log_prefix} Updated orchestration history: {orchestrator_thought} -> {orchestrator_observation}")
                # Also log to shared context for visibility
                await shared_task_context.add_history("Orchestrator", orchestrator_observation)

                # --- Check for Plan Generation (if Planner was called) ---
                if plan_generated:
                    # Plan was generated, store it and continue loop (will execute plan next)
                    validated_plan = result_data["plan"] # Assumes plan is in data field
                    await shared_task_context.update_data("current_plan", validated_plan)
                    await shared_task_context.update_data("next_plan_step_index", 0)
                    self.logger.info(f"{log_prefix} New plan with {len(validated_plan)} steps stored from PlannerFragment. Continuing orchestration loop.")
                    continue # Go to next iteration to start executing the plan

                # --- Check for Replan Request ---
                if request_replan:
                    self.logger.warning(f"{log_prefix} Fragment '{component_name}' requested replanning. Clearing current plan and continuing loop.")
                    await shared_task_context.update_data("current_plan", None) # Clear existing plan
                    await shared_task_context.update_data("next_plan_step_index", 0)
                    # Optionally add reason to context?
                    await shared_task_context.add_history("Orchestrator", f"Replanning requested by {component_name}. Reason: {message_from_result}")
                    continue # Go to next iteration for LLM to decide next step (likely Planner again)

                # --- Check for Task Completion (Final Answer) ---
                if fragment_success and final_answer:
                    self.logger.info(f"{log_prefix} Fragment {component_name} provided a final answer. Concluding task.")
                    final_status = STATUS_SUCCESS
                    await notify_task_completion(shared_task_context.task_id, final_answer)
                    task_completed_successfully = True # Set flag to exit loop
                    break # Exit orchestration loop

                # --- Check for Fragment Failure ---
                if not fragment_success:
                    self.logger.error(f"{log_prefix} Fragment '{component_name}' failed (Status: {status_from_result}). Stopping task.")
                    final_status = STATUS_ERROR # Mark task as failed
                    # Use reason/message from the fragment result
                    final_reason = f"{REASON_FRAGMENT_FAILED}:{component_name}:{reason_from_result}"
                    final_answer = f"Task failed because fragment '{component_name}' encountered an error: {message_from_result}"
                    await notify_task_error(task_id, final_reason, {"fragment": component_name, "details": message_from_result})
                    break # Exit orchestration loop

            # --- End of Orchestration Loop ---

            # Final status determination if loop finishes without break
            if final_status == "in_progress": # If loop finished due to max_steps
                self.logger.warning(f"{log_prefix} Task reached max steps ({current_step}) without completion.")
                final_status = STATUS_MAX_ITERATIONS # Use constant
                final_reason = REASON_MAX_STEPS_REACHED # Use constant
                final_answer = "Task stopped after reaching the maximum number of steps."
                await notify_task_error(shared_task_context.task_id, final_reason, {"details": final_answer})

        except Exception as e:
            self.logger.exception("[Orchestrator] Critical error during orchestration loop:")
            final_status = STATUS_ERROR
            final_reason = REASON_ORCHESTRATION_CRITICAL_ERROR
            final_answer = f"Orchestration failed due to critical error: {e}"
            # Ensure notification even on critical error
            if 'shared_task_context' in locals():
                await notify_task_error(shared_task_context.task_id, final_reason, str(e))
            # Ensure history is preserved up to the error point
            if 'orchestration_history' not in locals():
                 orchestration_history = [] # Initialize if error happened before loop

        # --- Learning Cycle Invocation (Optional) --- 
        # await self._invoke_learning_cycle(objective, orchestration_history, final_status, shared_task_context)
        
        # --- Chat Monitor Cleanup ---
        self.logger.info(f"{log_prefix} Cleaning up Chat Monitor task...")
        if hasattr(self, 'monitor_task') and self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                self.logger.info(f"{log_prefix} Chat Monitor task cancelled.")
            except Exception as mon_clean_err:
                self.logger.error(f"{log_prefix} Error during chat monitor cleanup: {mon_clean_err}")
            else:
            self.logger.info(f"{log_prefix} Chat Monitor task already done or not started.")
            
        # --- Return final result --- 
            return {
            "status": final_status,
            "final_answer": final_answer,
            "history": orchestration_history, # Return the orchestrator-level history
            "task_id": shared_task_context.task_id if 'shared_task_context' in locals() else None,
            }

    async def shutdown(self):
        # Cancel any potentially running monitor tasks if the orchestrator is shut down externally
        if hasattr(self, 'monitor_task') and self.monitor_task and not self.monitor_task.done(): # Added hasattr check
            self.logger.info("[Orchestrator] Shutdown requested, cancelling active monitor task.")
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                self.logger.info("[Orchestrator] Monitor task successfully cancelled during shutdown.")
            except Exception as e:
                self.logger.error(f"[Orchestrator] Error waiting for monitor task cancellation during shutdown: {e}")
        self.logger.info("[Orchestrator] Shutdown complete.")

    # --- Method for executing structured plan (Removed for brevity, assumed to be refactored later if needed) ---
    # async def _execute_structured_plan_step(self, step_data: PlanStep, shared_task_context: SharedTaskContext) -> Dict[str, Any]:
    #    ...

    def _get_allowed_skills(self, fragment_name: str, tool_registry: ToolRegistry, fragment_registry: FragmentRegistry) -> List[str]:
        """Determines the list of allowed skills for a given fragment."""
        # If a fragment manages skills, only allow those. Otherwise, allow all.
        fragment_def = fragment_registry.get_fragment_definition(fragment_name)
        if fragment_def and fragment_def.managed_skills:
            self.logger.debug(f"Fragment '{fragment_name}' manages specific skills: {fragment_def.managed_skills}")
            # Verify managed skills exist in the tool registry
            allowed_skills = []
            all_registered_tools = tool_registry.list_tools().keys()
            for skill_name in fragment_def.managed_skills:
                if skill_name in all_registered_tools:
                    allowed_skills.append(skill_name)
                else:
                    self.logger.warning(f"Managed skill '{skill_name}' for fragment '{fragment_name}' not found in ToolRegistry.")
            if not allowed_skills:
                self.logger.warning(f"Fragment '{fragment_name}' defined managed skills, but none were found in registry. Allowing all skills as fallback.")
                allowed_skills = list(all_registered_tools)
        else:
            # Allow all registered skills if none are specifically managed or def not found
            allowed_skills = list(tool_registry.list_tools().keys())
            self.logger.debug(f"Fragment '{fragment_name}' does not manage specific skills or def not found. Allowing all {len(allowed_skills)} registered skills.")

        return allowed_skills