import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple, Type, Callable, Awaitable, NamedTuple, TYPE_CHECKING, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import asyncio
import json
from pathlib import Path

# Re-use the Fragment class from self_optimizer for metrics and state
# If it becomes more complex, it might need its own definition here.
from a3x.core.self_optimizer import FragmentState

# Placeholder for shared components if needed
# from a3x.core.llm_interface import LLMInterface
# from a3x.core.skills import SkillRegistry

from a3x.core.context import SharedTaskContext, FragmentContext 
from a3x.core.tool_registry import ToolRegistry  # Added import for ToolRegistry
from a3x.core.context_accessor import ContextAccessor  # Added import for ContextAccessor

from a3x.core.errors import A3XError

if TYPE_CHECKING:
    from a3x.core.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

# <<< NEW: Define FragmentDef dataclass >>>
@dataclass
class FragmentDef:
    name: str
    fragment_class: Type["BaseFragment"] # Forward reference to BaseFragment
    description: str
    category: str = "Execution" # Default category
    skills: List[str] = field(default_factory=list)
    managed_skills: List[str] = field(default_factory=list)
    prompt_template: Optional[str] = None # Added field for prompt template


class BaseFragment(ABC):
    """
    Classe base abstrata para todos os Fragments especializados no A³X.
    Define a interface comum para execução, atualização de métricas e otimização.
    A lógica dos Fragments é desacoplada para promover modularidade e evolução futura,
    alinhando-se aos princípios de 'Fragmentação Cognitiva' e 'Hierarquia Cognitiva em Pirâmide'.
    Cada Fragment deve encapsular sua própria lógica, interagindo com o sistema apenas por meio
    de interfaces bem definidas e do SharedTaskContext para dados compartilhados.
    """
    FRAGMENT_NAME: str = "BaseFragment"

    def __init__(self, fragment_def: FragmentDef, tool_registry: Optional[ToolRegistry] = None):
        self.metadata = fragment_def # Store the definition as metadata
        self._logger = logging.getLogger(__name__)
        self.state = FragmentState(self.metadata.name, self.metadata.skills, self.metadata.prompt_template)
        self.config = {}
        self.optimizer = self._create_optimizer()
        self._tool_registry = tool_registry or ToolRegistry()
        self._context_accessor = ContextAccessor()
        self._last_chat_index_read: int = 0 # <<< Keep this for non-realtime processing >>>
        # <<< ADDED Lock for internal state protection >>>
        self._internal_state_lock = asyncio.Lock()
        self._logger.info(f"Fragment '{self.state.name}' initialized with {len(self.state.current_skills)} skills.")

    def _create_optimizer(self):
        """Instantiates the optimizer for this fragment."""
        pass # Implementado por subclasses

    @abstractmethod
    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        pass

    def set_context(self, shared_task_context: SharedTaskContext) -> None:
        """
        Sets the SharedTaskContext for this Fragment via the ContextAccessor.
        
        Args:
            shared_task_context: The SharedTaskContext instance to use.
        """
        self._context_accessor.set_context(shared_task_context)
        self._logger.info(f"Context set for Fragment '{self.state.name}' with task ID: {shared_task_context.task_id}")

    async def run_and_optimize(
        self,
        objective: str,
        tools: List[Callable[[str, Dict[str, Any]], Awaitable[str]]] = None,
        max_iterations: int = 5,
        context: Optional[Dict] = None,
        shared_task_context: Optional[SharedTaskContext] = None  # Kept for backward compatibility
    ) -> str:
        """
        Executa o Fragment com o objetivo fornecido, utilizando as ferramentas disponíveis.
        Este método serve como ponto de entrada principal para a execução do Fragment,
        garantindo que a lógica interna seja desacoplada e que a interação com outros componentes
        seja feita por meio de interfaces claras.
        """
        if context is None:
            context = {}
        if shared_task_context and not self._context_accessor._context:
            self.set_context(shared_task_context)
        # Use provided tools if any, otherwise fetch from registry based on skills
        if tools is None:
            tools = []
            for skill in self.state.skills:
                try:
                    tool = self._tool_registry.get_tool(skill)
                    tools.append(tool)
                except KeyError:
                    self._logger.warning(f"Tool for skill '{skill}' not found in registry.")
        self._logger.info(f"Running {self.__class__.__name__} with objective: {objective}")
        result = await self.execute_task(objective, tools, context)
        self._logger.info(f"Completed {self.__class__.__name__} with result: {result}")
        return result

    async def execute_task(
        self,
        objective: str,
        tools: List[Callable[[str, Dict[str, Any]], Awaitable[str]]] = None,
        context: Optional[Dict] = None,
        shared_task_context: Optional[SharedTaskContext] = None  # Kept for backward compatibility
    ) -> str:
        """
        Implementação específica da execução da tarefa pelo Fragment.
        Subclasses devem sobrescrever este método para encapsular sua lógica interna,
        mantendo o desacoplamento do restante do sistema.
        """
        if context is None:
            context = {}
        if shared_task_context and not self._context_accessor._context:
            self.set_context(shared_task_context)
        if tools is None:
            tools = []
            for skill in self.state.skills:
                try:
                    tool = self._tool_registry.get_tool(skill)
                    tools.append(tool)
                except KeyError:
                    self._logger.warning(f"Tool for skill '{skill}' not found in registry.")
        # Default implementation - can be overridden by subclasses
        return await self._default_execute(objective, tools, context)

    async def _default_execute(
        self,
        objective: str,
        tools: List[Callable[[str, Dict[str, Any]], Awaitable[str]]],
        context: Dict
    ) -> str:
        # Simple execution without optimization
        if tools:
            tool = tools[0]  # Use the first tool by default
            input_data = {"objective": objective, "context": context}
            # Add context accessor data if available
            if self._context_accessor._context:
                input_data["shared_task_context"] = self._context_accessor._context
            return await tool(objective, input_data)
        return f"No tools available to execute objective: {objective}"

    def get_name(self) -> str:
        return self.state.name

    def get_skills(self) -> List[str]:
        return self.state.current_skills

    def get_current_prompt(self) -> str:
        return self.state.current_prompt

    def get_status_summary(self) -> str:
         return self.state.get_status_summary()

    def get_description_for_routing(self) -> str:
        """Generates a description string suitable for the routing LLM prompt."""
        purpose = self.get_purpose()
        skills = self.get_skills()
        # Format skills nicely, limit if too many
        skills_str = ", ".join(skills[:10]) # Show max 10 skills
        if len(skills) > 10:
            skills_str += ", ..." # Indicate more exist
        return f"- {self.get_name()}: {purpose} Skills: [{skills_str}]"

    async def run_sub_task(self, sub_task: str, shared_task_context: SharedTaskContext, tool_registry: ToolRegistry, llm_interface: 'LLMInterface') -> Dict:
        """
        Executes an internal ReAct loop for the given sub-task, allowing the Fragment to plan and act autonomously.
        
        Args:
            sub_task: The specific sub-task or objective for this Fragment to handle.
            shared_task_context: The shared context for cross-Fragment data sharing.
            tool_registry: Registry of available tools/skills this Fragment can use.
            llm_interface: Interface to the language model for generating thoughts and actions.
        
        Returns:
            A dictionary containing the result of the sub-task execution.
        """
        self.set_context(shared_task_context)
        self._logger.info(f"Starting internal ReAct loop for sub-task: {sub_task} in Fragment: {self.get_name()}")
        max_iterations = 5
        history = []
        
        for iteration in range(max_iterations):
            self._logger.info(f"Iteration {iteration + 1}/{max_iterations} for sub-task: {sub_task}")
            # Build prompt for LLM with allowed skills
            allowed_tools = [tool_registry.get_tool(skill) for skill in self.get_skills() if skill in tool_registry._tools]
            messages = await self.build_worker_messages(sub_task, history, allowed_tools, shared_task_context)
            
            # Get response from LLM
            try:
                response = await llm_interface.get_response(messages)
                self._logger.info(f"LLM response received for sub-task: {sub_task}")
                thought = response.get('thought', '')
                action = response.get('action', {})
                action_name = action.get('name', 'none')
                action_params = action.get('parameters', {})
                history.append({'role': 'assistant', 'content': response})
                
                if action_name == 'none':
                    self._logger.info(f"No action chosen for sub-task: {sub_task}. Concluding based on thought.")
                    return {'success': True, 'result': thought, 'sub_task': sub_task}
                
                # Execute the chosen action
                self._logger.info(f"Executing action {action_name} for sub_task: {sub_task}")
                try:
                    tool_func = tool_registry.get_tool(action_name)
                    observation = await tool_func(action_name, action_params)
                    self._logger.info(f"Action {action_name} executed with observation: {observation}")
                    history.append({'role': 'observation', 'content': observation})
                    
                    # Check if the action result indicates completion
                    if 'success' in observation and observation['success']:
                        self._logger.info(f"Sub-task {sub_task} completed successfully via action {action_name}.")
                        return {'success': True, 'result': observation, 'sub_task': sub_task}
                except Exception as e:
                    error_msg = f"Error executing action {action_name}: {str(e)}"
                    self._logger.error(error_msg)
                    history.append({'role': 'observation', 'content': error_msg})
            except Exception as e:
                error_msg = f"Error getting LLM response: {str(e)}"
                self._logger.error(error_msg)
                history.append({'role': 'error', 'content': error_msg})
                return {'success': False, 'error': error_msg, 'sub_task': sub_task}
        
        self._logger.warning(f"Max iterations reached for sub-task: {sub_task}. Returning last state.")
        return {'success': False, 'error': 'Max iterations reached', 'sub_task': sub_task, 'history': history}

    async def build_worker_messages(self, objective: str, history: List[Dict], allowed_tools: List[Callable], shared_task_context: SharedTaskContext) -> List[Dict]:
        """
        Builds the message list for the LLM worker prompt, including system instructions and history.
        
        Args:
            objective: The current objective or sub-task.
            history: List of previous interactions (thoughts, actions, observations).
            allowed_tools: List of tool functions this Fragment is allowed to use.
            shared_task_context: Shared context for accessing cross-Fragment data.
        
        Returns:
            A list of message dictionaries for the LLM.
        """
        tool_descriptions = []
        for tool in allowed_tools:
            try:
                desc = getattr(tool, 'description', f"Tool: {tool.__name__}")
                tool_descriptions.append(desc)
            except Exception as e:
                self._logger.warning(f"Could not get description for tool {tool.__name__}: {e}")
                tool_descriptions.append(f"Tool: {tool.__name__} (no description available)")
        
        tools_str = "\n".join(tool_descriptions) if tool_descriptions else "No tools available."
        system_prompt = f"""
        You are a specialized AI Fragment named {self.get_name()} in the A³X system, focused on a specific sub-task.
        Your purpose is: {await self.get_purpose()}
        
        Available tools/skills for your use:
        {tools_str}
        
        Respond in JSON format with 'thought' (your reasoning) and 'action' (the tool to use with parameters).
        If no action is needed, set 'action': {{'name': 'none', 'parameters': {{}}}}.
        Focus on the current objective and use your tools effectively.
        """
        messages = [{'role': 'system', 'content': system_prompt}]
        messages.append({'role': 'user', 'content': f"Current objective: {objective}"})
        messages.extend(history)
        return messages

    async def execute_autonomous_loop(self, objective: str, shared_task_context: SharedTaskContext, tool_registry: ToolRegistry, llm_interface: 'LLMInterface', max_cycles: int = 10) -> Dict:
        """
        Executes an autonomous loop for the Fragment to plan, act, and adapt based on the given objective.
        This loop enables the Fragment to select and combine skills, seek consensus with other Fragments if needed,
        and enter experimentation mode for new ideas, all while respecting isolation and focus.
        
        Args:
            objective: The primary objective or task for this Fragment to achieve.
            shared_task_context: Shared context for cross-Fragment data sharing and communication.
            tool_registry: Registry of available tools/skills this Fragment can use.
            llm_interface: Interface to the language model for generating thoughts and actions.
            max_cycles: Maximum number of autonomous cycles to run before concluding.
        
        Returns:
            A dictionary containing the result of the autonomous execution, including success status and learnings.
        """
        self.set_context(shared_task_context)
        self._logger.info(f"Starting autonomous execution loop for objective: {objective} in Fragment: {self.get_name()}")
        cycle_count = 0
        history = []
        result = {"success": False, "result": None, "objective": objective, "learnings": [], "cycles": 0}
        
        while cycle_count < max_cycles:
            cycle_count += 1
            self._logger.info(f"Cycle {cycle_count}/{max_cycles} for objective: {objective}")
            
            # Step 1: Plan and select skills for the current objective
            sub_task_result = await self.run_sub_task(objective, shared_task_context, tool_registry, llm_interface)
            history.append(sub_task_result)
            
            if sub_task_result.get("success", False):
                self._logger.info(f"Objective {objective} achieved successfully in cycle {cycle_count}.")
                result["success"] = True
                result["result"] = sub_task_result.get("result")
                break
            else:
                error_msg = sub_task_result.get("error", "Unknown error")
                self._logger.warning(f"Cycle {cycle_count} failed with error: {error_msg}")
                
                # Step 2: Fallback to internal conversation if in doubt or need support
                if "max iterations reached" in error_msg.lower() or "uncertainty" in error_msg.lower():
                    conversation_result = await self.initiate_internal_conversation(objective, history, shared_task_context)
                    history.append({"role": "conversation", "content": conversation_result})
                    self._logger.info(f"Internal conversation result for {objective}: {conversation_result.get('summary', 'No summary')}")
                    
                    if conversation_result.get("consensus_reached", False):
                        objective = conversation_result.get("revised_objective", objective)
                        self._logger.info(f"Revised objective after conversation: {objective}")
                
                # Step 3: Enter experimentation mode for new ideas if no progress
                elif cycle_count > max_cycles // 2 and not result["success"]:
                    experiment_result = await self.enter_experimentation_mode(objective, history, shared_task_context, tool_registry)
                    history.append({"role": "experimentation", "content": experiment_result})
                    result["learnings"].append(experiment_result.get("learnings", "No learnings recorded"))
                    self._logger.info(f"Experimentation mode result for {objective}: {experiment_result.get('summary', 'No summary')}")
                    
                    if experiment_result.get("breakthrough", False):
                        objective = experiment_result.get("new_approach", objective)
                        self._logger.info(f"New approach after experimentation: {objective}")
        
        result["cycles"] = cycle_count
        if not result["success"]:
            result["error"] = "Max cycles reached without achieving objective"
            self._logger.warning(f"Max cycles reached for objective: {objective}")
        
        self._logger.info(f"Autonomous loop completed for {objective} with success: {result['success']}")
        return result

    async def initiate_internal_conversation(self, objective: str, history: List[Dict], shared_task_context: SharedTaskContext) -> Dict:
        """
        Initiates a conversation with other active Fragments to seek consensus or support for the current objective.
        
        Args:
            objective: The current objective needing support or consensus.
            history: List of previous interactions for context.
            shared_task_context: Shared context to access other Fragments or communication channels.
        
        Returns:
            A dictionary summarizing the conversation outcome, including whether consensus was reached.
        """
        self._logger.info(f"Initiating internal conversation for objective: {objective} in Fragment: {self.get_name()}")
        # Placeholder for actual conversation logic
        # In a real implementation, this would interact with other Fragments via shared_task_context
        return {
            "success": True,
            "consensus_reached": False,
            "revised_objective": objective,
            "summary": "Conversation logic not fully implemented yet. Continuing with original objective."
        }

    async def enter_experimentation_mode(self, objective: str, history: List[Dict], shared_task_context: SharedTaskContext, tool_registry: ToolRegistry) -> Dict:
        """
        Enters experimentation mode to explore new ideas safely in a sandbox environment.
        
        Args:
            objective: The current objective or idea to explore.
            history: List of previous interactions for context.
            shared_task_context: Shared context for recording learnings.
            tool_registry: Registry to access sandbox tools like SandboxExplorer.
        
        Returns:
            A dictionary summarizing the experimentation outcome, including any breakthroughs or learnings.
        """
        self._logger.info(f"Entering experimentation mode for objective: {objective} in Fragment: {self.get_name()}")
        try:
            sandbox_tool = tool_registry.get_tool("explore_sandbox")
            experiment_result = await sandbox_tool(objective, {"max_attempts": 3, "shared_task_context": shared_task_context})
            return {
                "success": True,
                "breakthrough": False,
                "new_approach": objective,
                "learnings": experiment_result.get("results", "No results recorded"),
                "summary": "Experimentation completed with recorded learnings."
            }
        except KeyError:
            self._logger.error("SandboxExplorer tool not found in registry for experimentation mode.")
            return {
                "success": False,
                "breakthrough": False,
                "new_approach": objective,
                "learnings": "Sandbox tool unavailable.",
                "summary": "Failed to enter experimentation mode due to missing tool."
            }

    async def run_sandbox_mode(self, shared_task_context: SharedTaskContext, tool_registry: ToolRegistry, llm_interface: 'LLMInterface', max_interactions: int = 20) -> Dict:
        """
        Runs the Fragment in Sandbox Mode, allowing free-form natural language conversations with other Fragments
        without a specific objective. This mode is designed to observe the system's capabilities and emergent behaviors.
        
        Args:
            shared_task_context: Shared context for cross-Fragment data sharing and communication.
            tool_registry: Registry of available tools/skills this Fragment can use.
            llm_interface: Interface to the language model for generating thoughts and dialogue.
            max_interactions: Maximum number of interactions or conversation turns before concluding.
        
        Returns:
            A dictionary containing the summary of the sandbox session, including dialogue history and insights.
        """
        self.set_context(shared_task_context)
        self._logger.info(f"Starting Sandbox Mode for Fragment: {self.get_name()}")
        interaction_count = 0
        dialogue_history = []
        insights = []
        my_last_message_index = -1 # Track the last message this fragment saw
        
        # Start with a broad, open-ended prompt to initiate conversation
        initial_prompt = "Explore innovative ideas and share insights with other Fragments."
        self._logger.info(f"Initial sandbox prompt: {initial_prompt}")
        
        while interaction_count < max_interactions:
            interaction_count += 1
            self._logger.info(f"Sandbox Interaction {interaction_count}/{max_interactions} for {self.get_name()}")
            
            # Get new messages from others since last check
            new_messages = await shared_task_context.get_sandbox_messages(since_index=my_last_message_index + 1)
            if new_messages:
                last_sender, last_msg_content = new_messages[-1]
                dialogue_history.append({'role': last_sender, 'content': last_msg_content})
                my_last_message_index = len(await shared_task_context.get_sandbox_messages()) - 1
                self._logger.info(f"{self.get_name()} received {len(new_messages)} new messages.")
            
            # Generate a conversational turn based on history
            messages = await self.build_sandbox_messages(initial_prompt, dialogue_history, list(tool_registry._tools.values()), shared_task_context)
            try:
                # Call the real LLM interface (non-streaming)
                llm_response_content = ""
                async for chunk in llm_interface.call_llm(messages, stream=False):
                    llm_response_content += chunk
                
                # Attempt to parse the JSON response
                try:
                    # Extract JSON part from the potentially larger response string
                    json_start = llm_response_content.find('{')
                    json_end = llm_response_content.rfind('}')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_string = llm_response_content[json_start:json_end+1]
                        response = json.loads(json_string)
                        if not isinstance(response, dict):
                            raise ValueError("Extracted content is not a JSON object.")
                    else:
                         raise ValueError("Could not find valid JSON object boundaries ({...}) in the response.")

                except (json.JSONDecodeError, ValueError) as json_e:
                    self._logger.error(f"Failed to parse LLM JSON response: {json_e}")
                    self._logger.debug(f"Raw LLM Response: {llm_response_content}")
                    # Handle error - maybe add an error message to history and continue/break?
                    error_msg = f"Error parsing LLM response: {json_e}"
                    dialogue_history.append({'role': 'error', 'content': error_msg})
                    # Decide whether to break or try again in the next cycle
                    continue # Try next cycle

                thought = response.get('thought', 'No thought provided')
                dialogue = response.get('dialogue', 'No dialogue provided')
                my_message = {'thought': thought, 'dialogue': dialogue}
                if 'insight' in response:
                    my_message['insight'] = response['insight']
                    insights.append(response['insight'])

                # Check if the response includes an action to be executed
                if 'action' in response and response['action'].get('name') != 'none':
                    action = response['action']
                    action_name = action.get('name')
                    action_params = action.get('parameters', {})
                    self._logger.info(f"Action requested in sandbox mode: {action_name}")
                    try:
                        tool_func = tool_registry.get_tool(action_name)
                        observation = await tool_func(action_name, action_params)
                        self._logger.info(f"Action {action_name} executed with observation: {observation}")
                        my_message['action_result'] = observation
                    except Exception as e:
                        error_msg = f"Error executing action {action_name}: {str(e)}"
                        self._logger.error(error_msg)
                        my_message['action_result'] = {'error': error_msg}

                # Add my message to the shared queue
                await shared_task_context.add_sandbox_message(self.get_name(), my_message)
                dialogue_history.append({'role': 'self', 'content': my_message})
                my_last_message_index = len(await shared_task_context.get_sandbox_messages()) - 1
                self._logger.info(f"Sandbox dialogue from {self.get_name()}: {dialogue}")
                
                # Simulate a short delay or wait for others
                await asyncio.sleep(1) # Add a small delay to allow other fragments to respond

            except Exception as e:
                error_msg = f"Error in sandbox interaction {interaction_count}: {str(e)}"
                self._logger.exception(f"Unexpected error during LLM call or processing:") # Log full traceback
                dialogue_history.append({'role': 'error', 'content': error_msg})
                break
        
        summary = {
            "mode": "sandbox",
            "fragment": self.get_name(),
            "interactions": interaction_count,
            "dialogue_history": dialogue_history,
            "insights": insights,
            "summary": f"Sandbox session completed with {interaction_count} interactions and {len(insights)} insights recorded."
        }
        self._logger.info(f"Sandbox Mode completed for {self.get_name()} with {len(insights)} insights.")
        return summary

    async def build_sandbox_messages(self, prompt: str, history: List[Dict], allowed_tools: List[Callable], shared_task_context: SharedTaskContext) -> List[Dict]:
        """
        Builds the message list for the LLM in Sandbox Mode, focusing on open-ended conversation.
        
        Args:
            prompt: The initial or guiding prompt for the sandbox session.
            history: List of previous dialogue interactions for context.
            allowed_tools: List of tool functions this Fragment is allowed to reference.
            shared_task_context: Shared context for accessing cross-Fragment data.
        
        Returns:
            A list of message dictionaries for the LLM.
        """
        tool_descriptions = []
        # Use the provided allowed_tools list directly
        for tool in allowed_tools:
            try:
                # Attempt to get description from the callable directly if available
                # This assumes tools might have a .description attribute or similar
                desc = getattr(tool, 'description', f"Tool: {tool.__name__}") 
                tool_descriptions.append(desc)
            except Exception as e:
                # Fallback to just the function name if description retrieval fails
                try:
                    name = tool.__name__
                except AttributeError:
                    name = "Unnamed Tool"
                self._logger.warning(f"Could not get description for tool {name}: {e}")
                tool_descriptions.append(f"Tool: {name} (no description available)")
        
        tools_str = "\n".join(tool_descriptions) if tool_descriptions else "No tools available."
        
        # Get the overarching project objective from the context
        project_objective = shared_task_context.initial_objective or "general AI development"
        
        system_prompt = f"""
        You are a specialized AI Fragment named {self.get_name()} in the A³X system, participating in a sandbox session.
        Your purpose is: {await self.get_purpose()}
        The overall project goal you are contributing towards is: {project_objective}
        
        Available tools/skills for reference:
        {tools_str}
        
        Engage in natural language conversation with other Fragments to explore ideas and share insights relevant to the project goal.
        Respond in JSON format with 'thought' (your internal reasoning) and 'dialogue' (your message to others).
        If you have an insight or novel idea related to the project goal, include it as 'insight' in your response.
        Focus on creative, collaborative brainstorming directed towards {project_objective}.
        """
        messages = [{'role': 'system', 'content': system_prompt}]
        # Use the specific sandbox session prompt if needed, or just rely on system prompt guidance
        messages.append({'role': 'user', 'content': f"Sandbox session: {prompt} (Contribute towards the project goal: {project_objective})"})
        messages.extend(history)
        return messages

    # <<< ADDED Chat Helper Methods >>>
    async def post_chat_message(
        self,
        context: FragmentContext, 
        message_type: str, 
        content: Dict, 
        target_fragment: Optional[str] = None
    ):
        """Posts a message to the internal chat log via SharedTaskContext."""
        if not context or not context.shared_task_context:
            self._logger.error(f"[{self.get_name()}] Cannot post chat message: SharedTaskContext not available.")
            return

        # Prepare the content payload, adding target if specified
        message_payload = content.copy() # Avoid modifying original dict
        if target_fragment:
            message_payload['target_fragment'] = target_fragment
            log_target = f" to {target_fragment}" 
        else:
            log_target = " (broadcast)" 
            
        try:
            context.shared_task_context.add_chat_message(
                fragment_name=self.get_name(),
                message_type=message_type,
                message_content=message_payload
            )
            self._logger.info(f"[{self.get_name()}] Posted chat message type '{message_type}'{log_target}. Subject: {content.get('subject', 'N/A')}")
        except Exception as e:
            self._logger.exception(f"[{self.get_name()}] Error posting chat message:")

    async def read_chat_messages(
        self,
        context: FragmentContext,
    ) -> List[Dict[str, Any]]:
        """Reads new messages from the internal chat log since the last read index."""
        if not context or not context.shared_task_context:
            self._logger.error(f"[{self.get_name()}] Cannot read chat messages: SharedTaskContext not available.")
            return []
        
        try:
            new_messages = await context.shared_task_context.get_chat_messages(since_index=self._last_chat_index_read)
            if new_messages:
                self._logger.debug(f"[{self.get_name()}] Read {len(new_messages)} new chat messages.")
                # Update the index to the index of the last message read + 1
                # The index is based on the length before reading
                self._last_chat_index_read += len(new_messages)
            return new_messages
        except Exception as e:
            self._logger.exception(f"[{self.get_name()}] Error reading chat messages:")
            return []

    async def _process_chat_message(self, message: Dict[str, Any], context: FragmentContext):
        """Placeholder method to be overridden by subclasses to handle specific chat messages.
        This method is called by process_incoming_chat if a message is relevant.
        """
        sender = message.get("sender")
        msg_type = message.get("type")
        content = message.get("content")
        self._logger.debug(f"[{self.get_name()}] Received chat message from {sender} (Type: {msg_type}). Content keys: {list(content.keys()) if isinstance(content, dict) else 'N/A'}. Default handler, doing nothing.")
        # Subclasses should implement logic here based on sender, type, content
        pass

    async def process_incoming_chat(self, context: FragmentContext):
        """
        Reads new chat messages and calls _process_chat_message for relevant ones.
        This method is automatically called by the orchestrator before execute.
        """
        new_messages = await self.read_chat_messages(context)
        processed_count = 0
        for message in new_messages:
            content = message.get('content', {})
            target = content.get('target_fragment')
            # Process if broadcast (no target) or targeted at this fragment
            if target is None or target == self.get_name():
                try:
                    await self._process_chat_message(message, context)
                    processed_count += 1
                except Exception as e:
                    self._logger.exception(f"[{self.get_name()}] Error processing chat message ID {message.get('message_id')}:")
        if processed_count > 0:
             self._logger.info(f"[{self.get_name()}] Processed {processed_count} relevant chat messages.")
    # <<< END ADDED Chat Helper Methods >>>

    # <<< ADDED Real-time Chat Handler >>>
    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """
        Handles a chat message dispatched directly by the ChatMonitor.
        This method might be called concurrently with the main execute method.
        Subclasses MUST use locks to protect shared internal state.
        """
        sender = message.get("sender")
        msg_type = message.get("type")
        content = message.get("content")
        subject = content.get('subject', 'N/A') if isinstance(content, dict) else 'N/A'
        
        # Example: Log the reception. Subclasses implement actual logic.
        context.logger.info(f"[{self.get_name()} REALTIME] Received chat message via Monitor from {sender} (Type: {msg_type}, Subject: {subject}).")
        
        # Example of using lock (if modifying shared state):
        # async with self._internal_state_lock:
        #    # Safely modify self.some_internal_list or self.config etc.
        #    pass 

        # Default implementation does nothing further. 
        # Subclasses override this to react to specific messages in real-time.
        pass
    # <<< END ADDED Real-time Chat Handler >>>

# <<< NEW: Manager Fragment Base Class >>>
class ManagerFragment(BaseFragment):
    """
    Base class for Fragments that manage or coordinate a specific set of sub-skills.
    They typically select the appropriate sub-skill based on a sub-task description.
    """
    def __init__(self, fragment_def: FragmentDef, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(fragment_def, tool_registry)
        self.sub_fragments: Dict[str, BaseFragment] = {}
        # Ensure logger is available
        self._logger = logging.getLogger(__name__) 
        self._logger.info(f"ManagerFragment '{self.get_name()}' initialized.")

    # >>> ADD get_managed_skill_schemas method <<<
    def get_managed_skill_schemas(self, tool_registry: ToolRegistry) -> Dict[str, Any]:
        """Retrieves the schemas/descriptions for the skills managed by this fragment."""
        schemas = {}
        if not hasattr(self.metadata, 'managed_skills') or not self.metadata.managed_skills:
            self._logger.warning(f"Manager '{self.get_name()}' has no managed_skills defined in its FragmentDef.")
            return schemas
            
        for skill_name in self.metadata.managed_skills:
            try:
                # Assuming ToolRegistry has a method to get schema/description
                # Let's try getting the full tool details
                tool_details = tool_registry.get_tool_details(skill_name) 
                if tool_details:
                     # Extract relevant info (e.g., description, parameters)
                     # The exact structure depends on ToolRegistry.get_tool_details output
                     schemas[skill_name] = {
                         "description": tool_details.get("description", "No description available."),
                         "parameters": tool_details.get("parameters", {}) # Assuming parameters are stored
                     }
                else:
                     self._logger.warning(f"Details not found for managed skill '{skill_name}' in ToolRegistry.")
            except Exception as e:
                self._logger.error(f"Error retrieving details for managed skill '{skill_name}': {e}")
        
        if not schemas:
             self._logger.warning(f"Could not retrieve schemas for any managed skills of '{self.get_name()}'. Managed: {self.metadata.managed_skills}")
             
        return schemas

    def add_sub_fragment(self, fragment: BaseFragment):
        """Adds a sub-fragment to this manager's control."""
        self.sub_fragments[fragment.get_name()] = fragment
        self._logger.info(f"Added sub-fragment '{fragment.get_name()}' to Manager '{self.get_name()}'.")

    async def coordinate_execution(
        self,
        objective: str,
        tools: List[Callable[[str, Dict[str, Any]], Awaitable[str]]] = None,
        context: Dict = None,
        shared_task_context: Optional[SharedTaskContext] = None  # Kept for backward compatibility
    ) -> str:
        """
        Coordena a execução de sub-fragments para atingir o objetivo.
        Este método encapsula a lógica de delegação, mantendo o desacoplamento
        entre os Fragments e promovendo a modularidade.
        """
        if context is None:
            context = {}
        if shared_task_context and not self._context_accessor._context:
            self.set_context(shared_task_context)
        if tools is None:
            tools = []
            for skill in self.state.skills:
                try:
                    tool = self._tool_registry.get_tool(skill)
                    tools.append(tool)
                except KeyError:
                    self._logger.warning(f"Tool for skill '{skill}' not found in registry.")
        self._logger.info(f"Coordinating execution for objective: {objective} with {len(self._sub_fragments)} sub-fragments")
        results = []
        for fragment in self._sub_fragments:
            self._logger.info(f"Executing sub-fragment: {fragment.__class__.__name__}")
            result = await fragment.run_and_optimize(objective, tools, max_iterations=3, context=context)
            results.append(result)
            self._logger.info(f"Sub-fragment {fragment.__class__.__name__} result: {result}")
        return self.synthesize_results(objective, results, context)

    async def synthesize_results(
        self,
        objective: str,
        results: List[str],
        context: Dict
    ) -> str:
        # Default synthesis - can be overridden
        return f"Results for {objective}: {'; '.join(results)}"

    async def execute_task(
        self,
        objective: str,
        tools: List[Callable[[str, Dict[str, Any]], Awaitable[str]]] = None,
        context: Optional[Dict] = None,
        shared_task_context: Optional[SharedTaskContext] = None  # Kept for backward compatibility
    ) -> str:
        if context is None:
            context = {}
        if shared_task_context and not self._context_accessor._context:
            self.set_context(shared_task_context)
        if tools is None:
            tools = []
            for skill in self.state.skills:
                try:
                    tool = self._tool_registry.get_tool(skill)
                    tools.append(tool)
                except KeyError:
                    self._logger.warning(f"Tool for skill '{skill}' not found in registry.")
        return await self.coordinate_execution(objective, tools, context)

# <<< END NEW >>>

# Make sure existing classes are still defined below if any
# ... rest of the file ... 