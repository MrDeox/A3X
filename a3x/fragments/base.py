import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple, Type, Callable, Awaitable, NamedTuple, TYPE_CHECKING, Union, Coroutine
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import asyncio
import json
from pathlib import Path

# Re-use the Fragment class from self_optimizer for metrics and state
# If it becomes more complex, it might need its own definition here.
from a3x.core.self_optimizer import FragmentState

# Placeholder for shared components if needed


from a3x.core.context import SharedTaskContext, FragmentContext 
from a3x.core.tool_registry import ToolRegistry  # Added import for ToolRegistry
from a3x.core.context_accessor import ContextAccessor  # Added import for ContextAccessor
from a3x.core.execution.fragment_executor import FragmentExecutor  # NOVO IMPORT
from a3x.core.errors import A3XError
from a3x.core.communication.fragment_chat import FragmentChatManager  # NOVO IMPORT
from a3x.core.lifecycle.fragment_lifecycle import FragmentLifecycleManager  # NOVO IMPORT

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
    capabilities: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    version: str = "0.1.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Basic validation
        if not self.name or not self.fragment_class or not self.description:
            raise ValueError("All fields must be provided")
        if not self.skills:
            raise ValueError("Skills must be provided")
        if not self.managed_skills:
            raise ValueError("Managed skills must be provided")
        if not self.prompt_template:
            raise ValueError("Prompt template must be provided")

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

    def __init__(self, ctx: FragmentContext):
        self.ctx = ctx
        self.metadata = ctx.fragment_def
        self._logger = ctx.logger
        self.state = FragmentState(self.metadata.name, self.metadata.skills, self.metadata.prompt_template)
        self.config = ctx.config or {}
        self.optimizer = self._create_optimizer()
        self._tool_registry = ctx.tool_registry
        self._context_accessor = ContextAccessor()
        if ctx.shared_task_context:
            self._context_accessor.set_context(ctx.shared_task_context)
        self._last_chat_index_read: int = 0
        self._internal_state_lock = asyncio.Lock()
        self._logger.info(f"Fragment '{self.state.name}' initialized with {len(self.state.current_skills)} skills.")
        self._message_handler: Optional[Callable[[str, Dict[str, Any], str], Coroutine[Any, Any, None]]] = None
        self._chat_manager = FragmentChatManager(self._logger, self.get_name())
        self._lifecycle_manager = FragmentLifecycleManager(self._logger)

    @property
    def fragment_id(self) -> str:
        """Convenience property to access the fragment name (ID)."""
        return self.metadata.name

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
        Agora delega para o FragmentExecutor.
        """
        fragment_executor = FragmentExecutor(
            fragment_context=self.ctx,
            logger=self._logger,
            llm_interface=getattr(self.ctx, 'llm_interface', None),
            tool_registry=self._tool_registry,
            shared_task_context=getattr(self.ctx, 'shared_task_context', None)
        )
        return await fragment_executor.execute_task(
            objective=objective,
            tools=tools,
            context=context,
            max_iterations=max_iterations
        )

    async def execute_task(
        self,
        objective: str,
        tools: List[Callable[[str, Dict[str, Any]], Awaitable[str]]] = None,
        context: Optional[Dict] = None,
        shared_task_context: Optional[SharedTaskContext] = None  # Kept for backward compatibility
    ) -> str:
        """
        Implementação específica da execução da tarefa pelo Fragment.
        Agora delega para o FragmentExecutor.
        """
        fragment_executor = FragmentExecutor(
            fragment_context=self.ctx,
            logger=self._logger,
            llm_interface=getattr(self.ctx, 'llm_interface', None),
            tool_registry=self._tool_registry,
            shared_task_context=getattr(self.ctx, 'shared_task_context', None)
        )
        return await fragment_executor.execute_task(
            objective=objective,
            tools=tools,
            context=context
        )

    # _default_execute foi extraído para FragmentExecutor

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

    # run_sub_task foi extraído para FragmentExecutor

    def build_worker_messages(self, objective: str, history: List[Dict], allowed_tools: List[Callable], shared_task_context: SharedTaskContext) -> List[Dict]:
        """Wrapper para construir mensagens do worker usando FragmentChatManager."""
        return self._chat_manager.build_worker_messages(
            objective=objective,
            history=history,
            allowed_tools=allowed_tools,
            shared_task_context=shared_task_context
        )


    async def execute_autonomous_loop(self, objective: str, max_cycles: int = 10) -> Dict:
        """
        Executa o loop autônomo do fragmento (pensar, agir, observar) até critério de parada.
        Agora delega para o FragmentExecutor.
        """
        fragment_executor = FragmentExecutor(
            fragment_context=self.ctx,
            logger=self._logger,
            llm_interface=getattr(self.ctx, 'llm_interface', None),
            tool_registry=self._tool_registry,
            shared_task_context=getattr(self.ctx, 'shared_task_context', None)
        )
        return await fragment_executor.execute_autonomous_loop(
            objective=objective,
            max_cycles=max_cycles
        )

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
        message_type: str,
        content: Dict,
        target_fragment: Optional[str] = None
    ):
        """Wrapper para enviar mensagem de chat usando FragmentChatManager."""
        shared_context = self._context_accessor.get_context()
        await self._chat_manager.post_chat_message(
            shared_task_context=shared_context,
            message_type=message_type,
            content=content,
            target_fragment=target_fragment
        )


    async def read_chat_messages(
        self,
        context: FragmentContext,
    ) -> List[Dict[str, Any]]:
        """Wrapper para ler mensagens de chat usando FragmentChatManager."""
        shared_context = context.shared_task_context if context else None
        return await self._chat_manager.read_chat_messages(
            shared_task_context=shared_context,
            last_index=self._last_chat_index_read
        )


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

    # <<< ADDED Real-time Chat Handler >>>
    async def handle_message(self, message: Dict[str, Any]) -> None:
        """Handles an incoming message directed at this fragment.
        
        Subclasses should override this method to process specific message types.
        The base implementation simply logs the received message.
        """
        msg_type = message.get("message_type", "unknown")
        sender = message.get("content", {}).get("sender", "unknown_sender") # Assuming sender is in content
        logger.debug(f"[{self.get_name()}] Received message (Type: '{msg_type}', Sender: '{sender}') - Base handler, doing nothing.")
        # Subclasses implement specific logic here based on msg_type
        # Example:
        # if msg_type == "some_action":
        #    await self.do_some_action(message.get("content"))
    # <<< END ADDED Real-time Chat Handler >>>

    # <<< ADDED Lifecycle Methods >>>
    async def start(self):
        """Delegado: inicia o ciclo de vida do fragmento via FragmentLifecycleManager."""
        await self._lifecycle_manager.start(self.execute, name=f"{self.metadata.name}_execute")

    async def stop(self):
        """Delegado: para o ciclo de vida do fragmento via FragmentLifecycleManager."""
        await self._lifecycle_manager.stop()

    async def execute(self):
        """
        O método principal de execução do fragmento.
        Subclasses devem sobrescrever para implementar o loop ou tarefa principal.
        """
        self._logger.debug(f"[{self.metadata.name}] Base execute() called. No action defined.")
        pass

    # <<< END ADDED Lifecycle Methods >>>

    # <<< ADDED Reflection Methods >>>
    def generate_reflection_a3l(self) -> str:
        """Generates a basic A3L description of the fragment."""
        return f"fragmento '{self.metadata.name}' ({self.__class__.__name__}), {self.metadata.description}"

    # <<< END ADDED Reflection Methods >>>

    # <<< ADDED Context Management Methods >>>
    def set_context_store(self, context_store):
        """Allows injecting a ContextStore instance if needed."""
        # Example: self._context_store = context_store
        logger.warning(f"[{self.metadata.name}] set_context_store called but not implemented.")
        pass

    # <<< END ADDED Context Management Methods >>>

    # <<< ADDED Lifecycle Check Methods >>>
    def is_running(self) -> bool:
        """Delegado: verifica se o ciclo de vida está ativo via FragmentLifecycleManager."""
        return self._lifecycle_manager.is_running()

    # <<< END ADDED Lifecycle Check Methods >>>



# Make sure existing classes are still defined below if any
# ... rest of the file ... 