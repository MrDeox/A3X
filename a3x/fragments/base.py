import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple, Type, Callable, Awaitable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Re-use the Fragment class from self_optimizer for metrics and state
# If it becomes more complex, it might need its own definition here.
from a3x.core.self_optimizer import FragmentState

# Placeholder for shared components if needed
# from a3x.core.llm_interface import LLMInterface
# from a3x.core.skills import SkillRegistry

from a3x.core.context import SharedTaskContext  # Added import for SharedTaskContext
from a3x.core.tool_registry import ToolRegistry  # Added import for ToolRegistry
from a3x.core.context_accessor import ContextAccessor  # Added import for ContextAccessor

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
        self.fragment_def = fragment_def
        self._logger = logging.getLogger(__name__)
        self.state = FragmentState(fragment_def.name, fragment_def.skills, fragment_def.prompt_template)
        self.config = {}
        self.optimizer = self._create_optimizer()
        self._tool_registry = tool_registry or ToolRegistry()
        self._context_accessor = ContextAccessor()
        self._logger.info(f"Fragment '{self.state.name}' initialized with {len(self.state.skills)} skills.")

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
        return self.state.skills

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

# <<< NEW: Manager Fragment Base Class >>>
class ManagerFragment(BaseFragment):
    """
    Base class for Manager Fragments.
    Managers coordinate other Fragments or Skills within a specific domain.
    They receive a sub-task from the Orchestrator and decide how to fulfill it
    by delegating to lower-level components.
    A lógica dos Managers é desacoplada para permitir evolução futura e modularidade,
    alinhando-se aos princípios de 'Fragmentação Cognitiva'.
    """
    def __init__(self, fragment_def: FragmentDef, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(fragment_def, tool_registry)
        self._sub_fragments: List[BaseFragment] = []

    def add_sub_fragment(self, fragment: BaseFragment):
        self._sub_fragments.append(fragment)
        self._logger.info(f"Added sub-fragment {fragment.__class__.__name__} to {self.__class__.__name__}")

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