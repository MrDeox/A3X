# a3x/fragments/manager_fragment.py
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable

# Need to import BaseFragment and FragmentDef from base.py
from .base import BaseFragment, FragmentDef 
from ..core.tool_registry import ToolRegistry
from ..core.context import SharedTaskContext

logger = logging.getLogger(__name__)

# <<< Manager Fragment Base Class >>>
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
        Core logic for Manager Fragments. Selects appropriate sub-fragments or managed skills.
        """
        # Use context accessor if available
        if shared_task_context and not self._context_accessor._context:
            self.set_context(shared_task_context)
            
        self._logger.info(f"Coordinating execution for objective: {objective} with {len(self.sub_fragments)} sub-fragments")
        
        # Access sub_fragments with underscore
        if not self.sub_fragments:
            # If no sub-fragments, try to select a managed skill directly
            return await self.select_and_execute_managed_skill(objective, context, shared_task_context)

        # Logic for selecting and running sub-fragments...
        # (simplified example)
        results = []
        # Access sub_fragments with underscore
        for fragment in self.sub_fragments.values(): # Iterate through values (fragment instances)
            # Maybe filter fragments based on objective?
            # Pass context, not tools directly
            result = await fragment.run_and_optimize(objective, context=context, shared_task_context=shared_task_context)
            results.append(result)

        # Synthesize results from sub-fragments
        return await self.synthesize_results(objective, results, context)

    async def select_and_execute_managed_skill(
        self,
        objective: str,
        context: Dict,
        shared_task_context: Optional[SharedTaskContext] = None  # Kept for backward compatibility
    ) -> str:
        """
        Selects and executes a managed skill directly based on the objective.
        This method is called when no sub-fragments are available.
        """
        # Use context accessor if available
        if shared_task_context and not self._context_accessor._context:
            self.set_context(shared_task_context)
            
        # Example: Select a skill based on the objective
        # This is a placeholder logic and should be replaced with actual skill selection
        selected_skill = self.metadata.managed_skills[0] if self.metadata.managed_skills else None
        if not selected_skill:
            self._logger.warning(f"No managed skills available to execute objective: {objective}")
            return f"No managed skills available to execute objective: {objective}"
        
        # Example: Execute the selected skill
        # This is a placeholder logic and should be replaced with actual skill execution
        try:
            tool = self._tool_registry.get_tool(selected_skill)
            # Pass context to the tool if needed, assuming standard tool signature
            # If tool signature varies, this needs more complex handling
            result = await tool(objective, context) 
            self._logger.info(f"Managed skill '{selected_skill}' executed with result: {result}")
            return result
        except KeyError:
            self._logger.warning(f"Tool for managed skill '{selected_skill}' not found in registry.")
            return f"Tool for managed skill '{selected_skill}' not found in registry."
        except Exception as e:
            self._logger.exception(f"Error executing managed skill '{selected_skill}': {e}")
            return f"Error executing managed skill '{selected_skill}': {e}"

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
        """Overrides BaseFragment.execute_task to use coordinate_execution."""
        if context is None:
            context = {}
        # Ensure context is set before coordinating
        if shared_task_context and not self._context_accessor._context:
            self.set_context(shared_task_context)
            
        # ManagerFragments delegate primary execution logic to coordination
        # We don't necessarily need to resolve 'tools' here unless coordinate_execution needs them explicitly.
        # The current coordinate_execution doesn't seem to use the 'tools' argument directly.
        return await self.coordinate_execution(objective, tools=None, context=context) # Pass None for tools for now 