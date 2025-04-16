from typing import Dict, Callable, Any, Awaitable, List, Tuple, Optional
import logging
import inspect # Added for potential future use in schema generation/validation

logger = logging.getLogger(__name__)

# Define a type alias for clarity
# The callable is the bound method (instance.method)
ToolInfo = Tuple[Optional[Any], Callable[..., Awaitable[Any]]]  # (instance, bound_method)

# Define a type alias for the tool schema dictionary
ToolSchema = Dict[str, Any] # Example: {"name": str, "description": str, "parameters": {"type": "object", ...}}

class ToolRegistry:
    """
    A registry for managing tools and skills available to Fragments in the A³X system.
    Stores the instance, the bound method, and the JSON schema for each tool.
    """
    def __init__(self):
        # Stores {tool_name: (instance, bound_method)}
        self._tools: Dict[str, ToolInfo] = {}
        # Stores {tool_name: tool_schema_dict}
        self._tool_schemas: Dict[str, ToolSchema] = {}
        logger.info("ToolRegistry initialized.")

    def register_tool(
        self,
        name: str,
        instance: Optional[Any], # The skill instance
        tool: Callable[..., Awaitable[Any]], # The bound method
        schema: ToolSchema # The JSON schema for the tool
    ) -> None:
        """
        Registers a new tool (bound method) associated with its instance and schema.
        
        Args:
            name: The unique name of the tool (should match schema['name']).
            instance: The instance of the class containing the tool method. Can be None for static/class methods.
            tool: The callable representing the bound tool method (e.g., instance.method).
            schema: A dictionary representing the JSON schema of the tool's parameters and description.
                      Must contain at least 'name' and 'description' keys.
        """
        if not callable(tool):
            logger.error(f"Attempted to register non-callable tool for '{name}'. Type: {type(tool)}")
            raise ValueError(f"Tool provided for '{name}' must be callable.")
        
        if not isinstance(schema, dict) or 'name' not in schema or 'description' not in schema:
            logger.error(f"Invalid or incomplete schema provided for tool '{name}'. Schema: {schema}")
            raise ValueError(f"Schema for tool '{name}' must be a dictionary containing at least 'name' and 'description'.")

        if schema['name'] != name:
             logger.warning(f"Schema name '{schema['name']}' does not match registration name '{name}'. Using registration name.")
             # Optionally enforce schema['name'] == name? For now, allow discrepancy but use registration name.

        self._tools[name] = (instance, tool)
        self._tool_schemas[name] = schema # Store the full schema
        logger.info(f"Tool '{name}' registered (Instance: {type(instance).__name__ if instance else 'None'}). Schema keys: {list(schema.keys())}")

    def get_tool(self, name: str) -> Callable[..., Awaitable[Any]]:
        """
        Retrieves the callable tool (bound method) by its name.
        
        Args:
            name: The name of the tool to retrieve.
        
        Returns:
            The callable bound tool method if found, otherwise raises a KeyError.
        """
        if name not in self._tools:
            logger.error(f"Tool '{name}' not found in registry.")
            raise KeyError(f"Tool '{name}' not found.")
        instance, tool_method = self._tools[name]
        return tool_method

    def get_instance_and_tool(self, name: str) -> ToolInfo:
        """
        Retrieves the instance and the callable tool (bound method) by its name.
        
        Args:
            name: The name of the tool to retrieve.
            
        Returns:
            A tuple (instance, bound_method) if found, otherwise raises a KeyError.
        """
        if name not in self._tools:
            logger.error(f"Tool '{name}' not found in registry.")
            raise KeyError(f"Tool '{name}' not found.")
        # Directly return the stored tuple
        return self._tools[name]

    def get_tool_details(self, name: str) -> Optional[ToolSchema]:
        """
        Retrieves the full schema dictionary for a tool by its name.
        
        Args:
            name: The name of the tool.
        
        Returns:
            The schema dictionary if the tool is found, otherwise None.
        """
        details = self._tool_schemas.get(name)
        if not details:
            logger.warning(f"Details (schema) for tool '{name}' not found in registry.")
        return details

    def get_tools_by_capability(self, capability: str) -> List[Callable[..., Awaitable[Any]]]:
        """
        Retrieves a list of tools (bound methods) that match a given capability based on their descriptions within the schema.
        
        Args:
            capability: The capability or keyword to search for in tool descriptions.
        
        Returns:
            A list of callable bound tool methods that match the capability.
        """
        matching_tools = []
        for name, schema in self._tool_schemas.items():
            description = schema.get("description", "") # Safely get description from schema
            if capability.lower() in description.lower():
                instance, tool_method = self._tools[name]
                matching_tools.append(tool_method)
        logger.info(f"Found {len(matching_tools)} tools matching capability '{capability}'.")
        return matching_tools

    def list_tools(self) -> Dict[str, ToolSchema]:
        """
        Lists all registered tools with their full schemas.
        
        Returns:
            A dictionary mapping tool names to their schema dictionaries.
        """
        return self._tool_schemas.copy() 