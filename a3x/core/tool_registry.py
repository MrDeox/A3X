from typing import Dict, Callable, Any, Awaitable, List, Tuple, Optional, Union
import logging
import inspect # Added for potential future use in schema generation/validation

logger = logging.getLogger(__name__)

# Define a type alias for clarity
# The callable is the bound method (instance.method) OR a standalone function
ToolFunc = Callable[..., Union[Any, Awaitable[Any]]]
ToolInfo = Tuple[Optional[Any], ToolFunc]  # (instance, tool_function)

# Define a type alias for the structured tool schema dictionary expected by register_tool
# ToolSchema = Dict[str, Any] # Example: {"name": str, "description": str, "parameters": {"type": "object", ...}, "required": ["param1"]}

class ToolRegistry:
    """
    A registry for managing tools (skills) available to Fragments/Agents.
    Stores the instance, the function/method, and schema details for each tool.
    """
    def __init__(self):
        # Stores {tool_name: (instance, tool_function)}
        self._tools: Dict[str, ToolInfo] = {}
        # Stores {tool_name: structured_tool_schema_dict}
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}
        logger.info("ToolRegistry initialized.")

    def register_tool(
        self,
        name: str, # Explicit name for lookup
        instance: Optional[Any], # The skill instance (None for functions)
        tool: ToolFunc, # The function or bound method
        schema: Dict[str, Any] # The structured schema dict {"name": ..., "description": ..., "parameters": ...}
    ) -> None:
        """
        Registers a new tool (function or bound method) associated with its instance and schema.
        
        Args:
            name: The unique name for lookup (should match schema['name']).
            instance: The instance of the class containing the tool method. None for functions.
            tool: The callable representing the tool function or method.
            schema: A dictionary representing the tool's schema, MUST contain
                    at least 'name' and 'description' keys. 'parameters' dict is expected.
        """
        if not callable(tool):
            logger.error(f"Attempted to register non-callable tool for '{name}'. Type: {type(tool)}")
            raise ValueError(f"Tool provided for '{name}' must be callable.")
        
        # <<< MODIFIED VALIDATION: Check schema dict structure >>>
        if not isinstance(schema, dict) or 'name' not in schema or 'description' not in schema:
            logger.error(f"Invalid or incomplete schema dictionary provided for tool '{name}'. Schema: {schema}")
            raise ValueError(f"Schema for tool '{name}' must be a dictionary containing at least 'name' and 'description' keys.")

        # Validate consistency between lookup name and schema name
        schema_name = schema['name']
        if schema_name != name:
             logger.warning(f"Schema name '{schema_name}' inside schema dict does not match registration lookup name '{name}'. Using lookup name '{name}' for registration.")
             # Optionally: raise ValueError("Registration name must match schema name.")

        # Ensure parameters is a dict if present
        if 'parameters' in schema and not isinstance(schema['parameters'], dict):
            logger.error(f"Schema for tool '{name}' contains 'parameters' but it is not a dictionary. Schema: {schema}")
            raise ValueError(f"The 'parameters' key in the schema for tool '{name}' must be a dictionary.")

        # <<< Store the info using the lookup name >>>
        self._tools[name] = (instance, tool)
        self._tool_schemas[name] = schema # Store the whole structured schema dict
        logger.info(f"Tool '{name}' registered (Instance: {type(instance).__name__ if instance else 'Function'}). Schema keys: {list(schema.keys())}")
        # <<< END MODIFIED STORAGE >>>

    def get_tool(self, name: str) -> ToolInfo:
        """
        Retrieves the tuple (instance, tool_function) by its registration name.
        
        Args:
            name: The registration name of the tool.
        
        Returns:
            A tuple (instance, tool_function) if found, otherwise raises KeyError.
        """
        if name not in self._tools:
            logger.error(f"Tool '{name}' not found in registry.")
            raise KeyError(f"Tool '{name}' not found.")
        return self._tools[name] # Return the tuple directly

    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full structured schema dictionary for a tool by its name.
        
        Args:
            name: The registration name of the tool.
        
        Returns:
            The schema dictionary {"name": ..., "description": ..., "parameters": ...} if found, otherwise None.
        """
        schema_info = self._tool_schemas.get(name)
        if not schema_info:
            logger.warning(f"Schema details for tool '{name}' not found in registry.")
        return schema_info

    def list_tools(self) -> List[Dict[str, Any]]: # Return list of schemas
        """
        Lists all registered tools by returning their structured schema dictionaries.
        
        Returns:
            A list of schema dictionaries {"name": ..., "description": ..., "parameters": ...}.
        """
        return list(self._tool_schemas.values())

    # <<< REMOVED get_tools_by_capability - description is in the schema now >>>
    # def get_tools_by_capability(self, capability: str) -> List[ToolFunc]:
    #     ...

    # def get_instance_and_tool(self, name: str) -> ToolInfo:
    #     ...

    # def get_tool_details(self, name: str) -> Optional[ToolSchema]:
    #     ... 