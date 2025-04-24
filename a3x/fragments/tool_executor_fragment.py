import logging
import json
from typing import Any, Dict, Optional
from a3x.fragments.base import BaseFragment, FragmentContext, FragmentDef
from a3x.fragments.registry import fragment

logger = logging.getLogger(__name__)

@fragment(
    name="tool_executor",
    description="Executa uma ferramenta com base em um pedido estruturado.",
    category="execution",
    skills=["execute_tool", "validate_request", "handle_errors"]
)
class ToolExecutorFragment(BaseFragment):
    """
    Executes tools or skills based on structured requests.
    """

    def __init__(self, fragment_def: FragmentDef, tool_registry: Optional[Any] = None):
        super().__init__(fragment_def, tool_registry)
        self._logger = logging.getLogger(__name__)

    def get_purpose(self) -> str:
        """Returns the purpose of this fragment."""
        return "Executes tools or skills based on structured requests, handling validation and error management."

    async def execute(self, **kwargs) -> Dict[str, Any]:
        try:
            # Get tool call request from kwargs - handle both tool_call_request and sub_task
            tool_call_request = kwargs.get('tool_call_request')
            sub_task = kwargs.get('sub_task')
            
            if not tool_call_request and not sub_task:
                self._logger.error("No tool call request or sub_task provided")
                return {
                    "status": "error",
                    "error": "No tool call request or sub_task provided"
                }

            # If sub_task is provided, convert it to a tool_call_request
            if sub_task and not tool_call_request:
                try:
                    # First try to parse as JSON
                    tool_call_request = json.loads(sub_task)
                    self._logger.info(f"Parsed sub_task as JSON: {tool_call_request}")
                except json.JSONDecodeError:
                    # If not valid JSON, it's a plain text task.
                    # The ToolExecutorFragment cannot execute plain text tasks directly.
                    # It requires a structured tool_call_request.
                    self._logger.warning(f"Received plain text sub_task \'{sub_task}\', which cannot be executed directly. Requires a structured tool_call_request.")
                    return {
                        "status": "error",
                        "error": "ToolExecutorFragment received a plain text sub_task. It requires a structured tool_call_request (tool_name and parameters).",
                        "details": f"Plain text task received: {sub_task}"
                    }

            # Validate request
            if not tool_call_request: # Ensure we have a request after potential conversion
                self._logger.error("No valid tool call request could be determined.")
                return {"status": "error", "error": "No valid tool call request could be determined."}
                
            if not self._validate_request(tool_call_request):
                self._logger.error("Invalid tool call request")
                return {
                    "status": "error",
                    "error": "Invalid tool call request"
                }

            # Execute tool
            result = await self._execute_tool(tool_call_request)
            
            return {
                "status": "success",
                "result": result
            }
            
        except Exception as e:
            self._logger.error(f"Error in ToolExecutorFragment.execute: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _validate_request(self, request: Dict[str, Any]) -> bool:
        """Validates the tool call request structure."""
        required_fields = ['tool_name', 'parameters']
        return all(field in request for field in required_fields)

    async def _execute_tool(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the requested tool with its parameters."""
        try:
            tool_name = request['tool_name']
            parameters = request['parameters']
            
            # Get tool from registry
            tool = self._tool_registry.get_tool(tool_name)
            if not tool:
                raise ValueError(f"Tool not found: {tool_name}")
            
            # Execute tool
            result = await tool.execute(**parameters)
            return result
            
        except Exception as e:
            self._logger.error(f"Error executing tool {request.get('tool_name')}: {str(e)}")
            raise

# Example usage (optional, can be removed or kept for testing)
# async def main():
#     logging.basicConfig(level=logging.INFO)
#     
#     # Create a mock context
#     class MockContext(FragmentContext):
#         def __init__(self):
#             self.tool_registry = MockToolRegistry()
#     
#     class MockToolRegistry:
#         async def get_tool(self, name: str):
#             if name == "read_file":
#                 return MockTool()
#             return None
#     
#     class MockTool:
#         async def execute(self, **kwargs):
#             return {"status": "success", "content": "File content here..."}
#     
#     # Create fragment with mock context
#     ctx = MockContext()
#     fragment = ToolExecutorFragment(ctx=ctx)
#     
#     # Example tool call request
#     tool_request = {
#         "tool_name": "read_file",
#         "parameters": {"path": "/path/to/file.txt"}
#     }
#     
#     # Execute the fragment
#     result = await fragment.execute(tool_call_request=tool_request)
#     print("Execution result:", result)
# 
# if __name__ == '__main__':
#     import asyncio
#     asyncio.run(main()) 