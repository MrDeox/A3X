import logging
from typing import Dict, Any, List, Optional, Tuple
import json

# Importações Corretas
from ..core.skills import skill
from .manager_fragment import ManagerFragment # Base para o Manager
from a3x.core.tool_executor import ToolExecutor # Para executar a skill escolhida
from a3x.core.llm_interface import LLMInterface # Interface LLM
from a3x.core.agent_parser import parse_llm_response # Usar o parser principal que pode extrair JSON de tool calls
from a3x.core.context import _ToolExecutionContext # Keep this if needed by other parts
# Removidas importações não utilizadas como ReactAgent, AsyncGenerator, etc.

logger = logging.getLogger(__name__)

# Skills gerenciadas por este Manager
FILE_OPS_SKILLS: List[str] = [
    "read_file",
    "write_file",
    "list_directory",
    "append_to_file",
    "delete_path",
]

# Prompt do Sistema para o LLM do Manager (adaptado para formato ReAct simplificado)
# O Manager pensa, decide a ação (skill) e o input.
FILE_OPS_MANAGER_SYSTEM_PROMPT: str = """
You are the FileOpsManager, a specialized agent. Your role is to select and prepare the execution of the single most appropriate file operation skill
to fulfill the given sub-task.

Available Skills:
- read_file: Reads the content of a specified file.
- write_file: Writes content to a specified file.
- list_directory: Lists files and directories within a specified path.
- append_to_file: Appends content to an existing file.
- delete_path: Deletes a specified file or directory.

Analyze the sub-task and respond ONLY in the following simplified ReAct format:
Action: [Exact name of the skill to use, e.g., read_file]
Action Input: [Parameters for the skill in valid JSON format, e.g., {"file_path": "data/users.json"}]

Example Sub-task: "Read the requirements file."
Example Response:
Action: read_file
Action Input: {"file_path": "requirements.txt"}

Example Sub-task: "Create a new file named 'output.log' with 'Process started'."
Example Response:
Action: write_file
Action Input: {"file_path": "output.log", "content": "Process started"}

Example Sub-task: "Show me what's in the 'src' folder."
Example Response:
Action: list_directory
Action Input: {"directory_path": "src"}

Provide ONLY the Action and Action Input lines. Do not include "Thought:".
"""

class FileOpsManager(ManagerFragment):
    """
    Coordinates file operations by selecting and executing the appropriate file skill.
    Uses a simplified ReAct format (Action, Action Input) for LLM interaction.
    """
    FRAGMENT_NAME = "FileOpsManager" # Nome canônico

    def __init__(self, llm_interface: LLMInterface, managed_skills: List[str] = FILE_OPS_SKILLS):
        super().__init__(
            name=self.FRAGMENT_NAME,
            purpose="Coordinates file system operations like reading, writing, listing, appending, and deleting.",
            llm_interface=llm_interface,
            managed_skills=managed_skills
        )
        self._managed_skills = managed_skills
        logger.info(f"{self.name} initialized, managing skills: {self._managed_skills}")
        self.tool_executor = ToolExecutor() # Placeholder: Instantiate directly for now

    async def coordinate_execution(self, sub_task: str, context: _ToolExecutionContext) -> Dict[str, Any]:
        """
        Uses the LLM to choose the correct file skill (Action + Action Input) and executes it.
        """
        log_prefix = f"[{self.name}]"
        logger.info(f"{log_prefix} Coordinating sub-task: '{sub_task}'")

        # 1. Build the prompt for the manager's simplified ReAct
        user_prompt_content = f"Sub-task: \"{sub_task}\"\n\nRespond with the appropriate Action and Action Input."
        full_prompt = f"{FILE_OPS_MANAGER_SYSTEM_PROMPT}\n\nUSER: {user_prompt_content}"
        logger.debug(f"{log_prefix} Manager LLM Prompt:\n------\n{full_prompt}\n------")

        # 2. Call the LLM
        try:
            llm_response_text = await self.llm_interface.completion(prompt=full_prompt)
            if not llm_response_text:
                raise ValueError("LLM returned an empty response.")
            logger.debug(f"{log_prefix} LLM raw response for Action/Input: {llm_response_text}")

            # 3. Parse the LLM response using the general ReAct parser
            # It should extract Action and Action Input, ignoring the missing Thought
            thought, skill_name, parameters = parse_llm_response(llm_response_text, logger)

            # Validate the parsed result
            if skill_name is None:
                 # Try parsing as direct JSON as a fallback for the manager response
                 logger.warning(f"{log_prefix} Failed to parse Action/Input using ReAct parser. Trying direct JSON parse as fallback.")
                 try:
                      parsed_json = json.loads(llm_response_text.strip())
                      if isinstance(parsed_json, dict) and "skill_name" in parsed_json and "parameters" in parsed_json:
                           skill_name = parsed_json.get("skill_name")
                           parameters = parsed_json.get("parameters")
                           logger.info(f"{log_prefix} Successfully parsed direct JSON fallback: Skill={skill_name}")
                      else:
                           raise ValueError(f"Parsed JSON fallback lacks required keys ('skill_name', 'parameters'). Response: {llm_response_text}")
                 except Exception as json_err:
                      logger.error(f"{log_prefix} Direct JSON fallback also failed: {json_err}")
                      raise ValueError(f"Failed to parse LLM response using ReAct or direct JSON. Response: {llm_response_text}")

            if not skill_name or skill_name not in self._managed_skills:
                logger.error(f"{log_prefix} LLM selected skill '{skill_name}' which is not managed by {self.name}. Managed: {self._managed_skills}")
                raise ValueError(f"LLM chose an unmanaged skill: {skill_name}. Should be one of {self._managed_skills}")
            if parameters is None:
                 parameters = {}

            logger.info(f"{log_prefix} LLM selected skill: '{skill_name}' with parameters: {parameters}")

            # 4. Execute the selected skill using tool_executor
            if not hasattr(context, 'tools_dict') or not context.tools_dict:
                 logger.error(f"{log_prefix} Execution context missing 'tools_dict'. Cannot execute skill '{skill_name}'.")
                 raise ValueError("Execution context does not contain the required 'tools_dict'.")

            if skill_name not in context.tools_dict:
                 logger.error(f"{log_prefix} Skill '{skill_name}' selected by LLM not found in the global tools_dict.")
                 raise ValueError(f"Selected skill '{skill_name}' not found in available tools.")

            tool_exec_context = _ToolExecutionContext(
                logger=self._logger,
                workspace_root=PROJECT_ROOT,
                llm_url=None,
                tools_dict=context.tools_dict,
                llm_interface=self.llm_interface,
                fragment_registry=None,
                shared_task_context=None,
                allowed_skills=self._managed_skills,
                skill_instance=None,
                memory_manager=None
            )

            tool_result_dict = await self.tool_executor.execute_tool(
                tool_name=skill_name,
                tool_input=parameters,
                context=tool_exec_context
            )

            # 5. Return the result from the executed skill
            execution_result = tool_result_dict.get("result", {
                "status": "error",
                "action": f"{skill_name}_result_missing",
                "data": {"message": "Tool execution result format was unexpected or missing."}
            })
            logger.info(f"{log_prefix} Skill '{skill_name}' execution completed. Result status: {execution_result.get('status')}")
            return execution_result

        except Exception as e:
            logger.exception(f"{log_prefix} Failed during coordination for sub-task '{sub_task}'.")
            return {
                "status": "error",
                "action": f"{self.name}_coordination_failed",
                "data": {"message": f"Error coordinating file operation: {str(e)}"},
            }

    def get_purpose(self) -> str:
         return self.purpose

# Nota: O registro do fragment/manager deve ocorrer em a3x/fragments/registry.py
# Exemplo (não colocar aqui):
# register_fragment(
#     name="FileOpsManager",
#     fragment_class=FileOpsManager,
#     description="Coordinates file system operations.",
#     category="Management", # Nova categoria para Managers
#     managed_skills=FILE_OPS_SKILLS # Informa quais skills ele gerencia
# ) 