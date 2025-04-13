import logging
import json
import os
import re
from typing import Dict, Any, List, AsyncGenerator, Optional
from pathlib import Path

# Package imports
from a3x.core.config import (
    MAX_REACT_ITERATIONS,
    MAX_HISTORY_TURNS,
    PROJECT_ROOT,
    MAX_TOKENS_FALLBACK,
    LLAMA_SERVER_MODEL_PATH,
)
from a3x.core.skills import get_skill_descriptions, get_skill_registry
from a3x.core.db_utils import save_agent_state, load_agent_state
from a3x.core.prompt_builder import build_react_prompt
from a3x.core.agent_parser import parse_llm_response
from a3x.core.history_manager import trim_history
from a3x.core.tool_executor import execute_tool, _ToolExecutionContext
from a3x.core.llm_interface import call_llm
from a3x.core.planner import generate_plan
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from a3x.core.utils.param_normalizer import normalize_action_input

# Initialize logger
agent_logger = logging.getLogger(__name__)

# Constante para ID do estado do agente
AGENT_STATE_ID = 1

# --- Constantes Globais ---
AGENT_STATE_FILE = "a3x_agent_state.json"
MAX_TOKENS = MAX_TOKENS_FALLBACK
DEFAULT_MODEL = LLAMA_SERVER_MODEL_PATH
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
MAX_CONSECUTIVE_ERRORS = 3
MAX_STEPS = 100  # Maximum steps per run to prevent infinite loops

DEFAULT_REACT_SYSTEM_PROMPT = """
You are a helpful AI agent designed to achieve user objectives through reasoning and action.

**IMPORTANT: You MUST ALWAYS respond using the strict ReAct format below, for EVERY step, even for simple file or directory operations.**

Strictly follow this format in ALL your responses:

Thought: [Briefly explain your reasoning and plan for the next action.]
Action: [Skill name from the provided list, e.g., write_file]
Action Input: [Parameters for the skill in valid JSON format, e.g., {"file_path": "...", "content": "..."}]

Example:
Thought: I need to write the generated code to a file.
Action: write_file
Action Input: {"file_path": "output/my_script.py", "content": "print('Hello World!')"}

If you have completed the objective, respond ONLY with:
Final Answer: [Your final summary or result]

**Do NOT use any other format. Do NOT output explanations, markdown, or code blocks outside the required fields.**

Available Skills:
{tool_descriptions}

You may also propose and use hypothetical skills that are not in the list above if you believe they would help achieve the objective. If you do so, clearly specify the skill name and its intended function. The system will treat such attempts as opportunities to learn and expand its capabilities.

Previous conversation history:
{history}

User Objective for this step: {input}
"""

# --- Helper for Introspective Query Detection ---
def is_introspective_query(task: str) -> bool:
    """Checks if the task string contains introspective keywords."""
    task_lower = task.lower()
    keywords = [
        r"o que (você|o A³X) aprendeu",
        r"como você lida",
        r"você se lembra",
        r"qual foi sua última reflexão",
        r"memória recente",
        r"reflexão anterior",
        r"o que.*?A³X sabe sobre", # Match "o que A3X sabe sobre X"
        r"como.*?A³X funciona", # Match "como A3X funciona"
        r"o que aconteceu", # Match "o que aconteceu na ultima vez"
    ]
    # Use regex search for more flexible matching
    for pattern in keywords:
        if re.search(pattern, task_lower, re.IGNORECASE):
            agent_logger.info(f"[Introspection Check] Detected introspective query: '{task}' matching pattern: '{pattern}'")
            return True
    return False

# --- Helper for Simple Task Detection ---
def _is_simple_list_files_task(objective: str) -> bool:
    """Checks if the objective is likely a simple request to list files."""
    # Simple keyword check for now, can be improved
    objective_lower = objective.lower().strip()
    keywords = ["liste", "listar", "lista", "mostre", "arquivos", "diretório", "pasta"]
    # Check if it contains list keywords and not complex actions like "read and list" or "delete"
    if (
        any(kw in objective_lower for kw in keywords)
        and "ler" not in objective_lower
        and "conteúdo" not in objective_lower
        and "criar" not in objective_lower
        and "deletar" not in objective_lower
        and "apagar" not in objective_lower
        and "executar" not in objective_lower
    ):
        # Basic check, assumes simple listing if these keywords are present
        # And keywords for other actions are absent.
        # A more robust check might involve simple NLP or regex.
        return True
    return False


# --- Classe ReactAgent ---
class ReactAgent:
    def __init__(self, system_prompt: str, llm_url: Optional[str] = None, tools_dict: Optional[Dict[str, Dict[str, Any]]] = None):
        """Inicializa o Agente ReAct."""
        self.llm_url = llm_url
        self.system_prompt = system_prompt
        self.tools = tools_dict if tools_dict is not None else get_skill_registry()
        self._history = []  # Histórico de Thought, Action, Observation
        self.agent_id = "1"  # TODO: Make agent ID configurable/dynamic
        self.agent_logger = agent_logger
        self._memory: Dict[str, Any] = {}
        self.llm_url = llm_url or os.getenv(
            "LLM_API_URL"
        )  # Use env var if not provided

        # Load agent state if exists
        # <<< FIX: Use string "1" for agent_id >>>
        loaded_state = load_agent_state(agent_id="1")
        if loaded_state:
            self._history = loaded_state.get("history", [])
            self._memory = loaded_state.get("memory", {})
            agent_logger.info(
                f"[ReactAgent INIT] Estado do agente carregado para ID '1'. Memória: {list(self._memory.keys())}"
            )
        self.max_iterations = MAX_REACT_ITERATIONS
        agent_logger.info(
            f"[ReactAgent INIT] Agente inicializado. LLM URL: {'Default' if not self.llm_url else self.llm_url}. Memória carregada: {list(self._memory.keys())}"
        )

        # ADD workspace_root to the agent instance
        project_root_path = Path(PROJECT_ROOT).resolve()
        # <<< ADDED WORKAROUND: Check if path ends with 'a3x' and go up one level >>>
        if str(project_root_path).endswith('/a3x'):
             agent_logger.warning(f"Workspace root from config ended with /a3x ({project_root_path}). Adjusting to parent directory.")
             self.workspace_root = project_root_path.parent
        else:
             self.workspace_root = project_root_path
        # self.workspace_root = Path(PROJECT_ROOT).resolve() # <<< OLD
        agent_logger.info(f"Agent initialized with workspace root: {self.workspace_root}")

    # <<< Method to Call LLM and Parse Response >>>
    async def _process_llm_response(
        self, prompt: List[Dict[str, str]], log_prefix: str
    ) -> Optional[Dict[str, Any]]:
        """Chama o LLM com o prompt, parseia a resposta e retorna um dicionário estruturado ou None em caso de erro fatal."""
        agent_logger.info(f"{log_prefix} Calling LLM...")
        llm_response_raw = ""
        try:
            # Use call_llm (non-streaming for react cycle response)
            # <<< REVERT: Use async for as call_llm likely always returns async generator >>>
            # llm_response_raw = await call_llm(prompt, llm_url=self.llm_url, stream=False) # OLD CORRECTION
            async for chunk in call_llm(prompt, llm_url=self.llm_url, stream=False):
                llm_response_raw += chunk
            agent_logger.info(f"{log_prefix} LLM Response received.")
            agent_logger.debug(f"{log_prefix} Raw LLM Response:\\n{llm_response_raw}")

            # Parse the response
            parsed_output_tuple = parse_llm_response(llm_response_raw, agent_logger)
            if parsed_output_tuple is None:
                agent_logger.error(
                    f"{log_prefix} Failed to parse LLM response (parse_llm_response returned None). Raw: '{llm_response_raw[:100]}...'"
                )
                # Return an error structure instead of None for consistency
                return {
                    "type": "error",
                    "content": "Failed to parse LLM response (parser returned None).",
                }

            thought, action_name, action_input = parsed_output_tuple
            parsed_output = {}
            if thought:
                parsed_output["thought"] = thought
            if action_name:
                parsed_output["action_name"] = action_name
            parsed_output["action_input"] = (
                action_input if action_input is not None else {}
            )
            return parsed_output  # Return structured dictionary

        except json.JSONDecodeError as parse_err:
            agent_logger.error(
                f"{log_prefix} Failed to parse LLM response (JSONDecodeError). Raw: '{llm_response_raw[:100]}...'"
            )
            agent_logger.exception(f"{log_prefix} JSON Parsing Traceback:")
            return {
                "type": "error",
                "content": f"Failed to parse LLM response: {parse_err}",
            }
        except (
            Exception
        ) as llm_err:  # Catch other errors like connection issues during call_llm or general parsing
            agent_logger.exception(f"{log_prefix} Error during LLM call or processing:")
            return {
                "type": "error",
                "content": f"Failed to get or process LLM response: {llm_err}",
            }

    # <<< Method to Execute Action >>>
    async def _execute_action(
        self, action_name: str, action_input: Dict[str, Any], log_prefix: str
    ) -> Dict[str, Any]:
        """Executa a ferramenta especificada com os inputs fornecidos."""
        agent_logger.info(
            f"{log_prefix} Executing Action: {action_name} with input: {action_input}"
        )

        # Fallback elegante para skills inexistentes
        if action_name not in self.tools:
            try:
                from a3x.skills.core.learning_cycle import register_missing_skill_heuristic
                register_missing_skill_heuristic(
                    skill_name=action_name,
                    context={"action_input": action_input, "log_prefix": log_prefix}
                )
            except Exception as reg_err:
                agent_logger.warning(f"{log_prefix} Falha ao registrar heurística de skill ausente: {reg_err}")
            return {
                "status": "missing_skill",
                "action": f"{action_name}_not_found",
                "data": {
                    "message": f"Skill '{action_name}' não foi encontrada. Deseja criar ou aprender essa ferramenta?"
                }
            }
        try:
            # --- Parameter Normalization --- #
            normalized_action_input = normalize_action_input(action_name, action_input)
            if normalized_action_input != action_input:
                agent_logger.info(f"{log_prefix} Action input normalized to: {normalized_action_input}")
            # Use normalized_action_input from here onwards
            # --- End Parameter Normalization ---

            # --- Parameter Mapping/Correction for Specific Skills (Can be removed if normalization handles it) ---
            # if action_name == "generate_code" and "purpose" not in normalized_action_input:
            #     purpose_keys = ["objective", "prompt", "description", "task", "code_description"]
            #     for key in purpose_keys:
            #         if key in normalized_action_input:
            #             normalized_action_input["purpose"] = normalized_action_input.pop(key)
            #             agent_logger.info(f"{log_prefix} Mapped input key '{key}' to 'purpose' for generate_code.")
            #             break
            #     if "purpose" not in normalized_action_input:
            #         agent_logger.warning(f"{log_prefix} 'purpose' parameter missing for generate_code and could not be mapped. Validation might fail.")

            # --- CREATE CONTEXT for execute_tool ---
            exec_context = _ToolExecutionContext(
                logger=agent_logger, # Use the agent's logger
                workspace_root=self.workspace_root, # Use agent's workspace root
                llm_url=self.llm_url, # Pass the agent's llm_url
                tools_dict=self.tools # Pass the agent's tools_dict
            )
            # --- END CONTEXT CREATION ---

            # <<< ADDED LOGGING >>>
            agent_logger.debug(f"{log_prefix} Context passed to execute_tool: workspace_root='{exec_context.workspace_root}'")

            # Execute the tool
            tool_result = await execute_tool(
                tool_name=action_name,
                action_input=normalized_action_input, # Use NORMALIZED action_input
                tools_dict=self.tools,  # Pass self.tools
                context=exec_context # PASS CONTEXT
            )
            agent_logger.info(
                f"{log_prefix} Tool Result Status: {tool_result.get('status', 'N/A')}"
            )
            return tool_result
        except Exception as tool_err:
            agent_logger.exception(
                f"{log_prefix} Error executing tool '{action_name}':\""
            )
            return {
                "status": "error",
                "action": f"{action_name}_failed",
                "data": {"message": f"Error during tool execution: {tool_err}"},
            }

    # <<< Method to Handle Observation >>>
    def _handle_observation(
        self, observation_data: Dict[str, Any], log_prefix: str
    ) -> str:
        """Processa os dados da observação, formata para o histórico e log."""
        try:
            # Format observation for history (compact JSON or string fallback)
            observation_content = json.dumps(observation_data, ensure_ascii=False)
            agent_logger.info(
                f"{log_prefix} Observation: {observation_content[:150]}..."
            )
            self._history.append(f"Observation: {observation_content}")
            return observation_content  # Return the string version for potential reflection
        except TypeError:
            # Fallback if data is not JSON serializable
            observation_content = str(observation_data)
            agent_logger.warning(
                f"{log_prefix} Observation data not JSON serializable. Using str()."
            )
            agent_logger.info(
                f"{log_prefix} Observation (str): {observation_content[:150]}..."
            )
            self._history.append(f"Observation: {observation_content}")
            return observation_content
        except Exception as obs_err:
            agent_logger.exception(f"{log_prefix} Error handling observation:")
            error_content = f"Error processing observation: {obs_err}"
            self._history.append(
                f"Observation: {{'status': 'error', 'message': '{error_content}'}}"
            )
            return error_content

    # <<< Method to Perform ReAct Iteration >>>
    async def _perform_react_iteration(
        self, step_objective: str, log_prefix: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Performs a single Thought-Action-Observation cycle for a given step objective.

        Args:
            step_objective (str): The objective for the current step.
            log_prefix (str): Prefix for logging messages.

        Yields:
            Dict[str, Any]: Dictionaries for thought, action, observation, or error.
        """
        # Trim history before building prompt
        self._history = trim_history(self._history, MAX_HISTORY_TURNS, agent_logger)

        # Build Prompt
        prompt = build_react_prompt(
            objective=step_objective,
            history=self._history,
            system_prompt=self.system_prompt,
            tool_descriptions=get_skill_descriptions(),
            agent_logger=agent_logger,
        )

        # Call LLM and Parse Response
        parsed_output = await self._process_llm_response(prompt, log_prefix)

        # Check for processing errors
        if not parsed_output or parsed_output.get("type") == "error":
            yield parsed_output or {
                "type": "error",
                "content": "Unknown error processing LLM response.",
            }
            # Indicate failure to the caller (e.g., by returning False or raising exception?)
            # For now, yielding error and letting caller handle it.
            return

        # Yield Thought
        if parsed_output.get("thought"):
            thought = parsed_output["thought"]
            self._history.append(f"Thought: {thought}")
            yield {"type": "thought", "content": thought}
        else:
            agent_logger.warning(f"{log_prefix} No 'Thought' found in parsed output.")

        # Handle Action or Final Answer
        action_name = parsed_output.get("action_name")
        action_input = parsed_output.get("action_input", {})

        if action_name == "final_answer":
            final_answer = action_input.get(
                "answer", "No final answer provided."
            )  # Get answer from input dict
            agent_logger.info(
                f"{log_prefix} Final Answer received for step: '{final_answer[:100]}...'"
            )
            self._history.append(f"Final Answer: {final_answer}")
            # Yield a special type indicating the step finished with an answer
            yield {"type": "step_final_answer", "content": final_answer}
            return  # Iteration ends here for this step

        if not action_name:
            agent_logger.error(
                f"{log_prefix} No Action specified by LLM (and not Final Answer). Yielding error."
            )
            yield {"type": "error", "content": "Agent did not specify an action."}
            return

        # Yield Action
        self._history.append(f"Action: {action_name}")
        try:
            action_input_json = json.dumps(action_input, ensure_ascii=False)
            self._history.append(f"Action Input: {action_input_json}")
        except TypeError:
            agent_logger.warning(
                f"{log_prefix} Action input not JSON serializable for history. Using str()."
            )
            self._history.append(f"Action Input: {str(action_input)}")
        yield {"type": "action", "tool_name": action_name, "tool_input": action_input}

        # Execute Action
        observation_data = await self._execute_action(
            action_name, action_input, log_prefix
        )

        # Handle and Yield Observation
        _ = self._handle_observation(observation_data, log_prefix)  # Adds to history
        yield {"type": "observation", "content": observation_data}

        # Check if tool execution failed, potentially yield error
        if observation_data.get("status") == "error":
            yield {
                "type": "error",
                "content": f"Tool execution failed: {observation_data.get('data', {}).get('message', 'Unknown tool error')}",
            }
            # Decide if the loop should stop? For now, let the main loop decide based on error.

    def get_history(self):
        """Retorna o histórico interno para possível inspeção ou passagem para outros componentes."""
        return self._history

    # <<< ADDED Method to add history externally >>>
    def add_history_entry(self, role: str, content: str):
        """Adiciona uma entrada ao histórico interno (usado por CerebrumX para registrar resultados)."""
        # Simple append for now, role might be used for formatting later
        if role.lower() == "human" or role.lower() == "user":
            self._history.append(f"Human: {content}")
        elif role.lower() == "assistant":
            # Format based on likely type (Thought, Observation, etc.)
            if (
                content.startswith("Thought:")
                or content.startswith("Observation:")
                or content.startswith("Final Answer:")
                or content.startswith("Execution result for")
            ):
                self._history.append(content)
            else:  # Generic assistant message
                self._history.append(f"Assistant: {content}")
        else:
            self._history.append(f"{role.capitalize()}: {content}")
        # Optional: Trim history after adding
        # self._history = trim_history(self._history, MAX_HISTORY_TURNS, agent_logger)

    # <<< NEW: Save State Method (Example using file storage) >>>
    def _save_state(self):
        state = {"history": self._history, "memory": self._memory}
        # <<< FIX: Use string "1" for agent_id >>>
        save_agent_state(agent_id="1", state=state)  # Use self.agent_id later
        agent_logger.info("[ReactAgent] Agent state saved for ID '1'.")
