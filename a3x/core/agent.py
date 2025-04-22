import logging
import json
import os
import re
from typing import Dict, Any, List, AsyncGenerator, Optional, Tuple, Callable, Union
from pathlib import Path

# Package imports
from a3x.core.config import (
    MAX_REACT_ITERATIONS,
    MAX_HISTORY_TURNS,
    PROJECT_ROOT,
    MAX_TOKENS_FALLBACK,
    LLAMA_SERVER_MODEL_PATH,
    LLAMA_SERVER_URL,
)
from a3x.core.skills import get_skill_descriptions, get_skill_registry
from a3x.core.db_utils import save_agent_state, load_agent_state, add_episodic_record
from a3x.core.prompt_builder import (
    build_orchestrator_messages, 
    build_worker_messages, 
    build_planning_prompt, 
    build_final_answer_prompt
) # Import the new builders
from a3x.core.agent_parser import parse_llm_response
from a3x.core.history_manager import trim_history
from a3x.core.llm_interface import LLMInterface, DEFAULT_LLM_URL
from a3x.core.planner import generate_plan
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from a3x.core.utils.param_normalizer import normalize_action_input
# <<< REMOVED direct import from definitions >>>
# from a3x.fragments.definitions import (
#     AVAILABLE_FRAGMENTS, 
#     FRAGMENT_DESCRIPTIONS, 
#     get_skills_for_fragment, 
#     format_fragment_descriptions_for_prompt
# )
# <<< ADDED import for asyncio (needed for reload_fragments skill) >>>
import asyncio 
from a3x.fragments.registry import FragmentRegistry
# <<< ADDED import for SharedTaskContext AND _ToolExecutionContext from context >>>
from a3x.core.context import SharedTaskContext, _ToolExecutionContext
# <<< ADDED import for uuid >>>
import uuid
# <<< ADDED import for the new orchestrator >>>
from a3x.core.orchestrator import TaskOrchestrator
from a3x.core.tool_registry import ToolRegistry
# Correct import for agent state functions
from a3x.core.db_utils import save_agent_state, load_agent_state
# <<< CORRECTED IMPORT PATH >>>
from a3x.core.memory.memory_manager import MemoryManager

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

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

# <<< REMOVED prompt constants and building methods from ReactAgent >>>

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
    def __init__(
        self,
        agent_id: str,
        llm_interface: Any, # Replace Any with specific LLM Interface type
        skill_registry: Dict[str, Any], # Keep for now if used elsewhere?
        tool_registry: ToolRegistry, # Add ToolRegistry
        fragment_registry: FragmentRegistry,
        memory_manager: MemoryManager, # <<< ADD memory_manager PARAMETER
        workspace_root: Path,
        db_url: str = "sqlite:///agent_state.db",
        max_iterations: int = 10,
        temperature: float = 0.7,
        logger: Optional[logging.Logger] = None,
        agent_name: str = "A3X Agent",
        agent_description: str = "An autonomous agent using the A³X framework.",
        llm_url: Optional[str] = None
    ):
        """Initializes the React Agent container."""
        agent_logger.debug(f"[Agent INIT] Received llm_interface: {llm_interface}")
        # Ensure we have an LLMInterface instance. Create default if None.
        if llm_interface is None:
            effective_llm_url = llm_url or LLAMA_SERVER_URL or DEFAULT_LLM_URL
            self.llm_interface = LLMInterface(llm_url=effective_llm_url)
            agent_logger.info(f"[Agent INIT] No LLMInterface provided, created default instance with URL: {self.llm_interface.llm_url}")
        else:
            self.llm_interface = llm_interface
        
        self.tools = skill_registry if skill_registry is not None else get_skill_registry()
        self._history = []  # Histórico de Thought, Action, Observation
        self.agent_id = agent_id
        self.agent_logger = logger or agent_logger
        self._memory: Dict[str, Any] = {}
        
        # Always use the llm_url from the LLMInterface object
        self.llm_url = self.llm_interface.llm_url # LLMInterface handles URL determination logic internally
        
        self.workspace_root = workspace_root or Path(PROJECT_ROOT).resolve()
        agent_logger.info(f"Agent initialized with workspace root: {self.workspace_root}")
        # self.fragment_registry = FragmentRegistry() # OLD: Creates empty registry
        # NEW: Initialize FragmentRegistry with dependencies needed for discovery/loading
        self.fragment_registry = fragment_registry
        self.tool_registry = tool_registry # Store ToolRegistry
        self.memory_manager = memory_manager # <<< STORE memory_manager INSTANCE

        # >>> ADDED: Instantiate TaskOrchestrator <<<
        self.orchestrator = TaskOrchestrator(
            fragment_registry=self.fragment_registry,
            tool_registry=self.tool_registry, # NEW: Pass ToolRegistry instance
            memory_manager=self.memory_manager, # <<< ADDED: Pass memory_manager >>>
            llm_interface=self.llm_interface, # <<< ADDED: Pass LLM interface >>>
            workspace_root=self.workspace_root,
            agent_logger=self.agent_logger,
            # Add other necessary dependencies if orchestrator needs them
        )

        # Load agent state if exists
        loaded_state = load_agent_state(agent_id=self.agent_id)
        if loaded_state:
            self._history = loaded_state.get("history", [])
            self._memory = loaded_state.get("memory", {})
            agent_logger.info(
                f"[Agent INIT] Estado do agente carregado para ID '{self.agent_id}'. Memória: {list(self._memory.keys())}"
            )
        agent_logger.info(
            f"[Agent INIT] Agent initialized. LLM URL: {'Default' if not self.llm_url else self.llm_url}. Memória carregada: {list(self._memory.keys())}"
        )

        # Internal state
        self.last_run_context: Optional[SharedTaskContext] = None

    # <<< Method to Handle Observation >>>
    def _handle_observation(
        self, observation_data: Dict[str, Any], log_prefix: str
    ) -> str:
        """Formats the observation data into a string for the history."""
        # <<< ADDED DEBUG LOG (Start) >>>
        agent_logger.debug(f"{log_prefix} _handle_observation received raw data: {observation_data}")
        # <<< END ADDED DEBUG LOG >>>
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
        # <<< ADDED DEBUG LOG (End) >>>
        agent_logger.debug(f"{log_prefix} _handle_observation adding to history: 'Observation: {observation_content}'")
        # <<< END ADDED DEBUG LOG >>>

    def get_history(self):
        """Retorna o histórico interno para possível inspeção ou passagem para outros componentes."""
        return self._history

    # <<< ADDED Method to add history externally >>>
    def add_history_entry(self, role: str, content: str):
        """Adiciona uma entrada de dicionário {"role": ..., "content": ...} ao histórico."""
        role_lower = role.lower()
        if role_lower not in ["user", "assistant", "system", "tool", "human"]: # Add "human" for compatibility?
             agent_logger.warning(f"[History] Received unexpected role: '{role}'. Using 'assistant'.")
             role_lower = "assistant"
             
        # Ensure role is either user or assistant if it was human
        if role_lower == "human":
            role_lower = "user" 
            
        history_entry = {"role": role_lower, "content": content}
        self._history.append(history_entry)
        agent_logger.debug(f"[History] Added entry: {history_entry}")
        # Optional: Trim history after adding
        # self._history = trim_history(self._history, MAX_HISTORY_TURNS, agent_logger)

    # <<< NEW: Save State Method (Example using file storage) >>>
    def _save_state(self):
        state = {"history": self._history, "memory": self._memory}
        save_agent_state(agent_id=self.agent_id, state=state)
        agent_logger.info("[ReactAgent] Agent state saved for ID '{}'.".format(self.agent_id))

    # Simplificar o prompt do planejador
    def _build_planner_prompt(self, objective: str, relevant_skills_only: bool = True) -> str:
        """
        Constrói o prompt para o planejador com base no objetivo do usuário e nas skills disponíveis.
        Se relevant_skills_only for True, filtra as skills para incluir apenas aquelas relevantes ao objetivo.
        """
        base_prompt = (
            "Você é um assistente de planejamento de IA. Seu objetivo é decompor um objetivo do usuário em uma lista de passos acionáveis. "
            "Cada passo deve ser claro, conciso e utilizar as ferramentas disponíveis. "
            "O output deve ser uma lista JSON de strings, onde cada string é um passo lógico e sequencial. "
            "O último passo deve quase sempre usar a ferramenta 'final_answer', a menos que o objetivo indique o contrário. "
            "Se o objetivo for impossível ou pouco claro com as ferramentas disponíveis, retorne uma lista JSON vazia `[]`.\n\n"
        )

        base_prompt += f"Objetivo do usuário: \"{objective}\"\n\n"

        base_prompt += "Ferramentas disponíveis:\n"
        if relevant_skills_only:
            # Filtra skills relevantes para o objetivo (ex: relacionadas a 'file' ou 'json')
            relevant_skills = [
                skill for skill in self.tools
                if any(keyword in skill.lower() for keyword in ['file', 'json', 'read', 'write'])
            ]
            if not relevant_skills:
                relevant_skills = self.tools  # Fallback para todas as skills se nenhuma for relevante
            for skill in relevant_skills:
                base_prompt += f"- {skill}\n"
        else:
            for skill in self.tools:
                base_prompt += f"- {skill}\n"

        base_prompt += (
            "\nGere o plano como uma lista JSON de strings, cada uma descrevendo um passo objetivo e direto. "
            "Não inclua texto introdutório ou explicações fora do JSON.\n"
        )

        return base_prompt

    # Adicionar parâmetro para limitar skills no prompt do planejador
    def _build_planner_prompt(self, objective: str, relevant_skills_only: bool = False) -> str:
        # Implementation of _build_planner_prompt method
        pass

    def register_skills(self):
        """Registra todas as skills disponíveis para o agente."""
        agent_logger.info("[Agent] Registrando skills...")
        self.tools.clear()
        self.tools.update(get_skill_registry())
        
        agent_logger.info(f"[Agent] {len(self.tools)} skills registradas.")

    # Modify task completion points to call the helper
    async def run_task(self, objective: str, max_steps: Optional[int] = None) -> Dict:
        """Entry point to run a task. Instantiates and delegates to TaskOrchestrator."""
        log_prefix = "[Orchestrator]"
        agent_logger.info(f"{log_prefix} Starting task: {objective}")
        
        # Add the initial objective to the history
        self.add_history_entry(role="user", content=objective)
        self._log_info(f"[Agent Start] Added user objective to history: {objective[:100]}...")
        
        # Delegate the actual orchestration
        orchestrator_result = await self.orchestrator.orchestrate(objective, max_steps)
        
        # <<< Integrate full history returned by orchestrator >>>
        returned_history = orchestrator_result.pop("full_history", []) # Remove history from result dict
        if returned_history:
             self._history.extend(returned_history) # Add the steps from orchestration
             agent_logger.info(f"[Agent Final] Integrated {len(returned_history)} history steps from orchestrator.")
        else:
             agent_logger.warning("[Agent Final] Orchestrator did not return detailed history.")
             
        # Agent might still be responsible for final state saving or other wrap-up?
        # For now, assume orchestrator handles logging/learning cycle invocation.
        # The orchestrator returns the final result dict.
        
        # Add the final result to the agent's history before saving
        final_content = ""
        role_to_add = "assistant" # Default role
        # Check the status and extract final answer or message from the orchestrator result
        if orchestrator_result.get("status") == "success" and "final_answer" in orchestrator_result:
            # <<< FIX: Add "Final Answer: " prefix >>>
            final_content = f"Final Answer: {orchestrator_result['final_answer']}" 
            # role_to_add remains "assistant"
        elif "message" in orchestrator_result:
            # Log errors as assistant messages for now
            final_content = orchestrator_result['message']
            # role_to_add remains "assistant"

        if final_content:
             # Use the corrected method to add to history
             self.add_history_entry(role=role_to_add, content=final_content)
             self._log_info(f"[Agent Final] Added final result ({role_to_add}) to history: {final_content[:100]}...")
        else:
             self._log_info("[Agent Final] No specific final_answer or message found in result to add to history.")

        # Save the final state (now including integrated history)
        self._save_state()
        
        # Ensure full history is in the final result
        orchestrator_result["full_history"] = self._history
        orchestrator_result["shared_task_context"] = self.last_run_context

        self.agent_logger.info(f"{log_prefix} Orchestration finished for objective '{objective}'. Final Status: {orchestrator_result.get('status')}. Result Keys: {list(orchestrator_result.keys())}")
        return orchestrator_result # Return the result dict (without the history key)

    # Ensure _execute_action and _handle_observation are compatible
    # ... (_execute_action, _handle_observation, _save_state) ...

    # ... (rest of class, including parse_llm_response, _execute_action, etc.) ...

    def _log_info(self, message: str, color: str = "default"):
        """Helper to log info messages with optional color."""
        if self.agent_logger.isEnabledFor(logging.INFO):
            self.agent_logger.info(Text(message, style=color))
            
    def _log_debug(self, message: str, color: str = "default"):
        """Helper to log debug messages with optional color."""
        if self.agent_logger.isEnabledFor(logging.DEBUG):
            self.agent_logger.debug(Text(message, style=color))

    def _parse_react_output(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parses the ReAct output string to extract Thought, Action, and Action Input.

        Prioritizes 'Final Answer:' check. If found, returns it as the thought
        and None for action/input, signaling completion.
        """
        log_prefix = "[Agent Parse DEBUG]"
        # Use rich Text for colored logging
        agent_logger.debug(Text(f"{log_prefix} Raw LLM Response (expecting ReAct Text):\\n{text}", style="cyan"))

        # --- Prioritize Final Answer ---
        final_answer_match = re.search(r"^\\s*Final Answer:(.*)", text, re.MULTILINE | re.DOTALL)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            agent_logger.info(f"[Agent Parse INFO] Final Answer detected: {final_answer}")
            # Return the final answer in the 'thought' position for simplicity in the loop handling
            # Action and Action Input are None, signaling the loop to stop.
            return final_answer, None, None

        # --- If no Final Answer, parse Thought, Action, Action Input ---
        thought_match = re.search(r"^\s*Thought:(.*)", text, re.MULTILINE | re.DOTALL)
        action_match = re.search(r"^\s*Action:(.*)", text, re.MULTILINE | re.DOTALL)
        action_input_match = re.search(r"Action Input:(.*)", text, re.MULTILINE | re.DOTALL) # DOTALL allows matching across newlines

        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        action_input_str = action_input_match.group(1).strip() if action_input_match else None

        agent_logger.debug(f"{log_prefix} Extracted Thought: {thought}")
        agent_logger.debug(f"{log_prefix} Extracted Action: {action}")
        agent_logger.debug(f"{log_prefix} Raw Action Input Content: {action_input_str}")


        # Validate and clean Action Input (attempt to parse JSON)
        validated_action_input_str = None
        if action_input_str:
            # Try to find the first valid JSON object (handles extra text after JSON)
            json_match = re.search(r"\{.*\}", action_input_str, re.DOTALL)
            if json_match:
                potential_json = json_match.group(0)
                agent_logger.debug(f"{log_prefix} Potential JSON string extracted via regex: '{potential_json}'")
                try:
                    # Validate if it's actual JSON
                    json.loads(potential_json)
                    validated_action_input_str = potential_json
                    agent_logger.debug(f"{log_prefix} Final Validated Action Input String: {validated_action_input_str}")
                except json.JSONDecodeError as e:
                    agent_logger.warning(f"{log_prefix} Failed to decode JSON from Action Input: {e}. Raw input: '{action_input_str}'")
                    # Fallback or error handling? For now, pass None or raw string?
                    # Passing raw string might break tool validation later. Let's pass None.
                    validated_action_input_str = None # Or potentially pass action_input_str and let tool handle?
            else:
                 agent_logger.warning(f"{log_prefix} Could not find JSON object pattern in Action Input: '{action_input_str}'")


        # Check if essential parts are missing (excluding Final Answer case)
        if not action:
            agent_logger.warning(f"{log_prefix} Could not parse Action from response: {text}")
            # Decide how to handle this - maybe return error? For now, return None action.
            return thought, None, None # Return thought if available, but no action

        if not validated_action_input_str and action != "final_answer": # final_answer might not need input
             agent_logger.warning(f"{log_prefix} Could not parse valid JSON Action Input for action '{action}'. Raw input: '{action_input_str}'")
             # Return action but None input, tool execution will likely fail validation gracefully
             return thought, action, None


        agent_logger.info(f"[Agent Parse INFO] Text parsed. Action: '{action}'. Input String: '{validated_action_input_str}'")
        return thought, action, validated_action_input_str

    async def _execute_action(
        self,
        action_name: str,
        action_input_dict: Dict[str, Any],
        allowed_skills: List[str]
    ) -> Dict[str, Any]:
        if action_name not in allowed_skills:
            self.agent_logger.error(
                f"[Agent Action ERROR] Action '{action_name}' is NOT in allowed skills {allowed_skills}!"
            )
            return {"status": "error", "message": f"Action '{action_name}' is not allowed for this Fragment."}
        
        # Log execution attempt, colored green
        self._log_info(f"[Agent Action EXEC] Executing: {action_name} with input: {action_input_dict}", color="green")
        
        try:
            result = await self.tool_executor.execute_tool(
                action_name,
                action_input_dict,
                self.workspace_root
            )
            return result
        except Exception as e:
            self.agent_logger.error(f"[Agent Action ERROR] Error executing action: {e}")
            return {"status": "error", "message": f"Error executing action: {e}"}
