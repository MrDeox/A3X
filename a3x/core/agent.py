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
from a3x.core.tool_executor import execute_tool, _ToolExecutionContext
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
# <<< ADDED import for SharedTaskContext >>>
from a3x.core.context import SharedTaskContext
# <<< ADDED import for uuid >>>
import uuid
# <<< ADDED import for the new orchestrator >>>
from a3x.core.orchestrator import TaskOrchestrator

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
    def __init__(self, agent_id="1", agent_name="A3X", llm_interface: LLMInterface = None, 
                 workspace_root=None, initial_memory=None, skill_registry=None, 
                 skill_descriptions=None):
        """Initializes the React Agent container."""
        agent_logger.debug(f"[Agent INIT] Received llm_interface: {llm_interface}")
        self.llm_interface = llm_interface or DEFAULT_LLM_URL # This line might be problematic if llm_interface is None and DEFAULT_LLM_URL is just a string
        
        self.tools = skill_registry if skill_registry is not None else get_skill_registry()
        self._history = []  # Histórico de Thought, Action, Observation
        self.agent_id = agent_id
        self.agent_logger = agent_logger
        self._memory: Dict[str, Any] = initial_memory or {}
        
        # Correct attribute access from .url to .llm_url
        self.llm_url = self.llm_interface.llm_url or os.getenv(
            "LLM_API_URL"
        )  # Use env var if not provided
        
        self.workspace_root = workspace_root or Path(PROJECT_ROOT).resolve()
        agent_logger.info(f"Agent initialized with workspace root: {self.workspace_root}")
        self.fragment_registry = FragmentRegistry()

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
        
        # Instantiate the orchestrator, passing necessary dependencies
        orchestrator = TaskOrchestrator(
            llm_interface=self.llm_interface,
            fragment_registry=self.fragment_registry,
            tools=self.tools,
            workspace_root=self.workspace_root,
            agent_logger=self.agent_logger,
            # Pass other dependencies if needed by orchestrator methods
        )
        
        # Delegate the actual orchestration
        final_result = await orchestrator.orchestrate(objective, max_steps)
        
        # Agent might still be responsible for final state saving or other wrap-up?
        # For now, assume orchestrator handles logging/learning cycle invocation.
        # The orchestrator returns the final result dict.
        return final_result

    # Ensure _execute_action and _handle_observation are compatible
    # ... (_execute_action, _handle_observation, _save_state) ...

    # ... (rest of class, including parse_llm_response, _execute_action, etc.) ...
