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
from a3x.core.prompt_builder import build_react_prompt
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
You are an evolving AI agent designed to achieve user objectives through reasoning, action, and continuous learning. Your primary goal is to solve tasks effectively while improving your capabilities with each interaction.

**IMPORTANT: You MUST ALWAYS respond using the strict ReAct format below, for EVERY step. NO other text outside this format is allowed.**

Strictly follow this format in ALL your responses:

Thought: [Briefly explain your reasoning, plan for the *single* next action, and how it contributes to solving the task.]
Action: [The *exact name* of ONE skill from the provided list, e.g., read_file. DO NOT write sentences or descriptions here.]
Action Input: [Parameters for the skill in valid JSON format, e.g., {"path": "data/users.json"}]

Example:
Thought: I need to read the JSON file specified in the user objective to access the user data.
Action: read_file
Action Input: {"path": "data/users.json"}

**CRITICAL RULES:**
1.  **Action Field:** The `Action:` field MUST contain ONLY the exact name of ONE skill from the list below. No extra words, descriptions, or sentences.
2.  **Action Input Field:** The `Action Input:` field MUST be a valid JSON object containing the parameters for the chosen skill.
3.  **No Code Blocks:** DO NOT generate code blocks (like ```python ... ```) anywhere in your response. Only provide the skill name and JSON input.
4.  **One Step at a Time:** Focus on the immediate next step in your thought and action.

If you have completed the objective, respond ONLY with:
Final Answer: [Your final summary or result.]

**Do NOT use any other format. Do NOT output explanations, markdown, or code blocks outside the required fields.**

Available Skills (use ONLY these exact names for Action):
{tool_descriptions}

Previous conversation history:
{history}

User Objective for this step: {input}

**Focus on executing one valid action at a time based on your thought process.**
"""

# --- New Prompt for Orchestrator --- 
DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT = """
You are the A3X Orchestrator. Your role is to analyze the user's overall objective and the conversation history, then delegate the *next single step* to the most appropriate specialized component: either a Manager (for coordination) or a direct Executor Fragment.

You must choose one component from the available list and define a clear, concise sub-task for it.

Available Components (Workers):
{fragment_descriptions} 
# Note: Descriptions now include (Category: Management/Execution) and Managed/Skills info.

Choose a component based on the task requirements:
- If the task requires coordination of multiple low-level actions within a specific domain (e.g., file operations, code operations), choose the appropriate **Manager** (Category: Management).
- If the task is self-contained or represents the final step, choose a direct **Executor** Fragment (Category: Execution, e.g., FinalAnswerProvider, Planner).

Respond ONLY with a JSON object containing two keys: 'component' (the name of the chosen Manager or Fragment) and 'sub_task' (the specific instruction for that component).

Example (File Operation Task):
{{
  "component": "FileOpsManager",
  "sub_task": "Read the content of the file 'config.yaml'"
}}

Example (Final Step Task):
{{
  "component": "FinalAnswerProvider",
  "sub_task": "Inform the user that the file has been successfully updated."
}}

Do not attempt to perform the task yourself. Only delegate.
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
    def __init__(self, agent_id="1", agent_name="A3X", llm_interface: LLMInterface = None, 
                 workspace_root=None, initial_memory=None, skill_registry=None, 
                 skill_descriptions=None, system_prompt=DEFAULT_REACT_SYSTEM_PROMPT):
        """Inicializa o Agente ReAct."""
        # Add diagnostic log before accessing llm_interface attributes
        agent_logger.debug(f"[ReactAgent INIT] Received llm_interface - Type: {type(llm_interface)}, Value: {llm_interface}")
        
        self.llm_interface = llm_interface or DEFAULT_LLM_URL # This line might be problematic if llm_interface is None and DEFAULT_LLM_URL is just a string
        
        # Add another log AFTER the potential assignment to DEFAULT_LLM_URL string
        agent_logger.debug(f"[ReactAgent INIT] self.llm_interface AFTER assignment - Type: {type(self.llm_interface)}, Value: {self.llm_interface}")
        
        self.system_prompt = system_prompt
        self.tools = skill_registry if skill_registry is not None else get_skill_registry()
        self._history = []  # Histórico de Thought, Action, Observation
        self.agent_id = agent_id
        self.agent_logger = agent_logger
        self._memory: Dict[str, Any] = initial_memory or {}
        
        # This line causes the AttributeError: 'LLMInterface' object has no attribute 'url'
        # Let's check the type of self.llm_interface RIGHT BEFORE this line
        agent_logger.debug(f"[ReactAgent INIT] Checking self.llm_interface JUST BEFORE accessing .url - Type: {type(self.llm_interface)}")
        # Correct attribute access from .url to .llm_url
        self.llm_url = self.llm_interface.llm_url or os.getenv(
            "LLM_API_URL"
        )  # Use env var if not provided
        
        self.workspace_root = workspace_root or Path(PROJECT_ROOT).resolve()
        agent_logger.info(f"Agent initialized with workspace root: {self.workspace_root}")
        self.max_iterations = MAX_REACT_ITERATIONS
        self.fragment_registry = FragmentRegistry()
        self.fragment_descriptions = self.fragment_registry.get_available_fragments_description()
        if not self.fragment_descriptions or "Error" in self.fragment_descriptions or "No fragments" in self.fragment_descriptions : # Check for empty/error string
            self.logger.error(f"Failed to get valid fragment descriptions from registry! Got: {self.fragment_descriptions}")
            # Use a minimal default if loading fails or is empty
            self.fragment_descriptions = "- FinalAnswerProvider (Execution): Provides the final answer."

        self.orchestrator_system_prompt = DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT.format(
            fragment_descriptions=self.fragment_descriptions
        )
        # Keep the original system prompt for the workers
        self.worker_system_prompt = system_prompt 

        # Load agent state if exists
        loaded_state = load_agent_state(agent_id=self.agent_id)
        if loaded_state:
            self._history = loaded_state.get("history", [])
            self._memory = loaded_state.get("memory", {})
            agent_logger.info(
                f"[ReactAgent INIT] Estado do agente carregado para ID '{self.agent_id}'. Memória: {list(self._memory.keys())}"
            )
        agent_logger.info(
            f"[ReactAgent INIT] Agente inicializado. LLM URL: {'Default' if not self.llm_url else self.llm_url}. Memória carregada: {list(self._memory.keys())}"
        )

    # <<< Method to Call LLM and Parse Response >>>
    async def _process_llm_response(self, messages: List[Dict[str, str]], log_prefix: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """Calls LLM and parses Thought/Action/Input format.\n           Returns (thought, action_name, action_input_dict) or (None, None, None)."""
        llm_response_raw = ""
        try:
            # <<< ADDED DIAGNOSTIC LOGGING >>>
            agent_logger.debug(f"{log_prefix} Checking llm_interface before call. Type: {type(self.llm_interface)}, Value: {self.llm_interface}")
            # <<< END ADDED DIAGNOSTIC LOGGING >>>
            
            # --- Corrected: Iterate over the async generator --- 
            agent_logger.debug(f"{log_prefix} Calling llm_interface.call_llm (stream=False expected for ReAct cycle)")
            async for chunk in self.llm_interface.call_llm(
                messages=messages,
                stream=False, # Explicitly set stream=False for ReAct response
                # Stop sequence etc. can be passed as kwargs if needed
                stop=["\\nObservation:"] # Standard ReAct stop sequence
            ):
                llm_response_raw += chunk
            # --- End Correction --- 
                
            if not llm_response_raw:
                 agent_logger.warning(f"{log_prefix} LLM call returned empty response.")
                 return None, None, None # Return tuple on error
                 
            agent_logger.debug(f"{log_prefix} Raw LLM response length: {len(llm_response_raw)}")
            agent_logger.debug(f"{log_prefix} Raw LLM response (first 500): {llm_response_raw[:500]}")
        except Exception as e:
            agent_logger.error(f"{log_prefix} Error calling LLM: {e}", exc_info=True)
            return None, None, None # Return tuple on error

        # Parse the response using the function from agent_parser
        # This already returns the correct tuple format
        return parse_llm_response(llm_response_raw, self.agent_logger)

    # <<< Method to Execute Action >>>
    async def _execute_action(
        self, action_name: str, action_input: Dict[str, Any], log_prefix: str, shared_task_context: SharedTaskContext
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
                tools_dict=self.tools, # Pass the agent's tools_dict
                llm_interface=self.llm_interface, # Pass the agent's llm_interface
                fragment_registry=self.fragment_registry, # <<< ADDED >>>
                shared_task_context=shared_task_context # <<< ADDED shared_task_context >>>
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

    # <<< Method to Perform ReAct Iteration >>>
    async def _perform_react_iteration(self, objective: str, log_prefix="[Agent]") -> AsyncGenerator[Dict, None]:
        """Performs one full Thought-Action-Observation cycle of the ReAct loop."""
        # Trim history before building prompt
        self._history = trim_history(self._history, MAX_HISTORY_TURNS, agent_logger)

        try:
            # === Orchestrator Step ===
            chosen_fragment, sub_task = await self._get_next_step_delegation(objective, self._history)

            if not chosen_fragment or not sub_task:
                agent_logger.error(f"{log_prefix} Orchestrator failed to delegate. Aborting cycle.")
                yield {"type": "error", "status": "error", "content": "Orchestrator failed to delegate next step."}
                return

            # Handle delegation to FinalAnswerProvider separately? Or let it run react?
            if chosen_fragment == "FinalAnswerProvider":
                # Maybe FinalAnswerProvider just needs the sub_task as the answer?
                agent_logger.info(f"{log_prefix} Orchestrator chose FinalAnswerProvider. Assuming sub_task is the answer.")
                # Directly execute final_answer? Requires final_answer skill to handle this.
                final_answer_result = await self._execute_action(chosen_fragment, {"answer": sub_task}, log_prefix, None)
                # How to structure the return to signal completion?
                yield {"type": "final_answer", "status": "success", "final_answer": sub_task, "observation": final_answer_result} # Or similar
            
            # === Worker Fragment Step ===
            allowed_skills = get_skills_for_fragment(chosen_fragment)
            agent_logger.info(f"{log_prefix} Running Worker Fragment: {chosen_fragment} with skills: {allowed_skills}")
            
            # Build prompt for the chosen worker fragment
            worker_prompt = self._build_worker_prompt(sub_task, self._history, allowed_skills)
            
            agent_logger.info(f"{log_prefix} Calling LLM for Worker Fragment {chosen_fragment}...")
            parsed_result = await self._process_llm_response(
                 [{"role": "system", "content": self.worker_system_prompt}, {"role": "user", "content": worker_prompt}],
                 log_prefix=log_prefix
            )
            if not parsed_result:
                 error_msg = "LLM processing failed completely."
                 agent_logger.error(f"{log_prefix} Error processing LLM response: {error_msg}")
                 yield {"type": "error", "status": "error", "content": error_msg}
                 return

            thought, action, action_input = parsed_result

            if thought:
                agent_logger.info(f"{log_prefix} Thought: {thought}")
            else:
                agent_logger.warning(f"{log_prefix} No 'Thought' found in parsed output.")

            # --- Handle Action (if provided) --- 
            if not action:
                agent_logger.error(f"{log_prefix} No Action specified by LLM.")
                observation = {"status": "error", "message": "LLM Fragment did not specify an action."}
                action_str_for_history = "Error: No Action"
            else: # Action was specified, proceed with validation and execution
                action_str_for_history = action # Default to action name for history
                # --- Action Validation --- 
                if action not in allowed_skills:
                    agent_logger.error(f"{log_prefix} Action '{action}' is NOT in allowed skills {allowed_skills}!")
                    observation = {"status": "error", "message": f"Action '{action}' is not allowed for this Fragment."}
                    action_str_for_history = f"Error: Disallowed Action ({action})"
                elif action not in self.tools:
                    agent_logger.error(f"{log_prefix} Action '{action}' not found in skill registry.")
                    observation = {"status": "error", "message": f"Skill '{action}' not found."}
                    action_str_for_history = f"Error: Unknown Skill ({action})"
                else:
                    # --- Execute Action --- 
                    agent_logger.info(f"{log_prefix} Action ({chosen_fragment}): {action}({json.dumps(action_input)})")
                    agent_logger.info(f"{log_prefix} Executing Action: {action} with input: {action_input}")
                    try:
                        observation = await self._execute_action(action, action_input, log_prefix, None)
                        # <<< ADDED DEBUG LOG >>>
                        agent_logger.debug(f"{log_prefix} Raw observation received from _execute_action for '{action}': {observation}")
                        # <<< END ADDED DEBUG LOG >>>
                        agent_logger.info(f"{log_prefix} Observation: {str(observation)[:500]}...") # Log truncated observation
                    except Exception as e:
                        agent_logger.error(f"{log_prefix} Error executing action '{action}': {e}", exc_info=True)
                        observation = {"status": "error", "message": f"Error executing action '{action}': {e}"}
                        action_str_for_history = f"Error: Execution Failed ({action})"
            
            # Record step in history (Orchestrator step isn't an action itself, worker step is)
            self._history.append((f"{chosen_fragment}:{action}" if action else f"{chosen_fragment}:Error", observation))
            self._memory.record_episodic_event(context="react_step", action=action, outcome=observation, metadata={"thought": thought, "input": action_input})

            # Check for final answer explicitly here?
            if action == "final_answer":
                agent_logger.info(f"{log_prefix} Final answer provided by {chosen_fragment}. Ending interaction.")
                yield {"type": "final_answer", "status": "success", "final_answer": action_input.get("answer"), "observation": observation}
            
            # Loop limit check etc. remains the same
            # ... (rest of loop logic) ...

        except Exception as e:
            agent_logger.exception(f"{log_prefix} Error during react iteration: {e}")
            # Ensure state saving happens
            self._save_state()
        
        yield {"type": "error", "status": "error", "content": "Max iterations reached or unexpected error."} # Fallback error

    # --- Orchestration Logic ---

    async def _get_next_step_delegation(self, objective: str, history: List[Tuple[str, str]]) -> Tuple[Optional[str], Optional[str]]:
        """
        Calls the LLM (acting as Orchestrator) to determine the next Fragment and sub-task.
        Returns (chosen_fragment_name, sub_task_description) or (None, None) on failure.
        """
        agent_logger.info("[Orchestrator] Deciding next step delegation...")
        orchestrator_prompt = self._build_orchestrator_prompt(objective, history)

        raw_response_str = "" # Store the raw string response
        try:
            # Call LLM directly, expecting a non-streaming response containing JSON
            agent_logger.debug(f"[Orchestrator] Calling llm_interface directly for JSON response.")
            # Ensure llm_interface is valid before calling
            if not isinstance(self.llm_interface, LLMInterface):
                 agent_logger.error(f"[Orchestrator] Invalid llm_interface type: {type(self.llm_interface)}")
                 return None, None

            # Use await with async for loop for asynchronous generator
            async for chunk in self.llm_interface.call_llm(
                messages=[{"role": "system", "content": self.orchestrator_system_prompt}, {"role": "user", "content": orchestrator_prompt}],
                stream=False,
                # Add specific parameters for orchestrator if needed, e.g., lower temperature?
                # temperature=0.2
            ):
                 # Assuming chunk is already a string, adjust if it's bytes
                 if isinstance(chunk, bytes):
                     chunk = chunk.decode('utf-8') # Or appropriate encoding
                 raw_response_str += chunk # Accumulate response chunks into a single string

            if not raw_response_str:
                 agent_logger.error("[Orchestrator] LLM call returned empty string.")
                 return None, None

            agent_logger.debug(f"[Orchestrator] Raw LLM String Response: {raw_response_str}")

            # Parse the JSON response from the orchestrator string
            try:
                # Find JSON block (handle potential markdown formatting)
                # Regex updated slightly to be less strict about newline after ```json
                # Also handle potential leading/trailing whitespace around JSON object
                json_match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', raw_response_str, re.DOTALL | re.IGNORECASE)
                json_str = None
                if json_match:
                     # Extract JSON string (use the first non-empty group)
                     json_str = next((group for group in json_match.groups() if group is not None), None)
                     if json_str:
                         agent_logger.debug(f"[Orchestrator] Extracted JSON block: {json_str}")
                         delegation_data = json.loads(json_str.strip()) # Added strip()
                     else:
                         # This case should ideally not happen with the regex, but good to handle
                         agent_logger.warning("[Orchestrator] Regex matched but no JSON group found.")
                         json_str = None # Ensure json_str is None if extraction fails

                # If no JSON block found OR extraction failed, attempt to parse the whole string
                if not json_str:
                    agent_logger.warning(f"[Orchestrator] No ```json block found or extracted in response string: {raw_response_str}. Attempting to parse whole string.")
                    try:
                        # Attempt to parse the whole string as JSON as a fallback
                        delegation_data = json.loads(raw_response_str.strip()) # Added strip()
                        agent_logger.info("[Orchestrator] Parsed entire response as JSON (fallback).")
                    except json.JSONDecodeError as fallback_err:
                         agent_logger.error(f"[Orchestrator] Failed to decode fallback JSON: {raw_response_str} - Error: {fallback_err}")
                         return None, None # Give up if fallback also fails


                chosen_fragment = delegation_data.get("component")
                sub_task = delegation_data.get("sub_task")

                # <<< CORRECTED VALIDATION: Use get_all_definitions() >>>
                # if chosen_fragment and chosen_fragment in self.fragment_registry.get_definitions() and sub_task:
                if chosen_fragment and chosen_fragment in self.fragment_registry.get_all_definitions() and sub_task:
                    agent_logger.info(f"[Orchestrator] Delegating to Fragment: {chosen_fragment}, Sub-task: {sub_task}")
                    return chosen_fragment, sub_task
                else:
                    agent_logger.error(f"[Orchestrator] Invalid delegation data received: {delegation_data}")
                    # Log specific missing field
                    if not chosen_fragment: agent_logger.error("[Orchestrator] 'component' key missing or invalid in JSON.")
                    # <<< CORRECTED ERROR LOGGING: Use get_all_definitions() >>>
                    # if chosen_fragment and chosen_fragment not in self.fragment_registry.get_definitions(): agent_logger.error(f"[Orchestrator] Fragment '{chosen_fragment}' not found in FragmentRegistry definitions.")
                    if chosen_fragment and chosen_fragment not in self.fragment_registry.get_all_definitions(): agent_logger.error(f"[Orchestrator] Fragment '{chosen_fragment}' not found in FragmentRegistry definitions (get_all_definitions).")
                    if not sub_task: agent_logger.error("[Orchestrator] 'sub_task' key missing in JSON.")
                    return None, None

            except json.JSONDecodeError as e:
                agent_logger.error(f"[Orchestrator] Failed to decode JSON response string: '{json_str or raw_response_str}' - Error: {e}")
                return None, None
            except Exception as e: # Catch other potential errors during parsing/access
                agent_logger.error(f"[Orchestrator] Error parsing delegation response string: '{json_str or raw_response_str}' - Error: {e}", exc_info=True)
                return None, None

        except Exception as e: # Catch errors during the LLM call itself
            agent_logger.error(f"[Orchestrator] LLM call failed: {e}", exc_info=True) # Add traceback
            return None, None

    def _build_orchestrator_prompt(self, objective: str, history: List[Tuple[str, str]]) -> str:
        """Builds the prompt for the Orchestrator LLM call."""
        prompt_lines = []
        prompt_lines.append(f"Overall Objective: {objective}")
        prompt_lines.append("\nConversation History (Action/Observation):")
        if not history:
            prompt_lines.append("(No history yet)")
        else:
            for i, (action, observation) in enumerate(history):
                prompt_lines.append(f"Step {i+1} Action: {action}")
                # Limit observation length for prompt clarity?
                obs_summary = str(observation)[:500] + ('... (truncated)' if len(str(observation)) > 500 else '')
                prompt_lines.append(f"Step {i+1} Observation: {obs_summary}")
        
        prompt_lines.append("\nBased on the objective and history, determine the next single step.")
        prompt_lines.append("Specify the Fragment worker and the sub-task for it in the required JSON format.")
        return "\n".join(prompt_lines)
        
    def _build_worker_prompt(self, sub_task: str, history: List[Tuple[str, str]], allowed_skills: List[str]) -> str:
        """Builds the ReAct prompt for the selected Fragment Worker."""
        # This is similar to the original _build_agent_prompt, but uses sub_task and allowed_skills
        prompt_lines = []
        # System prompt is handled separately in call_llm
        prompt_lines.append(f"Your Current Task: {sub_task}")
        prompt_lines.append("\nConversation History (Action/Observation for this task):")
        # Note: History might need filtering/adjustment if it contains steps from other fragments
        # For now, pass the full history related to the overall objective
        if not history:
            prompt_lines.append("(No history yet)")
        else:
             for i, (action, observation) in enumerate(history):
                prompt_lines.append(f"Previous Step {i+1} Action: {action}")
                obs_summary = str(observation)[:500] + ('... (truncated)' if len(str(observation)) > 500 else '')
                prompt_lines.append(f"Previous Step {i+1} Observation: {obs_summary}")

        prompt_lines.append("\nAvailable Skills (Use ONLY these):")
        if not allowed_skills:
             prompt_lines.append("(No skills available for this fragment!)") # Should not happen
        else:
             # Use the skill descriptions from the registry, filtered by allowed_skills
             for skill_name in allowed_skills:
                  if skill_name in self.tools:
                       prompt_lines.append(f"- {skill_name}: {self.tools[skill_name]['description']}")
                  else:
                       agent_logger.warning(f"Description for allowed skill '{skill_name}' not found in registry.")
                       prompt_lines.append(f"- {skill_name}") # Fallback to just name

        prompt_lines.append("\nStrictly follow the ReAct format: Thought, Action, Action Input.")
        prompt_lines.append("Action must be exactly one of the available skills listed above.")
        prompt_lines.append("Action Input must be a valid JSON object or empty if not needed.")
        prompt_lines.append("What is your next Thought and Action?")
        return "\n".join(prompt_lines)

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

    async def _execute_fragment_task(
        self, 
        fragment_name: str, 
        sub_task: str, 
        allowed_skills: List[str],
        parent_history: List[Tuple[str, str]], # History from orchestrator for context
        shared_task_context: SharedTaskContext # <<< ADDED shared_task_context >>>
    ) -> Dict:
        """
        Executes the ReAct cycle for a specific Fragment worker until the sub-task is completed.
        Uses only the allowed_skills for this fragment.
        Returns a dict with status and final answer/observation.
        """
        log_prefix = f"[{fragment_name} Fragment]" # Specific log prefix
        agent_logger.info(f"{log_prefix} Starting execution for sub-task: '{sub_task}' with skills: {allowed_skills}")
        
        current_task_history: List[Tuple[str, str]] = [] # History *within* this sub-task
        iterations = 0
        max_fragment_iterations = 10 # Limit iterations *per fragment task* 

        while iterations < max_fragment_iterations:
            iterations += 1
            agent_logger.info(f"{log_prefix} Iteration {iterations}/{max_fragment_iterations}")

            # Build prompt for the worker fragment using its focused history
            worker_prompt = self._build_worker_prompt(sub_task, current_task_history, allowed_skills)
            
            agent_logger.info(f"{log_prefix} Calling LLM...")
            # Use _process_llm_response which handles the call and basic parsing
            # We pass the worker system prompt here
            parsed_result = await self._process_llm_response(
                 # Build the message list format expected by _process_llm_response
                 [{"role": "system", "content": self.worker_system_prompt}, {"role": "user", "content": worker_prompt}],
                 log_prefix=log_prefix 
            )
            
            # Check for LLM call or parsing errors from _process_llm_response
            if not parsed_result:
                 error_msg = "LLM processing failed completely."
                 agent_logger.error(f"{log_prefix} Error processing LLM response: {error_msg}")
                 return {"status": "error", "message": error_msg, "fragment_history": current_task_history}
            thought, action, action_input = parsed_result

            if thought:
                agent_logger.info(f"{log_prefix} Thought: {thought}")
            else:
                agent_logger.warning(f"{log_prefix} No 'Thought' found in parsed output.")

            # --- Handle Action (if provided) --- 
            if not action:
                agent_logger.error(f"{log_prefix} No Action specified by LLM.")
                observation = {"status": "error", "message": "LLM Fragment did not specify an action."}
                action_str_for_history = "Error: No Action"
            else: # Action was specified, proceed with validation and execution
                action_str_for_history = action # Default to action name for history
                # --- Action Validation --- 
                if action not in allowed_skills:
                    agent_logger.error(f"{log_prefix} Action '{action}' is NOT in allowed skills {allowed_skills}!")
                    observation = {"status": "error", "message": f"Action '{action}' is not allowed for this Fragment."}
                    action_str_for_history = f"Error: Disallowed Action ({action})"
                elif action not in self.tools:
                    agent_logger.error(f"{log_prefix} Action '{action}' not found in skill registry.")
                    observation = {"status": "error", "message": f"Skill '{action}' not found."}
                    action_str_for_history = f"Error: Unknown Skill ({action})"
                else:
                    # --- Execute Action --- 
                    agent_logger.info(f"{log_prefix} Action ({fragment_name}): {action}({json.dumps(action_input)}) ") # Added fragment name
                    try:
                        # Use the internal _execute_action method
                        # <<< MODIFIED: Pass shared_task_context >>>
                        observation = await self._execute_action(
                            action, 
                            action_input, 
                            log_prefix, 
                            shared_task_context=shared_task_context
                        )
                        # <<< END MODIFIED >>>
                        agent_logger.info(f"{log_prefix} Observation: {str(observation)[:500]}...") # Log truncated observation
                        # action_str_for_history remains the action name if successful
                    except Exception as e:
                        agent_logger.error(f"{log_prefix} Error executing action '{action}': {e}", exc_info=True)
                        observation = {"status": "error", "message": f"Error executing action '{action}': {e}"}
                        action_str_for_history = f"Error: Execution Failed ({action})"
            
            # Append to the *fragment's* history for the next iteration
            current_task_history.append((action_str_for_history, observation))

            # --- Log step to Episodic Memory DB --- 
            try:
                # Format context, action, outcome for the database record
                context_str = f"Sub-task: {sub_task} | History: {str(current_task_history[:-1])[-200:]}" # Context is sub-task + recent internal history
                action_str = action_str_for_history # Use the string representation (includes errors)
                outcome_str = json.dumps(observation, ensure_ascii=False) # Outcome is the observation
                metadata = {
                    "fragment": fragment_name,
                    "thought": thought, 
                    "action": action, # Original action name
                    "action_input": action_input,
                    "iteration": iterations
                }
                # Call the imported function directly
                add_episodic_record(context_str, action_str, outcome_str, metadata)
                agent_logger.debug(f"{log_prefix} Step logged to episodic memory.")
            except Exception as db_err:
                 agent_logger.error(f"{log_prefix} Failed to log step to episodic memory: {db_err}")
            # --- End Logging ---

            # --- Check for Sub-Task Completion --- 
            # Check completion *after* adding the step to history
            if action == "final_answer":
                 final_answer_content = action_input.get("answer", "(No answer content provided)")
                 agent_logger.info(f"{log_prefix} Sub-task '{sub_task}' completed with final answer.")
                 return {
                     "status": "success", 
                     "final_answer": final_answer_content, 
                     "observation": observation, # Observation from final_answer skill itself
                     "fragment_history": current_task_history
                 }
            
            # Check if tool execution resulted in an error that should stop the fragment
            # Check *after* adding to history so the error is recorded
            if isinstance(observation, dict) and observation.get("status") == "error":
                 agent_logger.warning(f"{log_prefix} An error occurred during action execution or validation. Stopping fragment task.")
                 return {
                      "status": "error", 
                      "message": observation.get("message", "Error during action execution/validation"),
                      "fragment_history": current_task_history
                 }

        # If loop finishes without final_answer
        agent_logger.warning(f"{log_prefix} Max iterations ({max_fragment_iterations}) reached for sub-task '{sub_task}' without completion.")
        return {
             "status": "error", 
             "message": f"Max iterations reached for sub-task", 
             "fragment_history": current_task_history
        }

    # <<< ADDED: Helper method to invoke learning cycle >>>
    async def _invoke_learning_cycle(self, objective: str, main_history: List, final_status: str, shared_context: SharedTaskContext):
        """Invokes the learning cycle skill with the final task state."""
        log_prefix = "[Learning Cycle Invoker]"
        try:
            agent_logger.info(f"{log_prefix} Invoking learning cycle skill for objective: {objective[:100]}... Status: {final_status}")
            # Note: We might need a more structured plan representation than just history
            # For now, passing empty plan, history acts as results.
            # Use the agent's internal _execute_action to ensure context propagation
            await self._execute_action(
                action_name="learning_cycle_skill", 
                action_input={
                    "objective": objective,
                    "plan": [], 
                    "execution_results": main_history, 
                    "final_status": final_status,
                    "agent_tools": self.tools,
                    "agent_workspace": str(self.workspace_root.resolve()),
                    "agent_llm_url": self.llm_url,
                    "shared_task_context": shared_context # Pass context to skill input
                },
                log_prefix=log_prefix, 
                shared_task_context=shared_context # Pass context for execution context setup
            )
            agent_logger.info(f"{log_prefix} Learning cycle skill executed.")
        except Exception as learn_err:
            agent_logger.error(f"{log_prefix} Error executing learning cycle: {learn_err}", exc_info=True)

    # Modify task completion points to call the helper
    async def run_task(self, objective: str, max_steps: Optional[int] = None) -> Dict:
        """Orchestrates Fragments to achieve the overall objective."""
        log_prefix = "[Orchestrator]"
        agent_logger.info(f"{log_prefix} Starting task: {objective}")
        
        # Create SharedTaskContext for this task run
        task_id = str(uuid.uuid4())
        shared_context = SharedTaskContext(task_id=task_id, initial_objective=objective)
        agent_logger.info(f"{log_prefix} Initialized SharedTaskContext with ID: {task_id}")

        main_history: List[Tuple[str, str]] = [] 
        orchestrator_iterations = 0
        max_orchestrator_iterations = max_steps if max_steps is not None else 20 
        agent_logger.info(f"{log_prefix} Maximum orchestration steps allowed: {max_orchestrator_iterations}")

        last_failed_sub_task: Optional[str] = None
        consecutive_failures: int = 0
        MAX_CONSECUTIVE_FAILURES_BEFORE_DEBUG: int = 2 
        failure_history_for_debugger: List[Dict] = []
        
        final_result = None # Store the final result dict
        final_status = "unknown" # Track final status for learning cycle

        while orchestrator_iterations < max_orchestrator_iterations:
            orchestrator_iterations += 1
            agent_logger.info(f"{log_prefix} Orchestration Cycle {orchestrator_iterations}/{max_orchestrator_iterations}")

            chosen_fragment, sub_task = await self._get_next_step_delegation(objective, main_history)

            if not chosen_fragment or not sub_task:
                agent_logger.error(f"{log_prefix} Orchestrator failed to delegate. Aborting task.")
                final_status = "error_orchestrator_delegation"
                final_result = {"status": "error", "message": "Orchestrator failed to delegate next step."}
                break # Exit loop

            if sub_task != last_failed_sub_task:
                agent_logger.debug(f"{log_prefix} New sub-task '{sub_task}'. Resetting consecutive failure count.")
                last_failed_sub_task = sub_task
                consecutive_failures = 0
                failure_history_for_debugger = []

            if chosen_fragment == "FinalAnswerProvider":
                agent_logger.info(f"{log_prefix} Orchestrator chose FinalAnswerProvider. Task complete.")
                final_status = "completed"
                final_result = {"status": "success", "final_answer": sub_task}
                # Append this final step to history before breaking
                main_history.append(("FinalAnswerProvider", {"status": "success", "final_answer": sub_task}))
                break # Exit the loop on final answer
            
            fragment_def = self.fragment_registry.get_fragment_definition(chosen_fragment)
            allowed_skills = []
            if fragment_def:
                allowed_skills = fragment_def.managed_skills or fragment_def.skills
                agent_logger.info(f"{log_prefix} Retrieved allowed skills for {chosen_fragment}: {allowed_skills}")
            else:
                agent_logger.error(f"{log_prefix} Could not find definition for fragment '{chosen_fragment}'. Aborting task.")
                final_status = "error_fragment_definition_not_found"
                final_result = {"status": "error", "message": f"Fragment definition for {chosen_fragment} not found in registry."}
                break # Exit loop
            
            fragment_result = await self._execute_fragment_task(
                fragment_name=chosen_fragment,
                sub_task=sub_task,
                allowed_skills=allowed_skills,
                parent_history=main_history, 
                shared_task_context=shared_context
            )
            
            fragment_status = fragment_result.get("status")
            fragment_final_answer = fragment_result.get("final_answer")
            fragment_message = fragment_result.get("message")
            internal_history = fragment_result.get("fragment_history", [])
            
            summary_action = f"Fragment {chosen_fragment} completed sub-task: '{sub_task}'" if fragment_status == "success" else f"Fragment {chosen_fragment} failed sub-task: '{sub_task}'"
            summary_observation = { 
                "status": fragment_status,
                "sub_task_result": fragment_final_answer or fragment_message,
            }
            main_history.append((summary_action, summary_observation))

            if fragment_status == "error":
                agent_logger.error(f"{log_prefix} Fragment {chosen_fragment} failed sub-task '{sub_task}'. Message: {fragment_message}")
                consecutive_failures += 1
                if internal_history: 
                    failure_history_for_debugger.extend(internal_history) 
                agent_logger.warning(f"{log_prefix} Consecutive failures for sub-task '{sub_task}': {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES_BEFORE_DEBUG}")

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES_BEFORE_DEBUG:
                    agent_logger.warning(f"{log_prefix} Failure limit reached. Triggering DebuggerFragment...")
                    # ... (debugger execution logic as before, including passing shared_context) ...
                    debugger_result = await self._execute_fragment_task(
                        # ... (debugger args) ...
                        shared_task_context=shared_context
                    )
                    # ... (process debugger result, update history) ...
                    # Reset failure count after debugger runs
                    consecutive_failures = 0
                    failure_history_for_debugger = []
                    # Continue loop, let orchestrator decide next based on debugger output in history
                else:
                    # Failure limit not reached, continue loop
                    pass 
            else: # fragment_status == "success"
                if sub_task == last_failed_sub_task:
                     agent_logger.debug(f"{log_prefix} Sub-task '{sub_task}' succeeded. Resetting failure count.")
                     consecutive_failures = 0
                     failure_history_for_debugger = []
                agent_logger.info(f"{log_prefix} Fragment {chosen_fragment} completed sub-task successfully.")

        # === Loop End / Task Conclusion ===
        if final_result is None: # Loop finished without hitting FinalAnswerProvider or breaking early
            agent_logger.warning(f"{log_prefix} Max orchestrator iterations ({max_orchestrator_iterations}) reached or loop exited without success for '{objective}'.")
            final_status = "error_max_iterations_or_failed" # More specific status
            final_result = {"status": "error", "message": "Max orchestrator iterations reached or task failed to complete."} 
        # If final_result is already set (e.g., success or early error), final_status reflects that.

        # --- Invoke Learning Cycle --- 
        await self._invoke_learning_cycle(objective, main_history, final_status, shared_context)

        # --- Return Final Result --- 
        agent_logger.info(f"{log_prefix} Task finished for objective '{objective}'. Final Status: {final_status}. Result: {final_result}")
        return final_result

    # Ensure _execute_action and _handle_observation are compatible
    # ... (_execute_action, _handle_observation, _save_state) ...

    # ... (rest of class, including parse_llm_response, _execute_action, etc.) ...
