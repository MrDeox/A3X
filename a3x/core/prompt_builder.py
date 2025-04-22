import logging
from typing import List, Dict, Optional, Tuple
import json

# <<< ADDED Import >>>
from a3x.core.context import SharedTaskContext
from a3x.core.context_accessor import ContextAccessor

# <<< MOVED and RENAMED from agent.py >>>
DEFAULT_WORKER_SYSTEM_PROMPT = """
You are an evolving AI agent designed to achieve user objectives through reasoning, action, and continuous learning. Your primary goal is to solve tasks effectively while improving your capabilities with each interaction.

**IMPORTANT: You MUST ALWAYS respond using the strict ReAct format below, for EVERY step. NO other text outside this format is allowed.**

Strictly follow this format in ALL your responses:

Thought: [Briefly explain your reasoning, plan for the *single* next action, and how it contributes to solving the task.]
Action: [The *exact name* of ONE skill from the provided list, e.g., read_file. DO NOT write sentences or descriptions here.]
Action Input: [Parameters for the skill in valid JSON format, e.g., {{"operation": "read", "file_path": "data/config/users.json"}}]
Observation: [Result of the action]

Example 1 (Reading a file):
Thought: I need to read the user's configuration file to understand their settings.
Action: read_file
Action Input: {{"path": "config/settings.yaml"}}
Observation: {{"result": "Read successfully"}}

Example 2 (Writing to a file):
Thought: The user asked me to save the results to a file. I will use the write_file skill.
Action: write_file
Action Input: {{"path": "output/results.txt", "content": "These are the final results."}}
Observation: {{"result": "Saved successfully"}}

Example 3 (Using a planner):
Thought: The task is complex and requires multiple steps. I should use the hierarchical_planner first to break it down.
Action: hierarchical_planner
Action Input: {{"task_description": "Create a new web component and integrate it into the main page.", "available_tools": ["read_file", "write_file", "execute_code"]}}
Observation: {{"result": "Plan created successfully"}}

**CRITICAL RULES:**
1.  **ReAct Format:** Always use the "Thought:", "Action:", "Action Input:", "Observation:" structure.
2.  **Action Field:** The `Action:` field MUST contain ONLY the exact name of ONE skill from the available list. No extra words, descriptions, or sentences.
3.  **Action Input Field:** The `Action Input:` field MUST be a valid JSON object matching the parameters required by the chosen skill. Pay close attention to the required arguments for each skill. If a skill needs `path` and `content`, your JSON must provide both.
4.  **JSON Validity:** Ensure the JSON in `Action Input:` is correctly formatted (double quotes around keys and string values, correct braces and commas).
5.  **No Code Blocks:** DO NOT generate markdown code blocks (like ```json ... ```) around the Action Input JSON or anywhere else.
6.  **One Step at a Time:** Focus only on the immediate next action required to progress towards the objective.

If you have completed the objective and no further actions are needed, respond ONLY with:
Thought: I have completed the task.
Action: final_answer
Action Input: {{"answer": "[Your final summary or result.]"}}

**Do NOT use any other format. Do NOT output explanations, markdown, or code blocks outside the required fields.**

Available Skills (use ONLY these exact names for `Action` and provide required args in `Action Input`):
{tool_descriptions}

Previous conversation history (Actions and Observations):
{history}

Current User Objective: {input}

**Respond now with your next Thought, Action, and Action Input.**
"""

# <<< MOVED & UPDATED from agent.py >>>
DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT = """
You are the A3X Orchestrator. Your role is to analyze the user's overall objective, the conversation history, and the **current task context**, then delegate the *next single step* to the most appropriate specialized component: either a Manager (for coordination) or a direct Executor Fragment.

You must choose one component from the available list and define a clear, concise sub-task for it.

Available Components (Workers):
{fragment_descriptions}
# Note: Descriptions now include (Category: Management/Execution) and Managed/Skills info.

Choose a component based on the task requirements:
# **Use the Current Context to inform your decision.** For example, if the last action failed, consider using the DebuggerFragment or retrying with different parameters. If a file was just read, the next step might involve processing it.

- **If the overall objective seems complex, requires multiple steps, or involves actions across different domains (e.g., reading a file, then searching the web), ALWAYS choose the `PlannerFragment` first to create a structured plan.**
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

def build_react_prompt(
    objective: str,
    history: list,
    system_prompt: str,
    tool_descriptions: str,
    agent_logger: logging.Logger,
) -> list[dict]:
    """Constrói a lista de mensagens para o ciclo ReAct do LLM.

    Organiza o histórico e o objetivo em um formato que o LLM entende para
    decidir o próximo Thought e Action.
    """
    messages = [
        {
            "role": "system",
            "content": system_prompt.replace("[TOOL_DESCRIPTIONS]", tool_descriptions),
        }
    ]

    # Adiciona objetivo (pode ser principal ou sub-objetivo do passo do plano)
    # <<< FIXED: Added missing f-string prefix >>>
    messages.append({"role": "user", "content": f"Objective: {objective}"})

    # Processa histórico ReAct (Thought, Action, Observation)
    if history:
        assistant_turn_parts = []
        for entry in history:
            # <<< MODIFIED: Check start of string directly >>>
            # Thought/Action/Input são agregados na resposta do assistente
            if entry.startswith(("Thought:", "Action:", "Action Input:")):
                assistant_turn_parts.append(entry)
            # Observation é tratado como input do usuário (ambiente)
            elif entry.startswith(("Observation:", "Final Answer:")):
                # Se houver partes do assistente pendentes, adiciona-as primeiro
                if assistant_turn_parts:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "\n".join(assistant_turn_parts),
                        }
                    )
                    assistant_turn_parts = []
                # Adiciona a Observation/Final Answer como input do usuário
                messages.append({"role": "user", "content": entry})

        # Adiciona partes restantes do assistente se o histórico não terminar com Observation/Final Answer
        if assistant_turn_parts:
            messages.append(
                {"role": "assistant", "content": "\n".join(assistant_turn_parts)}
            )

    # Adiciona instrução final para o LLM pensar
    messages.append({"role": "user", "content": "What is your next Thought and Action?"})

    # agent_logger.debug(f"[Prompt Builder DEBUG] Final ReAct prompt messages: {messages}") # Optional verbose debug
    return messages


# <<< NEW FUNCTION >>>
def build_planning_prompt(
    objective: str, tool_descriptions: str, planner_system_prompt: str,
    heuristics_context: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Constrói a lista de mensagens para a fase de Planejamento do LLM,
    incluindo instruções restritivas, exemplos e contexto real das skills.
    """
    # Instruções explícitas e exemplos
    instructions = """
Você é o planejador do agente Arthur (A³X). Seu objetivo é gerar um plano de passos claros, objetivos e minimalistas para atingir o objetivo do usuário, usando apenas as skills disponíveis abaixo.

NÃO inclua passos que usem ferramentas não listadas abaixo.
NÃO utilize web_search, hierarchical_planner, read_file, ou qualquer skill não disponível.
Para tarefas de escrita em arquivo, use diretamente a skill write_file.
Decomponha o objetivo apenas se necessário para garantir clareza e segurança.
Evite passos genéricos, redundantes ou que não contribuem diretamente para o objetivo.

Exemplo de plano bom para "Salve o texto 'Teste heurística' no arquivo 'nao_existe2/teste2.txt'":
1. Use a skill write_file para salvar o texto no arquivo solicitado.
2. Use a skill final_answer para confirmar a operação.

Exemplo de plano ruim (NÃO FAÇA):
1. Use web_search para buscar exemplos de texto.
2. Use hierarchical_planner para decompor o objetivo.
3. Use skills não listadas abaixo.
"""

    # Monta a mensagem do usuário
    user_content = f'{instructions}\n\nObjetivo do usuário: "{objective}"'

    if heuristics_context:
        user_content += f'\n\nHeurísticas relevantes:\n{heuristics_context}'

    user_content += f'\n\nSkills disponíveis:\n{tool_descriptions}\n\nGere o plano como uma lista JSON de strings, cada uma descrevendo um passo objetivo e direto.'

    messages = [
        {"role": "system", "content": planner_system_prompt},
        {"role": "user", "content": user_content}
    ]
    return messages


# <<< NEW FUNCTION >>>
def build_final_answer_prompt(
    objective: str, steps: List[str], agent_logger: logging.Logger
) -> List[Dict[str, str]]:
    agent_logger.info("Construindo prompt para resposta final com streaming...")
    context_str = "\n".join(steps)
    return [
        {
            "role": "system",
            "content": "Você é um assistente de IA que responde com base nos passos anteriores de forma clara, direta e completa.",
        },
        {
            "role": "user",
            "content": f"""Com base nos seguintes passos e observações, gere uma resposta final para o usuário.

Objetivo: {objective}

Histórico:
{context_str}

Responda agora de forma direta e informativa, como um assistente humano finalizando a tarefa.""",
        },
    ]

# <<< NEW FUNCTION: Build Orchestrator Prompt >>>
async def build_orchestrator_messages(
    shared_task_context: Optional[SharedTaskContext] = None, # Use the actual class
    objective: str = "No objective specified.",
    history: List[Tuple[str, str]] = [], # Expecting agent's history format
    fragment_descriptions: str = "",
) -> List[Dict[str, str]]:
    """Builds the list of messages for the Orchestrator LLM call."""
    # Format history simply for now (can be improved)
    # Assuming history is List[Tuple[action_str, observation_dict]]
    formatted_history = "\n".join([f"{action}: {obs}" for action, obs in history])

    system_prompt = DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT.format(
        fragment_descriptions=fragment_descriptions
    )

    # --- Instantiate ContextAccessor ---
    context_accessor = ContextAccessor(shared_task_context)
    
    # --- Extract and Format Richer Shared Context --- 
    context_items = []
    
    # Get last history result summary
    last_result_summary = await context_accessor.get_last_history_result()
    if last_result_summary:
        # Truncate long results for the prompt
        result_str = str(last_result_summary)
        context_items.append(f"- Last Result Summary: {result_str[:200]}{'...' if len(result_str) > 200 else ''}")

    # Get last error using the correct method (and await)
    last_err = await context_accessor.get_task_data("last_error")
    if last_err:
        context_items.append(f"- Last Error: {last_err}")
        
    # Get last file read path using the correct method (and await)
    last_file_read = await context_accessor.get_task_data("last_file_read_path")
    if last_file_read:
        context_items.append(f"- Last file read: {last_file_read}")
        
    # >>> ADDED: Get last file written path <<<
    last_file_written = await context_accessor.get_task_data("last_file_written_path")
    if last_file_written:
        context_items.append(f"- Last file written: {last_file_written}")
        
    # Get last execution result using the correct method (and await)
    last_exec_result = await context_accessor.get_task_data("last_execution_result")
    if last_exec_result and isinstance(last_exec_result, dict):
        status = last_exec_result.get('status', 'unknown')
        # Correctly extract action from the 'data' dictionary if it exists
        action_data = last_exec_result.get('data', {})
        action = action_data.get('action', 'unknown_action') if isinstance(action_data, dict) else 'unknown_action'
        message = action_data.get('message', '') if isinstance(action_data, dict) else '' # Get message from data too if available
        # Avoid duplicating error if already captured by get("last_error")
        # Check if last_err was retrieved before using it
        if status != 'error' or (last_err is None):
             # Use the extracted action in the context summary
             context_items.append(f"- Last execution result detail ({action}): {status} {(' - ' + message) if message else ''}")
             
    # Add more context points here if needed in the future (e.g., summary, key variables)

    # --- ADDED: Explicit Logging of Retrieved Context --- 
    logger = logging.getLogger(__name__) # Ensure logger is available
    logger.info(f"[Orchestrator Prompt Builder] Retrieved Context for Prompt:")
    logger.info(f"  - Objective: {objective[:100]}...")
    logger.info(f"  - History Length: {len(history)}")
    logger.info(f"  - Shared Context ID: {shared_task_context.task_id if shared_task_context else 'N/A'}")
    logger.info(f"  - Raw Retrieved Data:")
    logger.info(f"    - last_result_summary: {last_result_summary}")
    logger.info(f"    - last_error: {last_err}")
    logger.info(f"    - last_file_read_path: {last_file_read}")
    logger.info(f"    - last_file_written_path: {last_file_written}") # Log the newly added check
    logger.info(f"    - last_execution_result: {last_exec_result}")
    # --- END ADDED LOGGING ---
    
    formatted_context = "\n".join(context_items) if context_items else "No relevant context from previous steps is available."
    # --- End Context Formatting ---

    # Construct messages list
    messages = [
        {"role": "system", "content": system_prompt},
        # Add history if any - Consider if this is still needed long-term if context is rich enough
        *([{"role": "assistant", "content": f"Previous conversation history (raw):\n{formatted_history}"}] if formatted_history else []),
        # Add the extracted context as a system message for emphasis
        {"role": "system", "content": f"Current Context Summary:\n{formatted_context}"}, 
        # Update user prompt slightly
        {"role": "user", "content": f"Overall Objective: {objective}\n\nGiven the objective, history, and **especially the current context summary above**, what is the most appropriate next component and sub-task?"}
    ]
    return messages

# <<< NEW FUNCTION: Build Worker Prompt (based on old agent._build_worker_prompt) >>>
def build_worker_messages(
    sub_task: str,
    history: List[Tuple[str, str]], # Expecting fragment's internal history format
    allowed_skills: List[str],
    all_tools: Dict, # The full tool registry {name: schema}
    max_history_items: int = 15 # Limit history in prompt
) -> List[Dict[str, str]]:
    """
    Builds the prompt messages for a worker Fragment's ReAct cycle.

    Args:
        sub_task: The specific objective for this fragment/cycle.
        history: The ReAct history (Thought, Action, Action Input, Observation tuples).
        allowed_skills: List of skill names this fragment is allowed to use.
        all_tools: Dictionary mapping all available tool names to their schemas.

    Returns:
        A list of messages formatted for the LLM.
    """
    logger = logging.getLogger(__name__) # Use module logger

    # <<< Inner helper function _format_allowed_tools >>>
    def _format_allowed_tools(skills_to_format: List[str], tool_schemas: Dict) -> str:
        formatted_lines = []
        if not tool_schemas:
            return "No tool schemas provided."
        
        for skill_name in skills_to_format:
            schema = tool_schemas.get(skill_name)
            if schema and isinstance(schema, dict):
                formatted_lines.append(f"### Skill: {schema.get('name', skill_name)}")
                formatted_lines.append(f"* Description: {schema.get('description', 'No description.')}")
                
                # Format parameters
                params_obj = schema.get('parameters', {})
                properties = params_obj.get('properties', {}) if isinstance(params_obj, dict) else {}
                required = set(params_obj.get('required', [])) if isinstance(params_obj, dict) else set()
                
                param_strs = []
                for p_name, p_info in properties.items():
                     if isinstance(p_info, dict) and p_name not in ('self', 'cls', 'ctx', 'context'):
                         p_type = p_info.get('type', 'any')
                         p_desc = p_info.get('description', '')
                         req_ind = " (required)" if p_name in required else ""
                         param_strs.append(f"  - {p_name} ({p_type}){req_ind}: {p_desc}")
                
                if param_strs:
                    formatted_lines.append("* Parameters:")
                    formatted_lines.extend(param_strs)
                else:
                    formatted_lines.append("* Parameters: None")
                formatted_lines.append("") # Add blank line after each tool
            else:
                logger.warning(f"Schema not found or invalid for allowed skill: {skill_name}")
                formatted_lines.append(f"### Skill: {skill_name} (Schema Error)")
                formatted_lines.append("")

        return "\n".join(formatted_lines).strip() if formatted_lines else "No allowed tools specified or schemas found."

    # <<< Inner helper function _format_history >>>
    def _format_history(history_list: List[Tuple[str, str]], max_items: int) -> str:
        if not history_list:
            return "No history yet."
        
        formatted_entries = []
        # Take the most recent 'max_items'
        start_index = max(0, len(history_list) - max_items)
        
        for i, (thought_action_input, observation) in enumerate(history_list[start_index:]):
            # Ensure structure matches expectation for worker prompt
            formatted_entries.append(f"Previous Step {i+1} Thought/Action/Input:")
            formatted_entries.append(thought_action_input)
            formatted_entries.append(f"Previous Step {i+1} {observation}") # Observation already has prefix
            formatted_entries.append("---") # Separator
            
        return "\n".join(formatted_entries).strip()
    
    # <<< MOVED: Prepare JSON list before using in f-string >>>
    allowed_skills_json = json.dumps(allowed_skills)
    # logger.debug(f"Allowed skills JSON: {allowed_skills_json}")

    # --- System Prompt ---
    # <<< MODIFIED: Changed from f-string to regular string, ensure placeholders use single braces >>>
    system_prompt = """You are an AI assistant fragment. Complete the sub-task using Thought/Action/Action Input.

Available Tools (Skills):
--- Tools Start ---
{formatted_allowed_tools}
--- Tools End ---

Respond ONLY in this format:

Thought: [Your reasoning]
Action: [EXACT tool name]
Action Input: [Arguments as a single JSON object. Keys MUST match tool parameters. Use {{}} if no arguments. DO NOT use a 'context' key.]

Example:
Thought: I need to plan the main task.
Action: hierarchical_planner
Action Input: {{
    "task_description": "{sub_task}",
    "available_tools": {allowed_skills_json}
}}
"""

    # --- Format Allowed Tools --- 
    formatted_allowed_tools = _format_allowed_tools(allowed_skills, all_tools)
    # logger.debug(f"Formatted allowed tools for prompt:\\n{formatted_allowed_tools}")

    # --- Format History ---
    formatted_history = _format_history(history, max_history_items) 

    # --- Build Messages ---
    # Pass sub_task and allowed_skills_json into the system prompt format
    messages = [{
        "role": "system", 
        "content": system_prompt.format(
            formatted_allowed_tools=formatted_allowed_tools, 
            sub_task=sub_task, # Pass sub_task for the example
            allowed_skills_json=allowed_skills_json # Pass JSON list for example
        )
    }]

    # Add history and user request (simplified)
    user_prompt = f"""Previous History:
{formatted_history}

===

Current Sub-Task: {sub_task}

Determine the next step."""
    messages.append({"role": "user", "content": user_prompt})

    return messages
