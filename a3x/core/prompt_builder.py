import logging
from typing import List, Dict, Optional, Tuple

# <<< MOVED and RENAMED from agent.py >>>
DEFAULT_WORKER_SYSTEM_PROMPT = """
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

# <<< MOVED from agent.py >>>
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
def build_orchestrator_messages(
    objective: str,
    history: List[Tuple[str, str]], # Expecting agent's history format
    fragment_descriptions: str
) -> List[Dict[str, str]]:
    """Builds the list of messages for the Orchestrator LLM call."""
    # Format history simply for now (can be improved)
    # Assuming history is List[Tuple[action_str, observation_dict]]
    formatted_history = "\n".join([f"{action}: {obs}" for action, obs in history])

    system_prompt = DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT.format(
        fragment_descriptions=fragment_descriptions
    )

    # Construct messages list
    messages = [
        {"role": "system", "content": system_prompt},
        # Add history if any
        *([{"role": "assistant", "content": f"Previous conversation history:\n{formatted_history}"}] if formatted_history else []),
        {"role": "user", "content": f"Overall Objective: {objective}"}
    ]
    return messages

# <<< NEW FUNCTION: Build Worker Prompt (based on old agent._build_worker_prompt) >>>
def build_worker_messages(
    sub_task: str,
    history: List[Tuple[str, str]], # Expecting fragment's internal history format
    allowed_skills: List[str],
    all_tools: Dict # The full tool registry to get descriptions
) -> List[Dict[str, str]]:
    """Builds the messages list for a worker fragment's ReAct cycle."""
    # Generate tool description string ONLY for allowed skills
    tool_desc_parts = []
    for skill_name in allowed_skills:
        skill_func = all_tools.get(skill_name)
        if skill_func and hasattr(skill_func, 'description'):
            tool_desc_parts.append(f"- {skill_name}: {getattr(skill_func, 'description', 'No description')}")
    tool_descriptions = "\n".join(tool_desc_parts) if tool_desc_parts else "No skills available for this fragment."

    # Format history (same as orchestrator for now)
    formatted_history = "\n".join([f"{action}: {obs}" for action, obs in history])

    system_content = DEFAULT_WORKER_SYSTEM_PROMPT.format(
        tool_descriptions=tool_descriptions, # Use filtered descriptions
        history=formatted_history,
        input=sub_task # Pass sub_task as 'input' here for ReAct
    )

    messages = [
        {"role": "system", "content": system_content},
        # Worker prompts often just need the system prompt containing the task (input) and history
    ]
    return messages
