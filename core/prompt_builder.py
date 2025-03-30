import logging
from typing import List, Dict

def build_react_prompt(objective: str, history: list, system_prompt: str, tool_descriptions: str, agent_logger: logging.Logger) -> list[dict]:
    """Constrói a lista de mensagens para o ciclo ReAct do LLM.

    Organiza o histórico e o objetivo em um formato que o LLM entende para
    decidir o próximo Thought e Action.
    """
    messages = [{"role": "system", "content": system_prompt.replace("[TOOL_DESCRIPTIONS]", tool_descriptions)}]

    # Adiciona objetivo (pode ser principal ou sub-objetivo do passo do plano)
    # <<< FIXED: Added missing f-string prefix >>>
    messages.append({"role": "user", "content": f"Meu objetivo atual é: {objective}"})

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
                      messages.append({"role": "assistant", "content": "\n".join(assistant_turn_parts)})
                      assistant_turn_parts = []
                 # Adiciona a Observation/Final Answer como input do usuário
                 messages.append({"role": "user", "content": entry})

         # Adiciona partes restantes do assistente se o histórico não terminar com Observation/Final Answer
         if assistant_turn_parts:
              messages.append({"role": "assistant", "content": "\n".join(assistant_turn_parts)})

    # Adiciona instrução final para o LLM pensar
    messages.append({"role": "user", "content": "Qual o seu próximo Thought e Action?"})

    # agent_logger.debug(f"[Prompt Builder DEBUG] Final ReAct prompt messages: {messages}") # Optional verbose debug
    return messages

# <<< NEW FUNCTION >>>
def build_planning_prompt(objective: str, tool_descriptions: str, planner_system_prompt: str) -> List[Dict[str, str]]:
    """Constrói a lista de mensagens para a fase de Planejamento do LLM.

    Args:
        objective: O objetivo final do usuário.
        tool_descriptions: Descrições das ferramentas disponíveis.
        planner_system_prompt: O prompt de sistema específico para o planejador.

    Returns:
        Lista de dicionários de mensagens para a chamada LLM de planejamento.
    """
    messages = [
        {"role": "system", "content": planner_system_prompt},
        {
            "role": "user",
            "content": f"User Objective: \"{objective}\"\n\nAvailable Tools:\n{tool_descriptions}\n\nGenerate the plan as a JSON list of strings."
        }
    ]
    # Logger não é passado aqui, mas pode ser adicionado se necessário debug
    # print(f"[Planner Prompt DEBUG] {messages}")
    return messages

# <<< NEW FUNCTION >>>
def build_final_answer_prompt(objective: str, steps: List[str], agent_logger: logging.Logger) -> List[Dict[str, str]]:
    agent_logger.info("Construindo prompt para resposta final com streaming...")
    context_str = "\n".join(steps)
    return [
        {"role": "system", "content": "Você é um assistente de IA que responde com base nos passos anteriores de forma clara, direta e completa."},
        {"role": "user", "content": f"""Com base nos seguintes passos e observações, gere uma resposta final para o usuário.

Objetivo: {objective}

Histórico:
{context_str}

Responda agora de forma direta e informativa, como um assistente humano finalizando a tarefa."""}
    ]

