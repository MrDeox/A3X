import logging
from typing import List, Dict, Optional


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
