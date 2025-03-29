import logging

def build_react_prompt(objective: str, history: list, system_prompt: str, tool_descriptions: str, agent_logger: logging.Logger) -> list[dict]:
    """Constrói a lista de mensagens para o LLM com base no objetivo e histórico."""
    messages = [{"role": "system", "content": system_prompt.replace("[TOOL_DESCRIPTIONS]", tool_descriptions)}]

    # Adiciona objetivo (pode ser principal ou meta)
    messages.append({"role": "user", "content": f"Meu objetivo atual é: {objective}"}) #<<<FIXED: Added missing f-string prefix>>>

    # Processa histórico ReAct
    if history:
         assistant_turn_parts = []
         for entry in history:
             # Agrupa Thought/Action/Input como 'assistant'
             # <<< MODIFIED: Check start of string directly >>>
             if entry.startswith(("Thought:", "Action:", "Action Input:")):
                 assistant_turn_parts.append(entry)
             # Trata Observation como 'user' (input do ambiente)
             elif entry.startswith("Observation:"):
                 if assistant_turn_parts:
                      messages.append({"role": "assistant", "content": "\n".join(assistant_turn_parts)})
                      assistant_turn_parts = []
                 messages.append({"role": "user", "content": entry}) # Observation vem do 'user' (ambiente)

         # Adiciona partes restantes do assistente se o histórico não terminar com Observation
         if assistant_turn_parts:
              messages.append({"role": "assistant", "content": "\n".join(assistant_turn_parts)})

    # <<< REMOVED: Tool description replacement (now done in system prompt) >>>
    # agent_logger.debug(f"[Prompt Builder DEBUG] Final constructed prompt messages: {messages}") # Optional debug
    return messages
