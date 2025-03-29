\
import logging
import json
from typing import Dict, Any, Optional, TYPE_CHECKING

# Avoid circular import for type hinting
if TYPE_CHECKING:
    from .agent import ReactAgent

# Import config and logger
try:
    from .config import MAX_META_DEPTH
except ImportError:
    MAX_META_DEPTH = 3 # Fallback

# Direct import for agent_logger
from .log_config import agent_logger

def try_autocorrect(agent_instance: 'ReactAgent', tool_result: Dict[str, Any], last_executed_code: Optional[str], current_history: list, meta_depth: int) -> Optional[str]:
    """
    Attempts to trigger an auto-correction cycle if a tool execution failed,
    specifically targeting 'execute_code' failures.

    Args:
        agent_instance: The ReactAgent instance to call run() recursively.
        tool_result: The result dictionary from the executed tool.
        last_executed_code: The code that was last attempted (if any).
        current_history: The current conversation history.
        meta_depth: The current meta-cycle depth.

    Returns:
        Optional[str]: The formatted Observation string with the auto-correction result,
                       or None if auto-correction was not triggered or not applicable.
    """
    if (
        tool_result.get("status") == "error" and
        tool_result.get("action") == "execution_failed" and
        last_executed_code and
        meta_depth < MAX_META_DEPTH
    ):
        log_prefix = f"[ReactAgent META-{meta_depth + 1}]" # Log for the *upcoming* meta cycle
        agent_logger.warning(f"{log_prefix} Erro na execução do código detectado. Tentando auto-correção (Profundidade: {meta_depth + 1})...")
        agent_logger.debug(f"{log_prefix} Código com erro: \n{last_executed_code}")
        agent_logger.debug(f"{log_prefix} Erro reportado: {tool_result.get('data', {}).get('message', 'N/A')}")

        # Construir o meta-objetivo (instrução simplificada para a LLM)
        # Usamos o histórico *atual* (antes da observação do erro original) como contexto
        last_thought_action = "Contexto anterior indisponível." # Fallback
        if len(current_history) >= 2 and current_history[-1].startswith("Action:") and current_history[-2].startswith("Thought:"):
             last_thought_action = f"{current_history[-2]}\n{current_history[-1]}"
        elif len(current_history) >= 1 and current_history[-1].startswith("Action:"):
             last_thought_action = current_history[-1]

        meta_objective = (
            f"META-OBJECTIVE (Auto-Correction Cycle {meta_depth + 1}/{MAX_META_DEPTH}):\\n"
            f"The previous attempt to execute code failed. Your goal is to FIX the code and attempt execution again.\\n"
            f"Reason for failure: {tool_result.get('data', {}).get('message', 'Unknown error')}\\n"
            f"Code that failed:\\n```python\\n{last_executed_code}\\n```\\n"
            f"Previous thought/action leading to the error:\\n{last_thought_action}\\n\\n"
            f"Instructions:\\n"
            f"1. Analyze the error message and the failing code.\\n"
            f"2. **Think** step-by-step how to correct the code.\\n"
            f"3. Use the 'modify_code' tool *only* if you need to change the code. Provide the *entire* corrected code block.\\n"
            f"4. Use the 'execute_code' tool with the corrected code.\\n"
            f"5. If modification is not needed (e.g., environment issue), explain the problem using 'final_answer'.\\n"
            f"6. Respond ONLY in the Thought/Action JSON format. Focus *only* on fixing and executing the code."
        )

        # Corrected logging: Log the prefix string first, then the meta_objective separately
        agent_logger.info(f"{log_prefix} Iniciando ciclo de auto-correção com meta-objetivo:")
        agent_logger.info(meta_objective) # Log the potentially long objective on its own line

        # Chamada recursiva para o ciclo de meta-correção
        meta_result = agent_instance.run(meta_objective, is_meta_objective=True, meta_depth=meta_depth + 1)
        agent_logger.info(f"{log_prefix} Ciclo de auto-correção finalizado. Resultado: {meta_result}")

        # Formatar a observação para o ciclo *principal*
        # Inclui o erro original E o resultado da tentativa de correção
        observation = (
            f"Observation: A execução do código anterior falhou: {tool_result.get('data', {}).get('message', 'N/A')}.\n"
            f"Auto-correction attempt (Cycle {meta_depth + 1}) result: {meta_result}"
        )
        return observation

    return None # Auto-correction not triggered

