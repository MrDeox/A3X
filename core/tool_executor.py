import logging
import inspect  # <<< ADDED import >>>
from typing import Dict, Any

# <<< Import TOOLS for validation >>>
# from core.tools import TOOLS # REMOVED - TOOLS does not exist


# <<< MODIFIED Signature: Added agent_memory AND made async >>>
async def execute_tool(
    tool_name: str,
    action_input: Dict[str, Any],
    tools_dict: dict,
    agent_logger: logging.Logger,
    agent_memory: Dict[str, Any],
) -> Dict[str, Any]:
    """Executa a ferramenta/skill selecionada com validação prévia e passagem correta de argumentos."""
    log_prefix = "[Tool Executor]"
    agent_logger.info(
        f"{log_prefix} Attempting tool: '{tool_name}', Input: {action_input}"
    )  # Log intent

    # --- Validação Rigorosa ---
    # if tool_name not in TOOLS: # Use the imported TOOLS constant # OLD
    if tool_name not in tools_dict:  # NEW - Use the passed tools_dict for validation
        # available_tools = list(TOOLS.keys()) # OLD
        available_tools = list(tools_dict.keys())  # NEW - Use the passed tools_dict
        error_message = (
            f"Ferramenta '{tool_name}' não existe. Disponíveis: {available_tools}"
        )
        agent_logger.error(f"{log_prefix} {error_message}")
        return {
            "status": "error",
            "action": "tool_not_found",  # Use a specific action code
            "data": {"message": error_message},
        }
    # --- Fim Validação ---

    # Proceed only if tool exists
    agent_logger.debug(
        f"{log_prefix} Tool '{tool_name}' found. Proceeding with execution."
    )
    tool = tools_dict[
        tool_name
    ]  # Use the passed tools_dict which should be same as TOOLS
    tool_function = tool["function"]

    try:
        # <<< MODIFIED: Inspect signature and call appropriately, check for async >>>
        sig = inspect.signature(tool_function)
        params = sig.parameters
        is_async = inspect.iscoroutinefunction(tool_function)

        call_args = action_input  # Arguments from the LLM action input
        result = None

        # Check if the skill function expects agent_memory
        if "agent_memory" in params:
            agent_logger.debug(
                f"{log_prefix} Passing agent_memory to skill '{tool_name}'. Async: {is_async}"
            )
            if is_async:
                result = await tool_function(**call_args, agent_memory=agent_memory)
            else:
                result = tool_function(**call_args, agent_memory=agent_memory)
        else:
            agent_logger.debug(
                f"{log_prefix} Skill '{tool_name}' does not take agent_memory. Calling with action_input args only. Async: {is_async}"
            )
            if is_async:
                result = await tool_function(**call_args)
            else:
                result = tool_function(**call_args)
        # <<< END MODIFICATION >>>

        if not isinstance(result, dict):
            agent_logger.error(
                f"{log_prefix} Skill '{tool_name}' returned unexpected type: {type(result)}. Expected: dict."
            )
            return {
                "status": "error",
                "action": "invalid_skill_return",
                "data": {"message": f"Skill '{tool_name}' returned invalid type."},
            }

        agent_logger.info(
            f"{log_prefix} Skill '{tool_name}' executed. Status: {result.get('status', 'N/A')}"
        )
        agent_logger.debug(f"{log_prefix} Skill '{tool_name}' result: {result}")

        # LÓGICA DE ATUALIZAÇÃO DE MEMÓRIA FOI MOVIDA PARA O LOOP run() PRINCIPAL NO AGENT

        return result  # Retorna o dicionário completo da skill

    except TypeError as e:
        # Catch TypeErrors specifically, often related to missing/extra arguments
        agent_logger.exception(
            f"{log_prefix} TypeError executing skill '{tool_name}': {e}"
        )
        # Provide more specific error message about argument mismatch
        return {
            "status": "error",
            "action": f"{tool_name}_argument_error",
            "data": {
                "message": f"Argument mismatch calling skill '{tool_name}': {str(e)}"
            },
        }
    except Exception as e:
        agent_logger.exception(f"{log_prefix} Error executing skill '{tool_name}':")
        # Retornar uma estrutura de erro padronizada
        return {
            "status": "error",
            "action": f"{tool_name}_failed",  # Keep specific failure action
            "data": {
                "message": f"Internal error executing skill '{tool_name}': {str(e)}"
            },
        }
