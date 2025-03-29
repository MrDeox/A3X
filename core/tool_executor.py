import logging
import json # Added json import
from typing import Dict, Any

def execute_tool(tool_name: str, action_input: Dict[str, Any], tools_dict: dict, agent_logger: logging.Logger) -> Dict[str, Any]:
    """Executa a ferramenta/skill selecionada."""
    log_prefix = f"[Tool Executor]"
    agent_logger.info(f"{log_prefix} Executing tool: '{tool_name}', Input: {action_input}")

    if tool_name not in tools_dict:
        agent_logger.error(f"{log_prefix} Unknown tool: '{tool_name}'")
        return {"status": "error", "action": "tool_not_found", "data": {"message": f"Tool '{tool_name}' not found."}}

    tool = tools_dict[tool_name]
    tool_function = tool["function"]

    try:
        # Simplificado: Todas as ferramentas agora só recebem action_input
        agent_logger.debug(f"{log_prefix} Calling skill '{tool_name}' with action_input only.")
        result = tool_function(action_input=action_input)

        if not isinstance(result, dict):
            agent_logger.error(f"{log_prefix} Skill '{tool_name}' returned unexpected type: {type(result)}. Expected: dict.")
            return {"status": "error", "action": "invalid_skill_return", "data": {"message": f"Skill '{tool_name}' returned invalid type."}}

        agent_logger.info(f"{log_prefix} Skill '{tool_name}' executed. Status: {result.get('status', 'N/A')}")
        agent_logger.debug(f"{log_prefix} Skill '{tool_name}' result: {result}")

        # LÓGICA DE ATUALIZAÇÃO DE MEMÓRIA FOI MOVIDA PARA O LOOP run() PRINCIPAL NO AGENT

        return result # Retorna o dicionário completo da skill

    except Exception as e:
        agent_logger.exception(f"{log_prefix} Error executing skill '{tool_name}':")
        # Retornar uma estrutura de erro padronizada
        return {
            "status": "error",
            "action": f"{tool_name}_failed",
            "data": {"message": f"Internal error executing skill '{tool_name}': {str(e)}"}
        }
