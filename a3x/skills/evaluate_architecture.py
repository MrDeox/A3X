# a3x/skills/evaluate_architecture.py

import logging
import json
from typing import Dict, Any, Optional, List

# Core components
from a3x.core.skills import skill, SkillContext
from a3x.core.llm_interface import LLMInterface
from a3x.core.context import Context
# from a3x.reflection.structure_reflector import StructureReflector # <<< COMMENTED OUT

logger = logging.getLogger(__name__)

@skill(
    name="evaluate_architecture",
    description="Avalia a arquitetura de um módulo do projeto (ex: a3x/core, a3x/cli) com base nos manifestos e retorna um diagnóstico.",
    parameters={
        "type": "object",
        "properties": {
            "module_path": {
                "type": "str",
                "description": "Caminho relativo ao root do projeto para o módulo a ser avaliado (ex: a3x/core, a3x/cli). Default: a3x",
            }
        },
        "required": [] # module_path is optional, defaults to 'a3x'
    }
)
async def evaluate_architecture(context: Any, module_path: str = "a3x") -> str:
    """Executa uma análise arquitetural semântica em um módulo do projeto.

    Args:
        context: The skill execution context.
        module_path: Path relative to project root for the module to scan (default: 'a3x').

    Returns:
        A string containing the summarized report of the architectural analysis.
    """
    logger.info(f"Starting architectural evaluation for module: {module_path}")

    # Get LLMInterface from context (assuming it's available as context.llm_interface)
    # If context structure is different, this needs adjustment.
    llm_interface: LLMInterface | None = getattr(context, 'llm_interface', None)

    if not llm_interface:
        error_msg = "LLMInterface not found in SkillContext. Cannot perform analysis."
        logger.error(error_msg)
        return f"Error: {error_msg}"

    try:
        # reflector = StructureReflector(llm_interface=llm_interface) # COMMENTED OUT
        pass # Added pass to make the try block valid
    except ValueError as e:
        logger.error(f"Failed to initialize StructureReflector: {e}")
        return f"Error initializing reflector: {e}"
    except Exception as e:
        logger.exception("Unexpected error initializing StructureReflector:")
        return f"Unexpected error initializing reflector: {e}"

    if not reflector.heuristics:
        return "Error: StructureReflector initialized, but no architectural heuristics were loaded."

    logger.info(f"Running project scan on: {module_path}")
    try:
        scan_results: Dict[str, str] = await reflector.run_project_scan(base_path=module_path)
    except Exception as e:
        logger.exception(f"Error during project scan for {module_path}:")
        return f"Error running project scan: {e}"

    # Format the results into a report
    report_lines = [f"# Relatório de Avaliação Arquitetural para: {module_path}", "---"]
    if not scan_results:
        report_lines.append("Nenhum resultado retornado pela varredura.")
    elif "error" in scan_results:
        report_lines.append(f"Erro durante a varredura: {scan_results['error']}")
    elif "info" in scan_results:
         report_lines.append(f"Info: {scan_results['info']}")
    else:
        file_count = len(scan_results)
        report_lines.append(f"Análise concluída para {file_count} arquivo(s).")
        report_lines.append("\n## Diagnósticos por Arquivo:")
        for filename, diagnosis in scan_results.items():
            report_lines.append(f"\n### Arquivo: `{filename}`")
            report_lines.append(f"```\n{diagnosis}\n```") # Use code block for potentially long diagnosis

    final_report = "\n".join(report_lines)
    logger.info(f"Architectural evaluation for {module_path} complete.")
    return final_report 