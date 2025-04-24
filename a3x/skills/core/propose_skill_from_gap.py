import logging
import json
import os
import asyncio
import re
import ast
from typing import Dict, Any, Optional, List

# Core imports
from a3x.core.skills import skill, get_skill_descriptions
from a3x.core.llm_interface import LLMInterface
from a3x.core.agent import _ToolExecutionContext
from a3x.core.context import Context
from a3x.skills.core.call_skill_by_name import call_skill_by_name
from a3x.core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# --- Constants ---

# Directory for saving auto-generated skills
AUTO_GENERATED_SKILLS_DIR = os.path.join("a3x", "skills", "auto_generated")
# AUTO_GENERATED_TESTS_DIR = os.path.join("tests", "skills", "auto_generated") # For future use

# LLM Prompt to generate Python skill code
CODE_GENERATION_PROMPT_TEMPLATE = '''
You are an expert Python programmer specializing in creating modular skills for the A³X agent framework.
Generate the complete Python code for a new A³X skill based *only* on the following specifications.

**Specifications:**
- Skill Name: `{skill_name}`
- Reason for Creation: {reason}
- Detailed Description: {suggestion_description}
- Parameters (as JSON string): {parameters_json}
- Example Usage: {example_usage}

**Instructions:**
1.  Create a complete, runnable Python file content.
2.  Include necessary imports (like `logging`, `typing`, `SkillContext`, `skill`, etc.).
3.  Define the skill function asynchronously (`async def`).
4.  Use the `@skill` decorator correctly with the provided name, description, and parsed parameters.
5.  Implement the core logic based on the description.
6.  Include basic logging using `ctx.logger`.
7.  Return a dictionary with `{{"status": "success", ...}}` or `{{"status": "error", ...}}`.
8.  Ensure the code is well-formatted and includes basic docstrings.
9.  Output ONLY the raw Python code for the file. Do not include explanations, markdown formatting (like ```python), or anything else.
'''

# --- Main Skill ---

@skill(
    name="propose_skill_from_gap",
    description="Proposes a new skill definition based on identified gaps or failures.",
    parameters={
        "context": {"type": Context, "description": "Execution context for LLM and skill registry access."},
        "gap_description": {"type": str, "description": "Description of the functional gap or failure pattern."},
        "relevant_logs": {"type": Optional[List[str]], "default": None, "description": "Optional list of relevant log entries."}
    }
)
async def propose_skill_from_gap(
    context: Context,
    gap_description: str,
    relevant_logs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyzes a described capability gap and proposes a Python skill function definition.

    Args:
        context (Context): Execution context.
        gap_description (str): Description of the missing capability.
        relevant_logs (Optional[List[str]]): Optional related log entries.

    Returns:
        Dict[str, Any]: Dictionary containing the proposed skill code or an error.
    """
    ctx = context # Assign for potential internal compatibility
    ctx.logger.info(f"Executing propose_skill_from_gap for: {gap_description[:100]}...")

    llm_interface = ctx.llm_interface
    if not llm_interface:
        ctx.logger.error("LLMInterface not found in execution context.")
        return {"status": "error", "message": "Internal error: LLMInterface missing."}

    # 1. Get current skills for context
    try:
        current_skills_description = get_skill_descriptions()
        ctx.logger.debug(f"Retrieved {len(current_skills_description.splitlines())} lines of current skill descriptions.")
    except Exception as e:
        ctx.logger.exception("Failed to get current skill descriptions:")
        current_skills_description = "Erro ao obter a lista de skills atuais."

    # 2. Build the prompt
    logs_section = "\n".join(relevant_logs) if relevant_logs else "Nenhum log relevante fornecido."

    prompt = f"""
Você é um Engenheiro de IA Sênior especializado em criar skills modulares para agentes autônomos.

Contexto:
Foi identificada a seguinte lacuna de capacidade ou padrão de falha no agente A³X:
GAP_DESCRIPTION_START
{gap_description}
GAP_DESCRIPTION_END

Logs Relevantes (se houver):
LOGS_START
{logs_section}
LOGS_END

Skills Atuais Disponíveis:
SKILLS_START
{current_skills_description}
SKILLS_END

Tarefa:
Proponha uma nova skill Python completa para preencher a lacuna descrita. A skill deve:
1.  Ser uma função `async def`.
2.  Aceitar `context: Context` como primeiro argumento, para acesso a logger, LLM, memória, etc.
3.  Usar o decorador `@skill` de `a3x.core.skills` com `name`, `description`, e `parameters` bem definidos.
4.  Os `parameters` no decorador devem ser um dicionário onde cada chave é o nome do parâmetro e o valor é outro dicionário com `{"type": TIPO, "description": "DESC"}`. Inclua tipos (`str`, `int`, `bool`, `List`, `Dict`, `Optional`) e descrições claras.
5.  Implementar a lógica principal da skill, incluindo logging (`context.logger`).
6.  Retornar um dicionário padronizado: `{"status": "success" | "error", "data": {... }}`.
7.  Incluir imports necessários.
8.  Focar em ser o mais atômica e reutilizável possível.

Responda SOMENTE com o bloco de código Python da nova skill. Não inclua explicações antes ou depois.
Exemplo de Formato:
```python
import logging
from typing import Dict, Any, Optional
from a3x.core.skills import skill
from a3x.core.context import Context

logger = logging.getLogger(__name__)

@skill(
    name="nome_da_nova_skill",
    description="Descrição clara do que a skill faz.",
    parameters={
        "context": {"type": Context, "description": "Contexto de execução."},
        "param1": {"type": str, "description": "Descrição do param1."},
        "param_opcional": {"type": Optional[int], "default": None, "description": "Descrição."}
    }
)
async def nome_da_nova_skill(context: Any, param1: str, param_opcional: Optional[int] = None) -> Dict[str, Any]:
    logger = context.logger
    logger.info(f"Executando skill 'nome_da_nova_skill' com param1: {{param1}}")
    try:
        # Lógica da skill aqui...
        result_data = f"Resultado processado de {{param1}}"
        logger.info("Skill executada com sucesso.")
        return {"status": "success", "data": {"resultado": result_data}}
    except Exception as e:
        logger.exception("Erro ao executar a skill:")
        return {"status": "error", "data": {"message": str(e)}}
```
"""

    # 3. Call LLM
    try:
        ctx.logger.info("Calling LLM to propose a new skill...")
        proposed_code = ""
        async for chunk in llm_interface.call_llm(messages=[{"role": "user", "content": prompt}], stream=True):
            proposed_code += chunk

        # Basic validation/cleanup
        if not proposed_code or not proposed_code.strip():
            ctx.logger.error("LLM returned empty response for skill proposal.")
            return {"status": "error", "message": "LLM did not propose any skill code."}

        # Extract code from markdown block if present
        proposed_code = proposed_code.strip()
        code_match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", proposed_code, re.DOTALL)
        if code_match:
            proposed_code = code_match.group(1).strip()
            ctx.logger.info("Extracted skill code from markdown block.")
        else:
            ctx.logger.debug("No markdown block found, assuming full response is code.")

        # Clean common leading comments
        lines = proposed_code.split('\n')
        cleaned_lines = []
        cleaned_something = False
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('# File:') or stripped_line.startswith('# -*- coding:'):
                cleaned_something = True
                continue # Skip this line
            cleaned_lines.append(line)
        
        if cleaned_something:
             proposed_code = '\n'.join(cleaned_lines).strip()
             ctx.logger.info("Removed common leading comment lines from generated code.")

        if not proposed_code:
            ctx.logger.error("Skill proposal resulted in empty code after potential extraction and cleaning.")
            return {"status": "error", "message": "LLM proposal resulted in empty code after cleaning."}

        ctx.logger.info("Successfully received and cleaned skill proposal from LLM.")
        
        # Syntax Validation
        try:
            ast.parse(proposed_code)
            ctx.logger.info("Generated code syntax check passed.")
        except SyntaxError as syntax_err:
            ctx.logger.error(f"Generated code has syntax errors: {syntax_err}")
            ctx.logger.error(f"Problematic Code:\n-------\n{proposed_code}\n-------")
            return {
                "status": "error", 
                "message": f"Generated code has syntax errors: {syntax_err}",
                "generated_code": proposed_code # Return the bad code for debugging
            }

        # --- Step 3: Save the proposed skill code --- #
        skill_filename = f"{skill_name_suggestion}.py"
        output_dir = os.path.join(PROJECT_ROOT, AUTO_GENERATED_SKILLS_DIR)
        os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
        file_path = os.path.join(output_dir, skill_filename)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(proposed_code)
            ctx.logger.info(f"Successfully saved proposed skill code to: {file_path}")
            return {
                "status": "success",
                "data": {
                    "file_path": file_path,
                    "skill_name": skill_name_suggestion,
                    "message": f"Skill '{skill_name_suggestion}' generated and saved."
                }
            }
        except IOError as io_err:
            ctx.logger.exception(f"Error writing proposed skill to file {file_path}:")
            return {"status": "error", "message": f"IOError saving skill code: {io_err}"}

    except Exception as e:
        ctx.logger.exception("Error during LLM call or processing for skill proposal:")
        return {"status": "error", "message": f"Failed to get skill proposal from LLM: {e}"}

    # --- Step 4: TODO - Generate Test Skeleton --- #
    ctx.logger.info("TODO: Implement test skeleton generation for the new skill.")
    # Placeholder for future test generation logic

    # --- Step 5: Return Summary --- #
    final_status = "success"
    summary_message = f"Proposal for skill processed. Code generated and saved."

    ctx.logger.info(summary_message)
    return {
        "status": final_status,
        "message": summary_message,
        "generated_skill_code": proposed_code
    } 