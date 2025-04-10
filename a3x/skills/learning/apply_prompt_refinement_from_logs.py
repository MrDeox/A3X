"""
Skill para aplicar automaticamente refinamentos ao prompt de outra skill,
baseado nas sugestões geradas pela análise de logs de reflexão.
"""

import logging
import re
import os
import datetime
from typing import Dict, Any, Optional, AsyncGenerator, List
import ast

# Core imports
from a3x.core.tools import skill
from a3x.skills.core.call_skill_by_name import call_skill_by_name
from a3x.core.llm_interface import call_llm

# Assume SkillContext e outras skills (como learn_from_reflection_logs) são acessíveis
# através do contexto ou import direto se necessário.

# Path to the target skill file
# TODO: Make this configurable or discoverable?
TARGET_SKILL_FILE = "skills/simulate/simulate_decision_reflection.py"
PROMPT_START_MARKER = "# --- START SIMULATION PROMPT ---"
PROMPT_END_MARKER = "# --- END SIMULATION PROMPT ---"

logger = logging.getLogger(__name__)

@skill(
    name="apply_prompt_refinement_from_logs",
    description="Refatora o prompt da skill simulate_decision_reflection com base nas sugestões extraídas dos logs de reflexão.",
    parameters={}
)
async def apply_prompt_refinement_from_logs(ctx) -> Dict[str, Any]:
    """
    Automatically refines the prompt of the simulate_decision_reflection skill
    based on suggestions extracted from recent reflection logs.

    Args:
        ctx: The skill execution context (must provide access to logger, llm_call,
             and potentially other skills like learn_from_reflection_logs and file I/O).

    Returns:
        A dictionary indicating success or failure and the new prompt if successful.
    """
    ctx.logger.info("Starting prompt refinement based on learning logs.")

    # --- Step 1: Get Prompt Improvement Suggestions --- #
    ctx.logger.info("Running learn_from_reflection_logs to get suggestions...")
    try:
        # <<< MODIFIED: Use call_skill_by_name exclusively >>>
        learn_skill_result = await call_skill_by_name(
            ctx,
            skill_name="learn_from_reflection_logs",
            skill_args={"n_logs": 5} # Use a reasonable default or make configurable?
        )

        # Check for errors from call_skill_by_name
        if learn_skill_result.get("status") == "error" or "error" in learn_skill_result:
            error_detail = learn_skill_result.get('error', 'Unknown error from learn_from_reflection_logs')
            ctx.logger.error(f"Failed to get insights from logs via call_skill_by_name: {error_detail}")
            # Return the error dict directly from the failed skill call
            return learn_skill_result

        # Extract insights from the successful result data
        # Assuming the result structure is something like {"status": "success", "data": {"insights_from_logs": "..."}}
        # Adjust based on the actual return structure of learn_from_reflection_logs
        insights_text = learn_skill_result.get("data", {}).get("insights_from_logs", "")
        if not insights_text:
            ctx.logger.error("No insights text returned from learn_from_reflection_logs call.")
            return {"status": "error", "error": "No insights text returned from learn_from_reflection_logs call."}

        # Extract suggestions (simple regex approach)
        suggestions_match = re.search(r"- Melhorias de Prompt Sugeridas(?:.*?):\s*(.*?)(?:\n- |$)", insights_text, re.IGNORECASE | re.DOTALL)
        if not suggestions_match:
            ctx.logger.warning("Could not find 'Melhorias de Prompt Sugeridas' section in the insights.")
            return {"status": "error", "error": "Seção de sugestões de prompt não encontrada nos logs."}

        prompt_suggestions = suggestions_match.group(1).strip()
        if not prompt_suggestions:
             ctx.logger.warning("'Melhorias de Prompt Sugeridas' section is empty.")
             return {"status": "error", "error": "Seção de sugestões de prompt está vazia."}

        ctx.logger.info(f"Extracted prompt suggestions:\n{prompt_suggestions}")

    except Exception as e:
        ctx.logger.exception("Error during learn_from_reflection_logs skill call or suggestion extraction:")
        return {"status": "error", "error": f"Falha ao obter/processar sugestões via call_skill_by_name: {e}"}

    # --- Step 2: Generate New Prompt using LLM --- #
    ctx.logger.info("Generating new prompt based on suggestions...")
    refinement_prompt = f"""Você é responsável por refatorar o prompt da skill 'simulate_decision_reflection'.
Abaixo estão sugestões de melhoria extraídas de logs de reflexão anteriores:

{prompt_suggestions}

Com base nisso, gere uma NOVA versão completa do prompt da skill.

Regras:
- O prompt deve instruir a IA (Arthur) a seguir um processo de Chain-of-Thought com etapas claras (ex: Análise do Contexto, Raciocínio Interno, Critérios, Decisão, Justificativa).
- O prompt deve incorporar as sugestões fornecidas da melhor forma possível.
- Não escreva introduções nem comentários.
- Responda APENAS com o texto completo do novo prompt, pronto para ser usado no código Python como uma f-string (portanto, não use f-string dentro da sua resposta). Certifique-se de que variáveis como {{user_input}} e {{examples_text}} estejam corretamente formatadas para substituição posterior (use chaves simples {{}}).

Novo prompt:
"""

    new_prompt_text = ""
    try:
        # Replaced safe_llm_call with standard call_llm and try/except
        ctx.logger.info("Calling LLM for prompt refinement (streaming)...")
        analysis_response = ""
        # Assuming call_llm expects a list of messages
        async for chunk in call_llm(messages=[{"role": "user", "content": refinement_prompt}], stream=True): # Consider adding timeout if supported/needed
            analysis_response += chunk
        new_prompt_text = analysis_response.strip() # Assign stripped response

        # Check for empty response
        if not new_prompt_text:
            ctx.logger.error("LLM failed to generate a new prompt (empty response).")
            return {"status": "error", "error": "LLM não gerou um novo prompt."}

        # Clean up potential LLM artifacts like leading/trailing quotes or markdown backticks
        new_prompt_text = new_prompt_text.strip('\"' + "\'" + "`") # Clean after checking empty
        ctx.logger.info("Successfully generated new prompt candidate.")
        # ctx.logger.debug(f"New prompt generated:\\n{new_prompt_text}") # Optional: log the full new prompt

        # --- ADICIONADO: Validar Sintaxe do Prompt Gerado --- #
        try:
            # Construir um snippet de código Python mínimo usando o prompt gerado
            # para verificar se ele seria válido dentro de uma f-string.
            code_to_validate = f'simulated_prompt = f"""{new_prompt_text}"""'
            ast.parse(code_to_validate)
            ctx.logger.info("Generated prompt passed syntax validation.")
        except SyntaxError as e:
            ctx.logger.error(f"Generated prompt failed Python syntax validation: {e}")
            ctx.logger.debug(f"Invalid prompt content:\n{new_prompt_text}") # Log o prompt inválido para debug
            return {
                "status": "error",
                "error": f"Generated prompt is not valid Python syntax: {e}",
                "invalid_prompt_snippet": new_prompt_text[:200] + "..." # Retorna um trecho para análise
            }
        # --- FIM: Validar Sintaxe --- #

    except Exception as e:
        ctx.logger.exception("Error during LLM call for prompt refinement:")
        return {"error": f"Falha ao gerar novo prompt via LLM: {e}"}

    # --- Step 3: Read Target File and Replace Prompt --- #
    ctx.logger.info(f"Applying new prompt to {TARGET_SKILL_FILE}...")
    try:
        # Assume ctx provides file reading/writing capabilities
        # Placeholder: Replace with actual mechanism if ctx doesn't have these methods
        # Example: Use FileManager skill if available

        # Read existing content using read_file skill
        ctx.logger.info(f"Reading target file {TARGET_SKILL_FILE} using read_file skill...")
        target_content_dict = await call_skill_by_name(
            ctx,
            skill_name="read_file",
            skill_args={"path": TARGET_SKILL_FILE}
        )
        # Error handling for read_file skill call
        if target_content_dict.get("status") == "error":
             ctx.logger.error(f"Failed to read target file {TARGET_SKILL_FILE} via skill: {target_content_dict.get('error', 'Unknown error')}")
             return {"error": f"Failed to read target file via skill: {target_content_dict.get('error', 'Unknown error')}"}
        if "content" not in target_content_dict.get("data", {}):
             ctx.logger.error(f"Read_file skill succeeded but did not return 'content' for {TARGET_SKILL_FILE}.")
             return {"error": "Read_file skill did not return content."}
        target_content = target_content_dict["data"]["content"]

        # Extract old prompt and check markers
        match = re.search(f"{re.escape(PROMPT_START_MARKER)}(.*?){re.escape(PROMPT_END_MARKER)}", target_content, re.DOTALL)
        if not match:
            ctx.logger.error(f"Could not find prompt markers in {TARGET_SKILL_FILE}.")
            return {"error": f"Marcadores de prompt não encontrados em {TARGET_SKILL_FILE}."}
        old_prompt = match.group(1).strip()

        # Create backup (optional but recommended) - Using write_file skill
        backup_filename = f"{TARGET_SKILL_FILE}.bak_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        ctx.logger.info(f"Creating backup of old prompt in {backup_filename} using write_file skill...")
        backup_write_result = await call_skill_by_name(
            ctx,
            skill_name="write_file",
            skill_args={"path": backup_filename, "content": old_prompt, "overwrite": True}
        )
        if backup_write_result.get("status") == "error":
             ctx.logger.warning(f"Failed to create backup file {backup_filename} via skill: {backup_write_result.get('error', 'Unknown error')}. Proceeding without backup.")

        # Prepare the new prompt content (ensure proper indentation/formatting if needed)
        # For now, assume new_prompt_text is ready
        # Ensure leading/trailing newlines match the original marker placement
        new_prompt_section = f"\n{new_prompt_text}\n        "

        # Replace the old prompt section with the new one
        new_target_content = target_content.replace(match.group(1), new_prompt_section)

        # Add automatic modification comment (optional)
        modification_comment = f"\n# Prompt automatically refined by apply_prompt_refinement_from_logs on {datetime.datetime.now().isoformat()}\n"
        # Find a suitable place to add it, e.g., after the END marker
        new_target_content = new_target_content.replace(PROMPT_END_MARKER, PROMPT_END_MARKER + modification_comment)

        # Write the modified content back to the file using write_file skill
        ctx.logger.info(f"Writing updated content to {TARGET_SKILL_FILE} using write_file skill...")
        write_result = await call_skill_by_name(
            ctx,
            skill_name="write_file",
            skill_args={"path": TARGET_SKILL_FILE, "content": new_target_content, "overwrite": True}
        )

        if write_result.get("status") == "error":
            ctx.logger.error(f"Failed to write updated file {TARGET_SKILL_FILE} via skill: {write_result.get('error', 'Unknown error')}")
            return {"error": f"Failed to write updated file via skill: {write_result.get('error', 'Unknown error')}"}

        ctx.logger.info("Prompt successfully updated.")
        return {
            "status": "Prompt atualizado com sucesso.",
            "new_prompt": new_prompt_text # Return the clean new prompt
        }

    except Exception as e:
        ctx.logger.exception(f"Error during file operation for {TARGET_SKILL_FILE}:")
        return {"error": f"Falha ao atualizar arquivo da skill: {e}"}

# TODO:
# - Implement actual ctx.run_skill or equivalent mechanism.
# - Implement actual ctx.read_file / ctx.write_file or use FileManager skill.
# - Refine error handling and logging.
# - Consider more robust prompt extraction/replacement if markers fail. 