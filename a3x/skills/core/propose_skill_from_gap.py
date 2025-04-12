import logging
import json
import os
import asyncio
from typing import Dict, Any, Optional

# Core imports
from a3x.core.skills import skill
# from a3x.core.context import SkillContext # REMOVED, USE Any
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
    description="Recebe uma proposta de nova skill (identificada por análise de logs ou via arquivo) e tenta gerar/salvar o código Python correspondente.",
    parameters={}
)
async def propose_skill_from_gap(
    ctx: Any,
    **kwargs
) -> Dict[str, Any]:
    """
    Receives a structured proposal for a new skill, generated from log analysis
    (or provided via file/kwargs), and attempts to generate and save the
    corresponding Python skill code. Arguments can be passed directly or via
    a JSON file specified by 'skill_args_file'.
    """
    skill_args = kwargs.copy() # Start with provided kwargs
    skill_args_file = skill_args.pop('skill_args_file', None) # Remove file path if present

    if skill_args_file:
        ctx.logger.info(f"Loading skill arguments from file: {skill_args_file}")
        try:
            with open(skill_args_file, 'r') as f:
                file_args = json.load(f)
            if not isinstance(file_args, dict):
                raise ValueError("JSON file content must be a dictionary.")
            # Merge file args with kwargs, giving file args precedence
            skill_args = {**kwargs, **file_args}
            ctx.logger.debug(f"Loaded arguments from {skill_args_file}: {skill_args}")
        except FileNotFoundError:
            ctx.logger.error(f"Skill arguments file not found: {skill_args_file}")
            return {"status": "error", "message": f"Skill arguments file not found: {skill_args_file}", "actions_taken": []}
        except (json.JSONDecodeError, ValueError) as e:
            ctx.logger.error(f"Error reading or parsing skill arguments file {skill_args_file}: {e}")
            return {"status": "error", "message": f"Invalid skill arguments file {skill_args_file}: {e}", "actions_taken": []}

    # --- Extract arguments (with defaults) ---
    skill_name = skill_args.get('skill_name')
    reason = skill_args.get('reason')
    suggestion_description = skill_args.get('suggestion_description')
    parameters_json = skill_args.get('parameters_json', "{}")
    example_usage = skill_args.get('example_usage', "")
    source_analysis_log_preview = skill_args.get('source_analysis_log_preview', "{}")
    source_skill = skill_args.get('source_skill', "unknown")

    # --- Basic Validation ---
    required_args = ['skill_name', 'reason', 'suggestion_description']
    missing_args = [arg for arg in required_args if arg not in skill_args or not skill_args[arg]]
    if missing_args:
        ctx.logger.error(f"Missing required arguments for propose_skill_from_gap: {missing_args}")
        return {"status": "error", "message": f"Missing required arguments: {missing_args}", "actions_taken": []}

    ctx.logger.info(f"Received proposal for new skill '{skill_name}' from '{source_skill}'.")
    ctx.logger.debug(f"Reason: {reason}")
    ctx.logger.debug(f"Description: {suggestion_description}")
    ctx.logger.debug(f"Parameters JSON: {parameters_json}")
    ctx.logger.debug(f"Example Usage: {example_usage}")
    ctx.logger.debug(f"Source Log Preview: {source_analysis_log_preview}")

    actions_taken = []
    generated_code = None
    saved_filepath = None

    # --- Step 1: Parse Parameters --- (Optional, but good practice)
    try:
        parameters_dict = json.loads(parameters_json)
        # You could perform validation on parameters_dict here if needed
    except json.JSONDecodeError:
        ctx.logger.error(f"Failed to parse parameters_json for skill '{skill_name}'. Assuming empty params.")
        parameters_dict = {}
        parameters_json = "{}" # Ensure consistency

    # --- Step 2: Generate Skill Code using LLM ---
    ctx.logger.info(f"Attempting to generate Python code for skill: {skill_name}")
    code_gen_prompt = CODE_GENERATION_PROMPT_TEMPLATE.format(
        skill_name=skill_name,
        reason=reason,
        suggestion_description=suggestion_description,
        parameters_json=parameters_json,
        example_usage=example_usage
    )

    try:
        # Call LLM for code generation
        # Add timeout for safety
        generated_code = await asyncio.wait_for(ctx.llm_call(code_gen_prompt), timeout=180.0) # 3 min timeout

        if not generated_code or not generated_code.strip():
             raise ValueError("LLM returned empty code.")

        # Basic validation: check if it looks like Python code (very rudimentary)
        if not ("def " in generated_code and "@skill" in generated_code):
            ctx.logger.warning(f"Generated code for '{skill_name}' might be incomplete or invalid. Proceeding anyway.")
            # Consider raising an error here for stricter validation

        ctx.logger.info(f"Successfully generated code for skill '{skill_name}'. Length: {len(generated_code)} chars.")
        actions_taken.append({"action": "generate_code", "status": "success"})

    except asyncio.TimeoutError:
         ctx.logger.error(f"LLM call timed out during code generation for skill '{skill_name}'.")
         actions_taken.append({"action": "generate_code", "status": "error", "message": "Timeout"})
         return {"status": "error", "message": "LLM timeout during code generation.", "actions_taken": actions_taken}
    except Exception as e:
        ctx.logger.exception(f"Error during LLM code generation for skill '{skill_name}':")
        actions_taken.append({"action": "generate_code", "status": "error", "message": str(e)})
        return {"status": "error", "message": f"LLM error during code generation: {e}", "actions_taken": actions_taken}

    # --- Step 3: Save Generated Code --- #
    if generated_code:
        # Ensure the target directory exists
        try:
            os.makedirs(AUTO_GENERATED_SKILLS_DIR, exist_ok=True)
            ctx.logger.debug(f"Ensured directory exists: {AUTO_GENERATED_SKILLS_DIR}")
        except OSError as e:
            ctx.logger.error(f"Failed to create directory {AUTO_GENERATED_SKILLS_DIR}: {e}")
            actions_taken.append({"action": "save_code", "status": "error", "message": f"Failed to create directory: {e}"})
            return {"status": "error", "message": f"Failed to create auto-generation directory: {e}", "actions_taken": actions_taken}

        # Construct the file path
        # Sanitize skill_name slightly for filename, although @skill handles registration name
        safe_filename = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in skill_name) + ".py"
        target_filepath = os.path.join(AUTO_GENERATED_SKILLS_DIR, safe_filename)

        ctx.logger.info(f"Attempting to save generated code to: {target_filepath}")

        try:
            # Use the 'write_file' skill via call_skill_by_name
            write_result = await call_skill_by_name(
                ctx,
                skill_name="write_file", # Assuming 'write_file' is the registered name for FileManagerSkill.write_file
                skill_args={
                    "path": target_filepath,
                    "content": generated_code,
                    "overwrite": True # Overwrite if it somehow exists?
                }
            )

            if write_result.get("status") == "success":
                ctx.logger.info(f"Successfully saved generated skill '{skill_name}' to {target_filepath}")
                saved_filepath = target_filepath
                actions_taken.append({"action": "save_code", "status": "success", "filepath": saved_filepath})
            else:
                error_msg = write_result.get("data", {}).get("message", "Unknown error from write_file skill")
                ctx.logger.error(f"Failed to save generated skill using 'write_file': {error_msg}")
                actions_taken.append({"action": "save_code", "status": "error", "message": error_msg})
                # Return error, as saving failed
                return {"status": "error", "message": f"Failed to save generated code: {error_msg}", "actions_taken": actions_taken}

        except Exception as e:
            ctx.logger.exception(f"Unexpected error calling 'write_file' skill for {target_filepath}:")
            actions_taken.append({"action": "save_code", "status": "error", "message": f"Unexpected error calling write_file: {e}"})
            return {"status": "error", "message": f"Unexpected error calling write_file: {e}", "actions_taken": actions_taken}

    # --- Step 4: TODO - Generate Test Skeleton --- #
    ctx.logger.info("TODO: Implement test skeleton generation for the new skill.")
    # Placeholder for future test generation logic

    # --- Step 5: Return Summary --- #
    final_status = "success" if saved_filepath else "error"
    summary_message = f"Proposal for skill '{skill_name}' processed. Code generated and saved to {saved_filepath}." if final_status == "success" else f"Proposal for skill '{skill_name}' processed, but failed during code generation or saving."

    ctx.logger.info(summary_message)
    return {
        "status": final_status,
        "message": summary_message,
        "actions_taken": actions_taken,
        "generated_skill_filepath": saved_filepath
    } 