# a3x/skills/propose_skill_from_gap.py
import logging
import json
from typing import Dict, Any, Optional
import re
import traceback

from a3x.core.skills import skill
from a3x.core.llm_interface import LLMInterface
from a3x.core.context import _ToolExecutionContext # Or Context if appropriate
from pathlib import Path

logger = logging.getLogger(__name__)

# Updated path
GENERATED_SKILLS_DIR = Path("a3x/skills/auto_generated")
GENERATED_SKILLS_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists

@skill(
    name="propose_skill_from_gap",
    description="Attempts to generate Python code for a new skill based on a provided description of its requirements and purpose.",
    parameters={
        # "ctx": {"type": _ToolExecutionContext, "description": "Execution context."},
        "skill_description": {"type": str, "description": "Detailed natural language description of the desired skill (functionality, inputs, outputs, potential libraries)."},
        "skill_name_suggestion": {"type": str, "description": "A suggested snake_case name for the new skill function."},
        # "failure_context": {"type": str, "optional": True, "description": "Optional context about the task failure that prompted this skill generation."},
    }
)
async def propose_skill_from_gap(
    ctx: _ToolExecutionContext,
    skill_description: str,
    skill_name_suggestion: str,
    # failure_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Uses an LLM to generate Python code for a new skill.

    Receives a description of the required skill, generates the corresponding
    Python code (including the @skill decorator, docstrings, and type hints),
    validates the syntax, and saves it to the generated skills directory.

    Args:
        ctx: The execution context, providing access to LLMInterface, logger, etc.
        skill_description: A detailed description of what the new skill should do.
        skill_name_suggestion: A suggested Pythonic name for the skill function.
        # failure_context: Optional context about why this skill is needed.

    Returns:
        A dictionary containing:
        - status: 'success' or 'error'.
        - data: 
            - If success: {'file_path': 'path/to/new_skill.py', 'skill_name': 'new_skill_name', 'message': 'Skill code generated and saved.'}
            - If error: {'message': 'Error description.'}
    """
    logger.info(f"Attempting to generate skill '{skill_name_suggestion}' based on description: {skill_description[:100]}...")

    if not ctx.llm_interface:
        logger.error("LLMInterface not available in context.")
        return {"status": "error", "data": {"message": "LLMInterface missing from context."}}

    # --- 1. Construct the Prompt for the LLM ---
    # Define the example structure separately to avoid f-string parsing issues
    example_code_structure = f"""
```python
# File: a3x/skills/generated/{skill_name_suggestion}.py (DO NOT INCLUDE THIS LINE IN OUTPUT)
import logging
from typing import Dict, Any, Optional # Add other imports if needed

from a3x.core.skills import skill
from a3x.core.context import _ToolExecutionContext

logger = logging.getLogger(__name__)

@skill(
    name=\"{skill_name_suggestion}\",
    description=\"<Generated concise description>\",
    parameters={{ # Parameters based on generated function signature (excluding ctx)
        \"arg1\": {{ \"type\": \"string\", \"description\": \"...\" }},
        # ... other args
    }}
)
async def {skill_name_suggestion}(ctx: _ToolExecutionContext, arg1: str, ...) -> Dict[str, Any]:
    \"\"\"
    <Generated detailed docstring>

    Args:
        ctx: Execution context.
        arg1: ...
        ...
    
    Returns:
        Dict[str, Any]: {{'status': 'success'/'error', 'data': {{...}}}}
    \"\"\"
    logger.info(f\"Executing skill: {skill_name_suggestion}\")
    try:
        # --- Implement skill logic here ---
        result_data = {{}}
        # ...
        return {{ \"status\": \"success\", \"data\": result_data }}
    except Exception as e:
        logger.error(f\"Error in skill {skill_name_suggestion}: {{e}}\", exc_info=True)
        return {{ \"status\": \"error\", \"data\": {{ \"message\": f\"Skill execution failed: {{e}}\" }} }}

```
""" # <<< End of example_code_structure

    generation_prompt = f"""
You are an expert Python programmer specializing in creating AI agent skills.
Your task is to generate the complete Python code for a new skill based on the following requirements.

**Desired Skill Name:** `{skill_name_suggestion}`
**Desired Functionality:** 
{skill_description}

**Instructions:**
1.  Generate a complete Python async function named `{skill_name_suggestion}`.
2.  The function MUST accept `ctx: _ToolExecutionContext` as its first argument. Define other arguments based ONLY on the description.
3.  Include the `@skill` decorator above the function definition. The `name` in the decorator MUST be `{skill_name_suggestion}`. Extract a concise `description` from the provided functionality description. Generate a basic `parameters` dictionary reflecting the function's arguments (excluding ctx).
4.  Add comprehensive docstrings explaining what the function does, its arguments (`Args:`), and what it returns (`Returns:`). The return type hint MUST be `-> Dict[str, Any]` and the function should return a dictionary with 'status' ('success' or 'error') and 'data'.
5.  Include necessary standard Python imports (e.g., `logging`, `json`, `typing`). Do NOT import `a3x` modules directly unless absolutely necessary and specified in the description.
6.  Implement the core logic described in the "Desired Functionality". Handle potential errors gracefully within the function and return an appropriate error status/message.
7.  Ensure the generated code is syntactically correct Python.
8.  Output ONLY the raw Python code for the skill. Do NOT include ```python or any other text before or after the code.

**Example Structure:**
{example_code_structure}

**Now, generate the Python code for the skill `{skill_name_suggestion}` based *only* on the description provided above.** 
"""

    llm_messages = [{"role": "user", "content": generation_prompt}]

    # --- 2. Call LLM to Generate Code ---
    generated_code = ""
    try:
        logger.info("Calling LLM to generate skill code...")
        # Use a lower temperature for more predictable code generation?
        async for chunk in ctx.llm_interface.call_llm(messages=llm_messages, stream=False, temperature=0.3):
            generated_code += chunk
        
        generated_code = generated_code.strip() # Remove leading/trailing whitespace

        if not generated_code:
             raise ValueError("LLM returned empty code.")
        
        # Log the raw response separately to avoid f-string issues
        logger.debug("LLM Raw Response (Generated Code) received:")
        logger.debug(f"\n-------\n{generated_code}\n-------")

        # <<< ADDED: Clean potential markdown fences and comments >>>
        # Remove ```python ... ``` fences
        cleaned_code = re.sub(r'^```python\n(?P<code>.*)\n```$', r'\g<code>', generated_code, flags=re.DOTALL | re.MULTILINE)
        # Remove potential leading comment like '# File: ...'
        cleaned_code = re.sub(r'^# File:.*\n', '', cleaned_code, flags=re.MULTILINE).strip()
        
        if not cleaned_code:
             logger.warning("Code became empty after cleaning markdown/comments.")
             # Decide how to handle: error or proceed with empty?
             # Let's error out for now.
             raise ValueError("Cleaned code is empty.")
             
        # Log if cleaning actually changed the code
        if cleaned_code != generated_code:
            logger.debug("Code after cleaning markdown/comments:")
            logger.debug(f"\n-------\n{cleaned_code}\n-------")
        else:
            logger.debug("No markdown/comment cleaning needed for generated code.")
        # <<< END ADDED >>>

    except Exception as e:
        logger.error(f"Error during LLM call for skill generation: {e}", exc_info=True)
        return {"status": "error", "data": {"message": f"LLM call failed: {e}"}}

    # --- 3. Validate Generated Code (Basic Syntax Check) ---
    try:
        compile(cleaned_code, '<string>', 'exec')
        logger.info("Generated code passed basic Python syntax validation.")
    except SyntaxError as e:
        # Log the error message and the problematic code separately
        error_message = f"Generated code has syntax errors: {e}"
        logger.error(error_message)
        # <<< MODIFIED: Log cleaned_code on error >>>
        logger.error(f"Problematic (cleaned) Code:\n-------\n{cleaned_code}\n-------")
        # Optionally: Try to self-correct? For now, just fail.
        return {"status": "error", "data": {"message": error_message, "generated_code": cleaned_code}}
    except Exception as e:
        logger.error(f"Error during code validation: {e}", exc_info=True)
        # <<< MODIFIED: Return cleaned_code on error >>>
        return {"status": "error", "data": {"message": f"Code validation failed: {e}", "generated_code": cleaned_code}}
        
    # --- 4. Save Code to File ---
    file_path = GENERATED_SKILLS_DIR / f"{skill_name_suggestion}.py"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_code)
        logger.info(f"Successfully saved generated skill code to: {file_path}")
        
        # TODO: Trigger dynamic skill registration here or notify a separate mechanism
        # For now, just return success and the path.

        return {
            "status": "success",
            "data": {
                "message": f"Skill '{skill_name_suggestion}' generated and saved.",
                "file_path": str(file_path),
                "skill_name": skill_name_suggestion # Return the name used
            }
        }
    except IOError as e:
        logger.error(f"Failed to write generated skill to file {file_path}: {e}", exc_info=True)
        return {"status": "error", "data": {"message": f"Failed to save skill file: {e}"}}
    except Exception as e:
        logger.error(f"Unexpected error saving skill file: {e}", exc_info=True)
        return {"status": "error", "data": {"message": f"Unexpected error saving skill: {e}"}} 