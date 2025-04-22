# LLM desativado. Este fragmento não deve mais acessar modelos. Toda cognição é feita pelo A³Net.
import logging
from typing import Dict, Any, Optional, List

from a3x.core.skills import skill
from a3x.core.context import Context

logger = logging.getLogger(__name__)

@skill(
    name="generate_module_from_directive",
    description="Generates Python module code based on a specific directive or task description.",
    parameters={
        "directive": {"type": str, "description": "Clear, concise description of the desired module's functionality and goals."},
        "target_path": {"type": str, "description": "Target file path for the generated module (e.g., 'a3x/utils/log_parser.py')."},
        "example_code": {"type": str, "optional": True, "description": "Optional example code snippet to guide generation."}
    }
)
async def generate_module_from_directive(context: Context, directive: str, target_path: str, example_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates Python code for a new module based on a directive using an LLM.

    Args:
        context: The Context containing the LLM interface and other resources.
        directive: The architectural directive.
        target_path: The target path for the new module file.
        example_code: Optional example code snippet.

    Returns:
        A dictionary containing the status, path, and generated content.
    """
    if not hasattr(context, 'llm_interface') or not context.llm_interface:
        logger.error("LLM interface not available in context.")
        return {"status": "error", "reason": "LLM interface not configured in Context."}

    prompt_template = f"""
    Objective: Generate the complete Python code content for a new module based on the following architectural directive.

    Directive: {directive}
    Suggested Path: {target_path}
    Optional Example Code Hint: {example_code or 'None'}

    Instructions:
    1. Generate the full Python code for the file specified by the Suggested Path.
    2. Ensure the code directly addresses the Directive.
    3. Include necessary imports.
    4. Add clear docstrings for the module and any functions/classes.
    5. Follow standard Python best practices (PEP 8).
    6. Keep the code modular and focused on the directive's goal.
    7. Avoid unnecessary external dependencies unless explicitly implied by the directive.
    8. Only output the raw Python code, starting with imports or module docstring. Do not include any preamble, explanation, or markdown formatting like ```python ... ```.

    Python Code Content:
    """

    try:
        logger.info(f"Generating module content for directive: '{directive}' (Target: {target_path})")
        # Use non-streaming call to get the full content at once
        # response_content = ""
        # async for chunk in context.llm_interface.call_llm(
        #     messages=[{"role": "user", "content": prompt_template}], 
        #     stream=False
        # ):
        #      response_content += chunk

        # --- Refactoring required ---
        logger.error("Direct LLM call in generate_module_from_directive is deprecated. Refactor needed.")
        return {"status": "error", "reason": "Direct LLM call in skill is deprecated. Use A3L command via A3Net."}
        # --- End Refactoring ---

        # Basic validation (can be improved)
        # if not response_content or not response_content.strip():
        #      logger.warning("LLM returned empty content for module generation.")
        #      return {"status": "error", "reason": "LLM returned empty content."}
        # 
        # generated_code = response_content.strip()
        # logger.info(f"Successfully generated code for {target_path}")
        # 
        # return {
        #     "status": "success",
        #     "path": target_path,
        #     "content": generated_code,
        # }

    except Exception as e:
        logger.exception(f"Error generating module from directive for {target_path}:")
        return {"status": "error", "reason": str(e)} 