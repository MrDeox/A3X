# a3x/skills/code/execute_code.py
import logging
from typing import Dict, Optional

from a3x.core.skills import skill, SkillContext # Assuming SkillContext is the standard context type
from a3x.core.sandbox import execute_code_in_sandbox
from a3x.core.context import SharedTaskContext # For type hinting if needed

logger = logging.getLogger(__name__)

@skill(
    name="execute_code",
    description=(
        "Executes a given code snippet (currently Python only) in a secure sandbox environment. "
        "Prioritizes Firejail if available and enabled, otherwise uses direct execution after AST safety checks. "
        "Can resolve context placeholders like $LAST_READ_FILE."
    ),
    parameters={
        "code": {"type": str, "description": "The code snippet to execute."},
        "language": {"type": Optional[str], "default": "python", "description": "The programming language (default: python). Currently only supports python."},
        "timeout": {"type": Optional[int], "default": 60, "description": "Maximum execution time in seconds (default: 60)."},
    }
)
async def execute_code(
    context: SkillContext,
    code: str,
    language: str = "python",
    timeout: Optional[int] = 60 # Match sandbox default
) -> Dict:
    """
    Skill wrapper for the centralized code execution sandbox.

    Args:
        context: The skill execution context (provides shared_task_context).
        code: The code snippet to execute.
        language: The programming language (currently only python).
        timeout: Maximum execution time.

    Returns:
        A dictionary containing the execution result from the sandbox.
    """
    logger.info(f"Skill 'execute_code' called for language '{language}'. Delegating to sandbox.")

    # Extract shared context if available within SkillContext
    shared_context: Optional[SharedTaskContext] = getattr(context, 'shared_task_context', None)
    
    # Prepare skill context for sandbox logging
    sandbox_skill_context = {"skill_name": "execute_code"}
    if context and hasattr(context, 'to_dict'): # Pass context if possible
         sandbox_skill_context.update(context.to_dict())
    elif isinstance(context, dict):
         sandbox_skill_context.update(context)

    # Delegate execution to the sandbox function
    result = execute_code_in_sandbox(
        code=code,
        language=language,
        timeout=timeout,
        shared_context=shared_context,
        skill_context=sandbox_skill_context
    )
    
    logger.info(f"Sandbox execution result: {result.get('status')}")
    
    # Return the raw result dictionary from the sandbox
    # The caller (e.g., agent loop, fragment) can interpret this as needed.
    return result 