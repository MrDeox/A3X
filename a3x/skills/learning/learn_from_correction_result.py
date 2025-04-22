# a3x/skills/learning/learn_from_correction_result.py

import logging
import difflib
from typing import Dict, Any, Optional, List

# Core imports
from a3x.core.skills import skill
# Import context type for hinting (adjust if needed based on actual context used)
from a3x.core.agent import _ToolExecutionContext
# from a3x.core.memory_manager import MemoryManager # Placeholder for future integration

logger = logging.getLogger(__name__)

@skill(
    name="learn_from_correction_result",
    description="Learns from a successful code correction by storing the error, original/corrected code diff, and metadata.",
    parameters={
        "type": "object",
        "properties": {
            "stderr": {
                "type": "str",
                "description": "The standard error output from the *initial* failed sandbox execution."
            },
            "original_code": {
                "type": "str",
                "description": "The code content *before* the successful correction was applied."
            },
            "corrected_code": {
                "type": "str",
                "description": "The code content *after* the successful correction (which passed sandbox)."
            },
            "metadata": {
                "type": "object",
                "description": "Optional dictionary containing metadata like file_path, original_action, etc.",
                "default": {}
            }
            # Context is implicitly passed
        },
        "required": ["stderr", "original_code", "corrected_code"]
    }
)
async def learn_from_correction_result(
    context: _ToolExecutionContext, # Or appropriate context type
    stderr: str,
    original_code: str,
    corrected_code: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates a diff between original and corrected code and stores it
    semantically linked to the stderr from the initial failure.

    Args:
        context: The execution context.
        stderr: The stderr output from the initial failed execution.
        original_code: The code before correction.
        corrected_code: The code after correction (which passed).
        metadata: Optional metadata (e.g., file path).

    Returns:
        A dictionary with status and a message.
    """
    if metadata is None:
        metadata = {}

    file_path_info = f"for file '{metadata.get('file_path')}'" if metadata.get('file_path') else "(unknown file)"
    logger.info(f"Executing learn_from_correction_result {file_path_info}...")

    # --- Input Validation ---
    if not stderr or not isinstance(stderr, str):
        logger.error("Invalid or missing 'stderr' parameter.")
        return {"status": "error", "data": {"message": "stderr parameter is required and must be a string."}}
    if not isinstance(original_code, str): # Allow empty string? Yes. Check for None.
        logger.error("Invalid or missing 'original_code' parameter.")
        return {"status": "error", "data": {"message": "original_code parameter must be a string."}}
    if not isinstance(corrected_code, str):
        logger.error("Invalid or missing 'corrected_code' parameter.")
        return {"status": "error", "data": {"message": "corrected_code parameter must be a string."}}

    if original_code == corrected_code:
         logger.warning("Original and corrected code are identical. No diff to learn.")
         return {"status": "skipped", "data": {"message": "Original and corrected code are identical. Nothing learned."}}

    # --- Generate Diff ---
    diff_str = ""
    try:
        logger.debug("Generating unified diff...")
        diff_lines = difflib.unified_diff(
            original_code.splitlines(keepends=True),
            corrected_code.splitlines(keepends=True),
            fromfile=f"original_{metadata.get('file_path', 'code')}",
            tofile=f"corrected_{metadata.get('file_path', 'code')}",
            lineterm='\n'
        )
        diff_str = "".join(diff_lines)

        if not diff_str:
             # Should only happen if inputs were identical, caught above, but check defensively
             logger.warning("Diff generation resulted in empty string unexpectedly.")
             return {"status": "skipped", "data": {"message": "Diff was empty unexpectedly."}}

        logger.info(f"Successfully generated diff {file_path_info}. Length: {len(diff_str)}")
        logger.debug(f"Generated Diff:\n---\n{diff_str[:1000]}...\n---") # Log preview

    except Exception as e:
        logger.exception(f"Error generating diff {file_path_info}: {e}")
        return {"status": "error", "data": {"message": f"Failed to generate diff: {e}"}}

    # --- Store Learning (Simulated) ---
    # TODO: Integrate with actual MemoryManager / Vector Store
    logger.info("Simulating storage of learning data (stderr -> diff)...")
    try:
        # --- Placeholder for actual storage logic ---
        # Example using a hypothetical MemoryManager:
        # if hasattr(context, 'memory_manager') and isinstance(context.memory_manager, MemoryManager):
        #     memory_manager = context.memory_manager
        #     # Store stderr as the queryable text, and the diff/context as payload
        #     storage_payload = {
        #         "diff": diff_str,
        #         "original_code_preview": original_code[:500] + "...",
        #         "corrected_code_preview": corrected_code[:500] + "...",
        #         "metadata": metadata
        #     }
        #     await memory_manager.add_semantic_memory(
        #         text_to_embed=stderr, # Embed the error message
        #         payload=storage_payload,
        #         source="learn_from_correction_result",
        #         tags=["code_correction", "sandbox_failure", metadata.get('file_path', 'unknown_file')]
        #     )
        #     logger.info("Successfully stored learning data in semantic memory (placeholder).")
        # else:
        #     logger.warning("MemoryManager not found in context. Skipping actual storage.")
        # --- End Placeholder ---

        # Simulate success for now
        storage_message = "Learning data (stderr -> diff) processed for storage (simulation)."
        logger.info(storage_message)

        return {
            "status": "success",
            "data": {
                "message": f"Successfully generated diff and simulated storage {file_path_info}.",
                "diff_preview": diff_str[:500] + "..." # Include preview in result
            }
        }

    except Exception as e:
         logger.exception(f"Error during simulated storage {file_path_info}: {e}")
         return {"status": "error", "data": {"message": f"Failed during (simulated) storage: {e}"}} 