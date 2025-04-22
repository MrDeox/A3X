import logging
from typing import Dict, Optional, Any, Callable, Awaitable

# Assuming these are available in the environment
from ..core.professor_llm_fragment import ProfessorLLMFragment
from ..core.memory_bank import MemoryBank

logger = logging.getLogger(__name__)

async def handle_learn_from_professor(
    directive: Dict[str, Any],
    memory_bank: MemoryBank, 
    post_message_handler: Optional[Callable[..., Awaitable[None]]]
) -> Optional[Dict[str, Any]]:
    """Handles the 'learn_from_professor' directive logic."""

    professor_id = directive.get("professor_id")
    question = directive.get("question")
    # Optional: context_fragment_id = directive.get("context_fragment_id")

    if not professor_id or not isinstance(professor_id, str):
        logger.error("[A3X Bridge Handler - Learn] 'professor_id' missing or invalid.")
        return { "status": "error", "message": "'professor_id' missing or invalid" }
    if not question or not isinstance(question, str):
        logger.error("[A3X Bridge Handler - Learn] 'question' missing or invalid.")
        return { "status": "error", "message": "'question' missing or invalid" }

    professor_fragment = memory_bank.load(professor_id)
    if not professor_fragment:
        logger.error(f"[A3X Bridge Handler - Learn] Professor fragment '{professor_id}' not found.")
        return { "status": "error", "message": f"Professor fragment '{professor_id}' not found." }
    if not isinstance(professor_fragment, ProfessorLLMFragment):
        logger.error(f"[A3X Bridge Handler - Learn] Fragment '{professor_id}' is not a ProfessorLLMFragment.")
        return { "status": "error", "message": f"Fragment '{professor_id}' is not a ProfessorLLMFragment." }
    if not professor_fragment.is_active:
         logger.warning(f"[A3X Bridge Handler - Learn] Professor '{professor_id}' is not active. Skipping.")
         return { "status": "skipped", "message": f"Professor '{professor_id}' is not active.", "professor_id": professor_id }

    try:
        logger.info(f"[A3X Bridge Handler - Learn] Consulting Professor '{professor_id}' with question: '{question[:100]}...'")
        response_text = await professor_fragment.ask_llm(question)

        if response_text:
            logger.info(f"[A3X Bridge Handler - Learn] Got response from '{professor_id}'. Posting for interpretation.")
            if post_message_handler:
                interpretation_request = {
                    "type": "interpret_text",
                    "text": response_text,
                    "origin": f"Response from learn_from_professor('{professor_id}')"
                }
                await post_message_handler(
                    message_type="a3l_directive",
                    content=interpretation_request,
                    target_fragment="ki_main_instance"
                )
                return { "status": "success", "message": f"Professor '{professor_id}' consulted. Response posted for interpretation.", "professor_id": professor_id }
            else:
                logger.error("[A3X Bridge Handler - Learn] Cannot post response: post_message_handler missing.")
                return { "status": "success_no_post", "message": f"Professor '{professor_id}' consulted, but response could not be posted.", "response": response_text, "professor_id": professor_id }
        else:
            logger.warning(f"[A3X Bridge Handler - Learn] Professor '{professor_id}' returned an empty response.")
            return { "status": "success", "message": f"Professor '{professor_id}' returned an empty response.", "professor_id": professor_id }

    except Exception as e:
        logger.error(f"[A3X Bridge Handler - Learn] Exception during consultation: {e}", exc_info=True)
        return { "status": "error", "message": f"Error consulting professor '{professor_id}': {e}", "professor_id": professor_id } 