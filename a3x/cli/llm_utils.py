# a3x/cli/llm_utils.py
import logging
import asyncio
from typing import Optional, AsyncGenerator
from collections import namedtuple

# Import core components needed
try:
    from a3x.core.llm_interface import LLMInterface
except ImportError as e:
    print(f"[CLI LLM Utils Error] Failed to import LLMInterface: {e}")
    LLMInterface = None

logger = logging.getLogger(__name__)

# Define the SkillContext structure used by create_skill_execution_context
# Consider moving this to a central types definition if used elsewhere
SkillContext = namedtuple("SkillContext", ["logger", "llm_call", "is_test"])

async def direct_llm_call_wrapper(prompt: str, llm_url_override: Optional[str] = None) -> AsyncGenerator[str, None]:
    """Wrapper for LLMInterface.call_llm to format prompt and enable streaming.
       Yields response chunks.
    """
    if not LLMInterface:
        logger.error("LLMInterface class not available due to import error.")
        yield "[Error: LLM Interface not loaded]"
        return

    messages = [{"role": "user", "content": prompt}]
    logger.debug("Direct LLM Wrapper: Calling call_llm with stream=True")
    try:
        llm_interface = LLMInterface(base_url=llm_url_override)
        async for chunk in llm_interface.call_llm(messages=messages, stream=True):
            yield chunk
    except Exception as e:
        logger.error(f"Error in direct_llm_call_wrapper (streaming): {e}", exc_info=True)
        yield f"[LLM Call Error: {e}]"

def create_skill_execution_context(llm_url_override: Optional[str], logger_instance: logging.Logger) -> SkillContext:
    """Creates the context object needed for direct skill execution."""

    async def non_streaming_llm_call_wrapper(prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Wrapper for non-streaming LLM calls for the skill context."""
        if not LLMInterface:
            logger_instance.error("LLMInterface class not available due to import error.")
            return "[Error: LLM Interface not loaded]"

        messages = [{"role": "user", "content": prompt}]
        logger_instance.debug(
            f"Skill Context LLM Wrapper: Calling call_llm (prompt: {prompt[:100]}...)"
        )
        try:
            llm_interface = LLMInterface(base_url=llm_url_override)
            response = await llm_interface.call_llm(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False, # Ensure non-streaming
            )
            return response if isinstance(response, str) else str(response or "")
        except Exception as e:
            logger_instance.error(
                f"Error in skill context non_streaming_llm_call_wrapper: {e}", exc_info=True
            )
            return f"[LLM Call Error: {e}]"

    ctx = SkillContext(
        logger=logger_instance,
        llm_call=non_streaming_llm_call_wrapper,
        is_test=True # Assume direct skill run is for testing/debug
    )
    return ctx 