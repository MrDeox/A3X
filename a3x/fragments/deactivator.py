import asyncio
import logging
from typing import Dict, Any, Optional

from .base import BaseFragment, FragmentDef
from a3x.core.context import FragmentContext

logger = logging.getLogger(__name__)

class DeactivatorFragment(BaseFragment):
    """Listens for 'deactivate_fragment' directives and simulates the deactivation process."""

    def __init__(self, fragment_def: FragmentDef, tool_registry=None):
        super().__init__(fragment_def, tool_registry)
        # In a real implementation, this might need access to a shared fragment registry
        self._logger.info(f"[{self.get_name()}] Initialized.")

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        """Returns a description of this fragment's purpose."""
        return "Listens for 'deactivate_fragment' directives and simulates the process of removing fragments from the active pool."

    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """Handles incoming chat messages, looking for 'deactivate_fragment' directives."""
        msg_type = message.get("type")
        sender = message.get("sender")
        content = message.get("content")

        # Check for the specific directive
        if (
            msg_type == "ARCHITECTURE_SUGGESTION"
            and isinstance(content, dict)
            and content.get("type") == "directive"
            and content.get("action") == "deactivate_fragment"
        ):
            directive = content
            target_fragment_name = directive.get("target") # Name of the fragment to deactivate

            if not target_fragment_name:
                self._logger.warning(f"[{self.get_name()}] Received invalid 'deactivate_fragment' directive (missing target name): {directive}")
                return

            self._logger.info(f"[{self.get_name()}] Received 'deactivate_fragment' directive from {sender} for target fragment: {target_fragment_name}")
            await self._handle_deactivate_directive(directive, context)
        else:
            pass # Ignore other messages

    async def _handle_deactivate_directive(self, directive: Dict[str, Any], context: FragmentContext):
        """Simulates the deactivation of a fragment."""
        target_fragment_name = directive.get("target")
        result_status = "success" # Assume success for simulation
        result_summary = f"SIMULATED: Deactivated fragment '{target_fragment_name}'."
        result_details = "Fragment removed from active processing (simulation). No actual removal implemented yet."

        # --- Simulation Logic --- 
        self._logger.info(f"[{self.get_name()}] Simulating deactivation for fragment: {target_fragment_name}")
        # TODO: Implement actual deactivation logic here. This would involve:
        # 1. Accessing the FragmentManager's list of active dynamic fragments (or a shared registry).
        # 2. Finding the target fragment's task/instance.
        # 3. Cancelling the fragment's task (if applicable, like the dynamic dispatcher task).
        # 4. Removing the fragment from the active list.
        # -----------------------

        # Send result back via chat
        await self.broadcast_result_via_chat(context, directive, result_status, result_summary, result_details)

    async def broadcast_result_via_chat(self, context: FragmentContext, original_directive: Dict, status: str, summary: str, details: str):
        """Broadcasts the result of handling the deactivate_fragment directive."""
        result_message_content = {
            "type": "deactivator_result", # Specific type for this fragment's results
            "status": status,
            "target_fragment": original_directive.get("target", "unknown"), # Target is the fragment name
            "original_action": "deactivate_fragment",
            "summary": summary,
            "details": details,
            "original_directive": original_directive
        }
        try:
            await self.post_chat_message(
                message_type="deactivator_result", # Use the specific type
                content=result_message_content
            )
            self._logger.info(f"[{self.get_name()}] Broadcasted deactivator_result for target '{result_message_content['target_fragment']}': {status}")
        except Exception as e:
            self._logger.error(f"[{self.get_name()}] Failed to broadcast deactivator_result via chat: {e}") 