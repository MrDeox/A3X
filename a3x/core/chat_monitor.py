# a3x/core/chat_monitor.py

import asyncio
import logging
from typing import Dict, Any
from pathlib import Path # Import Path

# Import necessary context/base classes using relative paths if needed, or full paths
# Adjust imports based on your project structure
from .context import SharedTaskContext, FragmentContext # Added FragmentContext
from ..fragments.base import BaseFragment # Need FragmentContext if passing it to handle_realtime_chat
# Also need access to other core components if handle_realtime_chat needs them in its context
from .llm_interface import LLMInterface 
from .tool_registry import ToolRegistry
from .memory.memory_manager import MemoryManager
# Assuming FragmentRegistry might be needed too?
from ..fragments.registry import FragmentRegistry

logger = logging.getLogger(__name__)

async def chat_monitor_task(
    task_id: str, # Pass task_id for logging
    shared_task_context: SharedTaskContext,
    # Pass other core components needed to construct FragmentContext
    llm_interface: LLMInterface,
    tool_registry: ToolRegistry,
    fragment_registry: FragmentRegistry,
    memory_manager: MemoryManager,
    workspace_root: Path # Assuming Path is imported or available
):
    """
    Monitors the internal chat queue for a specific task and dispatches
    messages to the handle_realtime_chat method of active fragments.
    """
    log_prefix = f"[ChatMonitor-{task_id[:8]}]"
    logger.info(f"{log_prefix} Starting...")
    queue = shared_task_context.internal_chat_queue

    try:
        while True:
            # Wait for the next message from the queue
            chat_entry = await queue.get()
            logger.debug(f"{log_prefix} Received message: {chat_entry}")

            sender = chat_entry.get("sender")
            msg_type = chat_entry.get("type")
            content = chat_entry.get("content", {})
            message_id = chat_entry.get("message_id")

            # Determine target fragment(s)
            target_name = content.get("target_fragment") # Specific target?

            target_instances: Dict[str, BaseFragment] = {}
            if target_name:
                # Get specific active fragment instance
                instance = await shared_task_context.get_active_fragment(target_name)
                if instance:
                    target_instances[target_name] = instance
                else:
                    logger.warning(f"{log_prefix} Target fragment '{target_name}' for msg {message_id} not found in active list.")
            else:
                # Broadcast to all active fragments (except sender?)
                all_active_names = await shared_task_context.get_all_active_fragment_names()
                for name in all_active_names:
                    if name != sender: # Don't echo back to sender
                        instance = await shared_task_context.get_active_fragment(name)
                        if instance:
                            target_instances[name] = instance

            if not target_instances:
                logger.debug(f"{log_prefix} No active target fragments found for message {message_id}.")
                queue.task_done() # Mark message as processed even if no target
                continue

            # Dispatch message to target(s)
            for frag_name, frag_instance in target_instances.items():
                if hasattr(frag_instance, "handle_realtime_chat"):
                    try:
                        # Construct a FragmentContext for the handler method
                        # Note: This context might be slightly different from the one
                        # the fragment has during its 'execute' call, depending on timing.
                        handler_context = FragmentContext(
                             logger=logger, # Use monitor's logger or fragment's? Needs decision.
                             llm_interface=llm_interface,
                             tool_registry=tool_registry,
                             fragment_registry=fragment_registry,
                             shared_task_context=shared_task_context,
                             workspace_root=workspace_root, # Need workspace root here
                             memory_manager=memory_manager
                         )

                        logger.debug(f"{log_prefix} Dispatching msg {message_id} to {frag_name}.handle_realtime_chat")
                        # Call the handler - NO await here if we want handlers to run concurrently?
                        # Or await if handler needs to complete before next message dispatch? Let's await for now.
                        await frag_instance.handle_realtime_chat(chat_entry, handler_context)

                    except asyncio.CancelledError:
                         logger.info(f"{log_prefix} Task cancelled while handling message for {frag_name}.")
                         raise # Re-raise cancellation
                    except Exception as e:
                        logger.exception(f"{log_prefix} Error calling handle_realtime_chat for {frag_name}: {e}")
                        # Continue monitoring other messages/fragments
                else:
                    logger.warning(f"{log_prefix} Fragment {frag_name} does not have handle_realtime_chat method.")

            # Mark the message as processed in the queue
            queue.task_done()

    except asyncio.CancelledError:
        logger.info(f"{log_prefix} Cancelled.")
    except Exception as e:
        logger.exception(f"{log_prefix} Unexpected error in main loop: {e}")
    finally:
        logger.info(f"{log_prefix} Stopped.") 