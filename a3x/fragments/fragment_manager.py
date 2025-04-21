import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import importlib.util # For dynamic loading
import inspect        # For inspecting loaded module

from .base import BaseFragment, FragmentDef
from a3x.core.context import FragmentContext, SharedTaskContext
from a3x.core.tool_registry import ToolRegistry

# Attempt to get PROJECT_ROOT, fallback if needed
try:
    from a3x.core.config import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    logging.getLogger(__name__).warning(f"Could not import PROJECT_ROOT from core.config, using fallback: {PROJECT_ROOT}")

logger = logging.getLogger(__name__)

# Basic regex to check for class definition inheriting from BaseFragment
# Corrected regex pattern (raw string and escaping)
FRAGMENT_CLASS_PATTERN = re.compile(r"class\s+(\w+Fragment)\(BaseFragment\):", re.MULTILINE)

class FragmentManagerFragment(BaseFragment):
    """Listens for 'register_fragment' directives and simulates the registration process."""

    def __init__(self, fragment_def: FragmentDef, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(fragment_def, tool_registry)
        self._logger.info(f"[{self.get_name()}] Initialized.")
        # In a real implementation, this might hold references to loaded fragments
        self.registered_fragments: Dict[str, Dict[str, Any]] = {}     # Store registration status
        self.dynamic_fragments: Dict[str, BaseFragment] = {} # Store instantiated dynamic fragments
        self.active_dynamic_fragments: Dict[str, Tuple[BaseFragment, FragmentContext]] = {} # Store active instances and their contexts
        self._dynamic_dispatcher_task: Optional[asyncio.Task] = None
        self._fragment_context: Optional[FragmentContext] = None # To access shared queue

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        """Returns a description of this fragment's purpose."""
        return "Listens for 'register_fragment' directives, validates, loads, instantiates, and activates new fragments dynamically."

    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """Handles incoming chat messages, looking for 'register_fragment' directives."""
        msg_type = message.get("type")
        sender = message.get("sender")
        content = message.get("content")

        # Check for the specific directive
        if (
            msg_type == "ARCHITECTURE_SUGGESTION"
            and isinstance(content, dict)
            and content.get("type") == "directive"
            and content.get("action") == "register_fragment"
        ):
            directive = content
            fragment_path = directive.get("path")
            fragment_name = directive.get("name") # Optional, but useful

            if not fragment_path:
                self._logger.warning(f"[{self.get_name()}] Received invalid 'register_fragment' directive (missing path): {directive}")
                return

            self._logger.info(f"[{self.get_name()}] Received 'register_fragment' directive from {sender} for path: {fragment_path}")
            await self._handle_register_fragment_directive(directive, context)
        else:
            pass # Ignore other messages

    async def _handle_register_fragment_directive(self, directive: Dict[str, Any], context: FragmentContext):
        """Handles the logic for validating and simulating the registration of a new fragment."""
        fragment_path_relative = directive.get("path")
        fragment_name = directive.get("name", Path(fragment_path_relative).stem)
        result_status = "error"
        result_summary = f"Failed to register fragment '{fragment_name}'."
        result_details = ""

        try:
            # 1. Read the fragment file content
            self._logger.info(f"[{self.get_name()}] Attempting to read fragment file: {fragment_path_relative}")
            # Assume read_file is available via ToolRegistry (likely from FileManagerSkill)
            read_result = await self._tool_registry.execute_tool(
                "read_file",
                {
                    "path": fragment_path_relative,
                }
            )

            if read_result.get("status") != "success":
                result_summary = f"Failed to read fragment file: {fragment_path_relative}"
                result_details = str(read_result.get("data", {}).get("message", "Read skill failed."))
                self._logger.error(f"[{self.get_name()}] {result_summary}. Details: {result_details}")
            else:
                fragment_code = read_result.get("data", {}).get("content")
                if not fragment_code:
                    result_summary = "Read fragment file but content was empty."
                    result_details = f"Path: {fragment_path_relative}"
                    self._logger.error(f"[{self.get_name()}] {result_summary}")
                else:
                    self._logger.info(f"[{self.get_name()}] Successfully read code for {fragment_path_relative}. Validating...")

                    # 2. Validate the content (basic check)
                    if FRAGMENT_CLASS_PATTERN.search(fragment_code):
                        # --- Simulation Success --- 
                        result_status = "success"
                        result_summary = f"SIMULATED: Successfully validated and registered fragment '{fragment_name}'."
                        result_details = f"Validated content from {fragment_path_relative}."

                        # --- Attempt Dynamic Loading and Instantiation ---
                        try:
                            absolute_path = (Path(PROJECT_ROOT) / fragment_path_relative).resolve()
                            module_name = f"a3x.fragments.dynamic.{fragment_name}" # Create a unique module name

                            spec = importlib.util.spec_from_file_location(module_name, absolute_path)
                            if spec is None or spec.loader is None:
                                raise ImportError(f"Could not create module spec for {fragment_path_relative}")

                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module) # Execute the module code
                            self._logger.info(f"[{self.get_name()}] Dynamically loaded module from {fragment_path_relative}")

                            # Find the Fragment class within the loaded module
                            FragmentClass = None
                            for name, obj in inspect.getmembers(module):
                                if inspect.isclass(obj) and issubclass(obj, BaseFragment) and obj is not BaseFragment:
                                    FragmentClass = obj
                                    break # Found the first one

                            if FragmentClass:
                                self._logger.info(f"[{self.get_name()}] Found Fragment class: {FragmentClass.__name__}. Instantiating...")
                                # Instantiate the fragment
                                # Create a simple FragmentDef for it
                                # TODO: Skills should ideally come from the directive
                                new_fragment_def = FragmentDef(name=fragment_name, description="Dynamically loaded fragment", skills=[])
                                new_fragment_instance = FragmentClass(fragment_def=new_fragment_def, tool_registry=self._tool_registry)

                                # Create a dedicated context for the new fragment
                                # It needs access to the essential components like logger, tools, shared context
                                new_fragment_context = FragmentContext(
                                    logger=logging.getLogger(f"a3x.fragments.dynamic.{fragment_name}"), # Unique logger name
                                    llm_interface=context.llm_interface, # Reuse existing interfaces/registries
                                    tool_registry=context.tool_registry,
                                    fragment_registry=context.fragment_registry,
                                    shared_task_context=context.shared_task_context,
                                    workspace_root=context.workspace_root,
                                    memory_manager=context.memory_manager
                                )

                                # Set context (basic shared context for now)
                                # The fragment's set_context might expect FragmentContext or SharedTaskContext
                                # Let's try passing the full FragmentContext first
                                try:
                                    new_fragment_instance.set_context(new_fragment_context)
                                except AttributeError:
                                     # Fallback if base set_context was removed/changed or it expects SharedTaskContext
                                     if hasattr(new_fragment_instance, '_context_accessor'):
                                         new_fragment_instance._context_accessor.set_context(context.shared_task_context)
                                     else:
                                          # If it directly expects SharedTaskContext in set_context
                                          try:
                                               new_fragment_instance.set_context(context.shared_task_context)
                                          except:
                                               self._logger.warning(f"Could not set context for dynamically loaded fragment {fragment_name}")

                                # Activate by adding to the dynamic list
                                self.active_dynamic_fragments[fragment_name] = (new_fragment_instance, new_fragment_context)
                                result_summary = f"Successfully loaded, instantiated, and ACTIVATED fragment '{fragment_name}'."
                                result_details += f" Instantiated '{FragmentClass.__name__}'. Now listening to messages."
                                self._logger.info(f"[{self.get_name()}] Activated dynamic fragment: {fragment_name}")
                            else:
                                result_status = "error" # Downgrade status if class not found
                                result_summary = f"Loaded module from {fragment_path_relative} but could not find a valid Fragment class."
                                result_details = "No class inheriting from BaseFragment found."
                                self._logger.error(f"[{self.get_name()}] {result_summary}")

                        except Exception as load_e:
                            result_status = "error" # Loading/Instantiation failed
                            result_summary = f"Error during dynamic loading/instantiation of {fragment_path_relative}."
                            result_details = f"Error: {load_e}"
                            self._logger.exception(f"[{self.get_name()}] Failed to load/instantiate fragment:")
                        # -------------------------------------------------
                        # Update registered status regardless of load success if validation passed initially
                        self.registered_fragments[fragment_name] = {"path": fragment_path_relative, "status": "validated"}
                        # ---------------------------
                    else:
                        result_summary = f"Validation failed for fragment file: {fragment_path_relative}"
                        result_details = "Could not find 'class ...Fragment(BaseFragment):' definition."
                        self._logger.warning(f"[{self.get_name()}] {result_summary} Details: {result_details}")

        except Exception as e:
            self._logger.exception(f"[{self.get_name()}] Unexpected error handling register_fragment directive for {fragment_path_relative}:")
            result_status = "error"
            result_summary = "Unexpected error during fragment registration simulation."
            result_details = str(e)

        # 3. Send result back via chat
        await self.broadcast_result_via_chat(context, directive, result_status, result_summary, result_details)

    async def broadcast_result_via_chat(self, context: FragmentContext, original_directive: Dict, status: str, summary: str, details: str):
        """Broadcasts the result of handling the register_fragment directive."""
        result_message_content = {
            "type": "manager_result", # Specific type for this fragment's results
            "status": status,
            "target": original_directive.get("path", "unknown"), # Target is the path being registered
            "fragment_name": original_directive.get("name", "unknown"),
            "original_action": "register_fragment",
            "summary": summary,
            "details": details,
            "original_directive": original_directive
        }
        try:
            await self.post_chat_message(
                context=context,
                message_type="manager_result", # Use the specific type
                content=result_message_content
            )
            self._logger.info(f"[{self.get_name()}] Broadcasted manager_result for target '{result_message_content['target']}': {status}")
        except Exception as e:
            self._logger.error(f"[{self.get_name()}] Failed to broadcast manager_result via chat: {e}")

    # Override set_context to store the context and start the dynamic dispatcher
    def set_context(self, context: FragmentContext):
        """Sets the context and loads initial fragments based on registry."""
        # The context passed from the runner loop IS the SharedTaskContext
        shared_context = context
        super().set_context(shared_context) # Call parent's set_context
        self._fragment_context = shared_context # Store the shared context
        # Optionally load initial fragments if needed upon context setting
        # asyncio.create_task(self.load_initial_fragments())
        self._logger.info(f"[{self.get_name()}] Context received.")
        self._start_dynamic_dispatcher_loop()

    def _start_dynamic_dispatcher_loop(self):
        """Starts the background task for dispatching messages to dynamic fragments."""
        if self._dynamic_dispatcher_task is None or self._dynamic_dispatcher_task.done():
            if self._fragment_context: # Check only if the context (which is SharedTaskContext) exists
                self._logger.info(f"[{self.get_name()}] Starting dynamic message dispatcher loop...")
                # Ensure the queue access is correct, assuming _fragment_context is the SharedTaskContext
                queue = self._fragment_context.internal_chat_queue
                self._dynamic_dispatcher_task = asyncio.create_task(
                    self._dynamic_dispatcher_loop(queue),
                    name=f"{self.get_name()}_DynamicDispatcher"
                )
                self._dynamic_dispatcher_task.add_done_callback(self._handle_loop_completion)
            else:
                self._logger.warning(f"[{self.get_name()}] Cannot start dynamic dispatcher: SharedTaskContext not set.")
        else:
             self._logger.info(f"[{self.get_name()}] Dynamic dispatcher loop already running.")

    async def _dynamic_dispatcher_loop(self, queue: asyncio.Queue):
        """Listens to the main queue and dispatches messages to dynamically loaded fragments."""
        self._logger.info(f"[{self.get_name()}] Dynamic dispatcher running.")
        while True:
            try:
                message = await queue.get()
                if not self.active_dynamic_fragments:
                    queue.task_done()
                    continue # Skip if no dynamic fragments are active

                # Dispatch to all active dynamic fragments concurrently
                dispatch_tasks = []
                for frag_name, (fragment, context) in self.active_dynamic_fragments.items():
                    dispatch_tasks.append(asyncio.create_task(
                        fragment.handle_realtime_chat(message, context),
                        name=f"DynHandler_{frag_name}_{message.get('type')}"
                    ))

                if dispatch_tasks:
                    done, pending = await asyncio.wait(dispatch_tasks, timeout=5.0) # Short timeout
                    # Handle pending/errors if necessary (similar to main dispatcher)
                    if pending:
                        self._logger.warning(f"[{self.get_name()}-DynDispatcher] {len(pending)} dynamic fragment handlers timed out processing message type '{message.get('type')}'")
                        for task in pending: task.cancel()
                    for task in done:
                        if task.exception():
                            frag_name_in_task = task.get_name().split('_')[1] if task.get_name().startswith("DynHandler_") else "UnknownDynamicFragment"
                            self._logger.error(f"[{self.get_name()}-DynDispatcher] Error in {frag_name_in_task} handling message type '{message.get('type')}':", exc_info=task.exception())

                queue.task_done()

            except asyncio.CancelledError:
                self._logger.info(f"[{self.get_name()}] Dynamic dispatcher loop cancelled.")
                break
            except Exception as e:
                self._logger.exception(f"[{self.get_name()}] Error in dynamic dispatcher loop:")
                await asyncio.sleep(1) # Avoid tight loop

    def _handle_loop_completion(self, task: asyncio.Task):
        """Callback to log completion or errors of the dynamic dispatcher loop."""
        try:
            exception = task.exception()
            if exception:
                self._logger.error(f"[{self.get_name()}] Dynamic dispatcher loop task failed:", exc_info=exception)
            else:
                self._logger.info(f"[{self.get_name()}] Dynamic dispatcher loop task completed.")
        except asyncio.CancelledError:
            self._logger.info(f"[{self.get_name()}] Dynamic dispatcher loop task was cancelled.")
        self._dynamic_dispatcher_task = None # Reset task reference

    # In a real implementation, might need methods to list/get registered fragments
    # async def list_registered_fragments(self):
    #     return list(self.registered_fragments.keys()) 

    # async def list_active_dynamic_fragments(self):
    #     return list(self.active_dynamic_fragments.keys())

    async def shutdown(self):
        """Cleans up the dynamic dispatcher task and shuts down dynamic fragments."""
        # Cancel the dynamic dispatcher task
        if self._dynamic_dispatcher_task and not self._dynamic_dispatcher_task.done():
            self._logger.info(f"[{self.get_name()}] Requesting cancellation of dynamic dispatcher loop task...")
            self._dynamic_dispatcher_task.cancel()
            try:
                await self._dynamic_dispatcher_task
            except asyncio.CancelledError:
                 self._logger.info(f"[{self.get_name()}] Dynamic dispatcher loop task successfully cancelled.")
            except Exception as e:
                 self._logger.error(f"[{self.get_name()}] Error during dynamic dispatcher loop task cleanup:", exc_info=e)

        # Call shutdown on dynamically loaded fragments
        self._logger.info(f"[{self.get_name()}] Shutting down dynamically loaded fragments...")
        shutdown_tasks = []
        for frag_name, (fragment, _) in self.active_dynamic_fragments.items():
            if hasattr(fragment, 'shutdown'):
                self._logger.debug(f"Calling shutdown for dynamic fragment {frag_name}")
                shutdown_tasks.append(asyncio.create_task(fragment.shutdown()))
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            self._logger.info(f"[{self.get_name()}] Dynamic fragment shutdown complete.")

        self._logger.info(f"[{self.get_name()}] Shutdown complete.") 

    async def unload_fragment(self, fragment_name: str) -> Dict[str, str]:
        """Unloads a fragment from the dynamic fragment list."""
        try:
            if fragment_name in self.active_dynamic_fragments:
                fragment, _ = self.active_dynamic_fragments[fragment_name]
                self._logger.info(f"[{self.get_name()}] Unloading fragment '{fragment_name}'...")
                # Optionally, post a status message
                await self.post_chat_message(
                    message_type="fragment_status",
                    content={
                        "name": fragment_name,
                        "status": "unloaded",
                        "reason": "Unload requested"
                    }
                )
                return {"status": "success", "message": f"Fragment '{fragment_name}' unloaded."}
            else:
                self._logger.warning(f"[{self.get_name()}] Fragment '{fragment_name}' not found or not loaded dynamically.")
                return {"status": "not_found", "message": f"Fragment '{fragment_name}' not found or cannot be unloaded."}
        except Exception as e:
            self._logger.error(f"[{self.get_name()}] Failed to unload fragment '{fragment_name}': {e}", exc_info=True)
            return {"status": "error", "message": f"Error unloading fragment: {str(e)}"} 