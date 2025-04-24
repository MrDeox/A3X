# a3x/fragments/mutator.py

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Core A³X components
from a3x.fragments.registry import fragment
from a3x.fragments.base import BaseFragment, FragmentDef
from a3x.core.context import FragmentContext
from a3x.core.tool_registry import ToolRegistry
# Import PROJECT_ROOT for path manipulation
try:
    from a3x.core.config import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent.parent # Fallback
    logging.getLogger(__name__).error("Failed to import PROJECT_ROOT from config in MutatorFragment")

logger = logging.getLogger(__name__)

@fragment(
    name="corrective_mutator",
    description="Listens for specific failure messages and attempts to automatically correct the associated code file.",
    category="Maintenance", # Or "Correction", "Learning"
    skills=["read_file", "modify_code", "write_file"], # Skills it utilizes
    managed_skills=[] # Does not manage other skills directly
)
class CorrectiveMutatorFragment(BaseFragment):
    """
    A fragment that listens for failure notifications (e.g., sandbox failures)
    and attempts to automatically mutate the code to fix the reported error.
    """

    def __init__(self, fragment_def: FragmentDef, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(fragment_def, tool_registry=tool_registry)
        self.memory_coins = 0 # Initialize balance
        self._logger.info(f"Fragment '{self.metadata.name}' initialized. Listening for failures... Balance: {self.memory_coins}")

    def get_purpose(self) -> str:
        """Returns the purpose of this fragment."""
        return "Listens for code execution failures and attempts automatic correction."

    def get_balance(self) -> int:
        """Returns the current memory coin balance."""
        return self.memory_coins

    async def execute_task(
        self,
        objective: str, # e.g., "Listen for and correct code failures"
        tools: list = None,
        context: Optional[FragmentContext] = None,
    ) -> str:
        """Main entry point (can be used to signify readiness)."""
        if not context:
             self._logger.error("FragmentContext is required for MutatorFragment.")
             return "Error: Context not provided."

        self._logger.info(f"[{self.get_name()}] Ready and listening for failure messages via handle_realtime_chat.")
        # This fragment primarily reacts to messages via handle_realtime_chat
        # No continuous loop needed here if messages are pushed via context/communicator
        return "MutatorFragment is active and ready to handle failure messages."

    async def handle_realtime_chat(self, message: Dict[str, Any], context: FragmentContext):
        """Handles incoming chat messages, looking for specific failure reports."""
        msg_type = message.get("type")
        content = message.get("content", {})
        sender = message.get("sender", "Unknown") # Good to know who reported the failure

        # --- Filter for relevant failure messages ---
        # Listen for 'refactor_result' with status 'error' and 'correction_initiated' potentially set
        # or a more generic 'status' message if defined elsewhere. Let's target the specific known failure first.
        is_failure_report = (
            msg_type == "refactor_result" and
            isinstance(content, dict) and
            content.get("status") == "error" and
            "sandbox_stderr" in content.get("details", {}) and # Check if details contain stderr
            "generated_path" in content.get("details", {})   # Check if details contain the path
        )

        # Could also listen for a hypothetical generic failure:
        # is_generic_failure = (
        #     msg_type == "status" and
        #     isinstance(content, dict) and
        #     content.get("status") == "failed" and
        #     "file_path" in content and
        #     "stderr" in content
        # )
        # if is_generic_failure:
        #     file_path = content.get("file_path")
        #     stderr = content.get("stderr")
        #     source_info = f"Generic failure report from {sender}"
        #     await self._attempt_mutation(file_path, stderr, source_info, context)

        if is_failure_report:
            details_str = content.get("details", "{}") # Get details as string
            try:
                details = json.loads(details_str) # Parse the JSON string
                if not isinstance(details, dict): # Ensure it parsed to a dict
                    self._logger.error(f"[{self.get_name()}] Parsed details is not a dictionary: {details}")
                    return 
            except json.JSONDecodeError as e:
                self._logger.error(f"[{self.get_name()}] Failed to parse details JSON: {e}. Details string: {details_str}")
                return # Cannot proceed without parsed details

            # Now access the parsed dictionary
            absolute_path = details.get("generated_path") 
            stderr = details.get("sandbox_stderr")
            original_action = content.get("original_action", "unknown")
            source_info = f"Failure report from {sender} (Original action: {original_action})"

            if absolute_path and stderr:
                # Convert absolute path to relative path based on PROJECT_ROOT
                try:
                    # Ensure PROJECT_ROOT is a Path object and exists
                    proj_root_path = Path(PROJECT_ROOT)
                    if not proj_root_path.is_dir():
                        raise FileNotFoundError(f"PROJECT_ROOT directory not found: {PROJECT_ROOT}")
                        
                    file_path_relative = str(Path(absolute_path).resolve().relative_to(proj_root_path.resolve()))
                    self._logger.info(f"[{self.get_name()}] Initiating mutation attempt for '{file_path_relative}' based on error from '{sender}'.")
                    await self._attempt_mutation(file_path_relative, stderr, source_info, context)
                except ValueError as e:
                    self._logger.error(f"[{self.get_name()}] Error determining relative path for '{absolute_path}' relative to '{PROJECT_ROOT}': {e}")
                except FileNotFoundError as e:
                    self._logger.error(f"[{self.get_name()}] Configuration error: {e}") 
                except Exception as e: # Catch other potential path errors
                    self._logger.exception(f"[{self.get_name()}] Unexpected error processing path '{absolute_path}':")
            else:
                self._logger.warning(f"[{self.get_name()}] Received error report from '{sender}' but missing 'generated_path' or 'sandbox_stderr' in details: {details}")
        
        # --- <<< ADDED: Handle mutation request from AnomalyDetector >>> ---
        elif msg_type == "mutation_request":
            self._logger.info(f"[{self.get_name()}] Received mutation request from '{sender}'.")
            target_path = content.get("target")
            failure_details = content.get("failure_reason") # This is expected to be a dict
            triggering_fragment = content.get("triggering_fragment", "Unknown")

            if not target_path or not isinstance(failure_details, dict):
                self._logger.warning(f"[{self.get_name()}] Invalid mutation request: Missing target path or failure details are not a dict. Content: {content}")
                return

            # Extract relevant error message (prioritize stderr if available)
            error_info = failure_details.get("sandbox_stderr", failure_details.get("details_string", failure_details.get("summary", "Unknown error details provided.")))
            source_info = f"Mutation requested by {sender} (AnomalyDetector) due to failure from {triggering_fragment}."
            
            # Ensure path is relative (AnomalyDetector should provide relative)
            if Path(target_path).is_absolute():
                 self._logger.warning(f"[{self.get_name()}] Received absolute path '{target_path}' in mutation request. Attempting conversion...")
                 try:
                     proj_root_path = Path(PROJECT_ROOT)
                     if not proj_root_path.is_dir():
                         raise FileNotFoundError(f"PROJECT_ROOT directory not found: {PROJECT_ROOT}")
                     target_path = str(Path(target_path).resolve().relative_to(proj_root_path.resolve()))
                     self._logger.info(f"[{self.get_name()}] Converted absolute path to relative: {target_path}")
                 except Exception as e:
                     self._logger.error(f"[{self.get_name()}] Failed to convert absolute path '{target_path}' from mutation request: {e}. Skipping mutation.")
                     return

            self._logger.info(f"[{self.get_name()}] Initiating mutation attempt for '{target_path}' based on mutation request.")
            await self._attempt_mutation(target_path, str(error_info), source_info, context)
        # --- <<< END ADDED Block >>> ---
        
        # Check for reward message *independently* or as part of the main if/else chain
        # Let's put it as an elif to the initial check
        elif msg_type == "reward":
             target = content.get("target")
             if target == self.get_name(): # Reward is for me
                 amount = content.get("amount")
                 reason = content.get("reason", "No reason specified")
                 sender = message.get("sender", "Unknown")
                 if isinstance(amount, int) and amount > 0:
                     self.memory_coins += amount
                     self._logger.info(f"[{self.get_name()}] Received {amount} memory_coins from '{sender}' for: '{reason}'. New balance: {self.memory_coins}")
                 else:
                      self._logger.warning(f"[{self.get_name()}] Received invalid reward amount '{amount}' from '{sender}' for target '{target}'. Ignoring.")
             # else: Reward was for someone else, ignore
        else:
            # Optionally log other messages received for debugging purposes
            # self._logger.debug(f"[{self.get_name()}] Received unrelated message type '{msg_type}'. Skipping.")
            pass # Ignore other message types

    async def _attempt_mutation(self, file_path: str, stderr: str, source_info: str, context: FragmentContext):
        """Reads the file, attempts modification, writes back, and broadcasts result."""
        original_code = ""
        modified_code = ""
        read_status = "failed"
        modify_status = "not_run"
        write_status = "not_run"

        try:
            # 1. Read original code
            self._logger.debug(f"[{self.get_name()}] Reading original code from: {file_path}")
            read_tool_func = self._tool_registry.get_tool("read_file")
            read_result = await read_tool_func(file_path=file_path) # Call directly
            
            read_status = read_result.get("status", "error")
            if read_status != "success":
                self._logger.error(f"[{self.get_name()}] Failed to read file {file_path} for mutation. Error: {read_result.get('data', {}).get('message')}")
                return # Cannot proceed without original code
            original_code = read_result.get("data", {}).get("content", "")
            self._logger.info(f"[{self.get_name()}] Read {len(original_code)} bytes from {file_path}.")

            # 2. Modify code
            modification_instruction = f"This code failed with the following error: {stderr}. Try to improve the code below to resolve the problem. ONLY return the improved Python code, without explanations."
            self._logger.debug(f"[{self.get_name()}] Calling modify_code for {file_path}...")
            modify_tool_func = self._tool_registry.get_tool("modify_code")
            modify_result = await modify_tool_func(
                file_path=file_path, # Pass file_path if skill needs it 
                original_code=original_code, 
                instructions=modification_instruction
            )
            modify_status = modify_result.get("status")

            if modify_status == "error":
                self._logger.error(f"[{self.get_name()}] modify_code skill failed for {file_path}. Error: {modify_result.get('data', {}).get('message')}")
                return # Failed to get modification
            elif modify_status == "no_change":
                self._logger.warning(f"[{self.get_name()}] modify_code skill reported no change for {file_path}. Mutation attempt skipped.")
                # Maybe broadcast a "mutation_skipped_no_change" status? For now, just log.
                return
            elif modify_status != "success":
                self._logger.error(f"[{self.get_name()}] modify_code skill returned unexpected status '{modify_status}' for {file_path}.")
                return # Unknown status

            modified_code = modify_result.get("data", {}).get("modified_code", "")
            if not modified_code:
                 self._logger.error(f"[{self.get_name()}] modify_code returned success but modified_code is empty for {file_path}. Aborting write.")
                 return
            self._logger.info(f"[{self.get_name()}] modify_code succeeded for {file_path}. Received modified code ({len(modified_code)} bytes).")

            # 3. Write modified code
            self._logger.debug(f"[{self.get_name()}] Writing modified code to {file_path}...")
            write_tool_func = self._tool_registry.get_tool("write_file")
            write_result = await write_tool_func(file_path=file_path, content=modified_code) # Assume overwrite=True is default or handled by skill
            
            write_status = write_result.get("status", "error")

            if write_status != "success":
                self._logger.error(f"[{self.get_name()}] Failed to write modified code to {file_path}. Error: {write_result.get('data', {}).get('message')}")
                # Maybe attempt rollback? For now, just log error.
                return # Failed to save modification

            self._logger.info(f"[{self.get_name()}] Successfully wrote modified code to {file_path}.")

            # 4. Broadcast mutation attempt success
            # Prepare details for the message, avoid sending full code unless necessary
            mod_details = {
                "original_code_snippet": original_code[:500], # Snippet
                "modified_code_snippet": modified_code[:500], # Snippet
                "instructions": modification_instruction # Record the instructions used
            }
            mutation_message = {
                "file_path": file_path,
                "status": "proposed", # Indicates a change was made, needs re-evaluation
                "from": self.get_name(),
                "based_on": source_info, # Include info about the source failure report
                "modification_details": mod_details 
            }
            await self.post_chat_message(
                context=context,
                message_type="mutation_attempt",
                content=mutation_message
                # target_fragment=None # Broadcast
            )
            self._logger.info(f"[{self.get_name()}] Broadcasted successful mutation proposal for {file_path}.")

        except KeyError as e:
            self._logger.error(f"[{self.get_name()}] Failed to find skill '{e}' in ToolRegistry during mutation attempt for {file_path}.")
        except Exception as e:
            self._logger.exception(f"[{self.get_name()}] Unexpected error during mutation attempt for {file_path}: {e}")
            # Optionally broadcast a mutation_failed message? 