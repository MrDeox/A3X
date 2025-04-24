import logging
import json
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Core A3X Imports (adjust based on actual project structure)
try:
    from a3x.fragments.base import BaseFragment, FragmentContext # Base class and context
    # Fragment registration mechanism (replace with actual import if known)
    # from a3x.fragment_registry import register_fragment, fragment_decorator
except ImportError as e:
    print(f"[A3LangEvolverFragment] Warning: Could not import core A3X components ({e}). Using placeholders.")
    # Define placeholders if imports fail
    class FragmentContext:
        workspace_root: Optional[str] = None
    class BaseFragment:
        def __init__(self, *args, **kwargs): pass

logger = logging.getLogger(__name__)

# --- Fragment Registration (Placeholder) ---
# This fragment might not be directly invoked by A3L but rather called internally
# when the A3L interpreter fails. Registration might not follow standard patterns.
# Consider how this fragment gets triggered.
# --- End Placeholder ---

class A3LangEvolverFragment(BaseFragment):
    """
    Analyzes unrecognized A3L commands and proposes potential language extensions
    by logging them to a dedicated file.

    Input: An invalid A3L command string.
    Output: Saves a proposal to a3x/a3lang/data/proposed_extensions.jsonl
            Returns a status message and the suggested symbolic action.
    """

    # Path relative to workspace root
    PROPOSAL_FILE = "a3x/a3lang/data/proposed_extensions.jsonl"

    async def execute(self, ctx: FragmentContext, invalid_command: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Analyzes the invalid command and logs a potential language extension.

        Args:
            ctx: The execution context, providing access to the workspace root.
            invalid_command: The original unrecognized A3L command string.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary indicating the outcome and the suggested symbolic action.
            Example: {"status": "success", "message": "Suggestion registered.",
                      "proposed_action": "criar skill para X"}
        """
        logger.info(f"Executing A3LangEvolverFragment for unrecognized command: '{invalid_command}'")

        if not invalid_command or not isinstance(invalid_command, str):
             logger.error("Invalid input: 'invalid_command' must be a non-empty string.")
             return {"status": "error", "message": "Invalid input command string.", "proposed_action": None}

        # 1. Validate Context
        if not hasattr(ctx, 'workspace_root') or not ctx.workspace_root:
             logger.error("Context (ctx) lacks a valid 'workspace_root' attribute.")
             return {"status": "error", "message": "Workspace root not found.", "proposed_action": None}

        workspace_root = Path(ctx.workspace_root)
        proposal_path = workspace_root / self.PROPOSAL_FILE

        # 2. Log the unrecognized attempt (using standard logging)
        logger.warning(f"Unrecognized A3L command attempt: '{invalid_command}'")

        # 3. Simple Analysis and Proposal Generation (Basic Heuristics)
        words = invalid_command.strip().split()
        proposed_action = "(No specific action proposed)"
        template = "(No template generated)"
        proposed_verb = None

        if words:
            proposed_verb = words[0].lower() # Assume first word is the intended verb
            # Simple proposal: suggest creating a skill for the verb + rest of command
            object_phrase = " ".join(words[1:]) if len(words) > 1 else "[unspecified object]"
            proposed_action = f"criar skill para {proposed_verb} {object_phrase}"
            # Simple template: verb + placeholders
            template = f"{proposed_verb} {{objeto}} ..." # Basic template
            if "com base em" in invalid_command:
                template = f"{proposed_verb} {{objeto}} com base em {{fonte}}"
            elif "para" in invalid_command:
                 template = f"{proposed_verb} {{objeto}} para {{destino/objetivo}}"
            # Add more simple template rules if needed

        # 4. Create Proposal Record
        proposal_record = {
            "original_input": invalid_command,
            "proposed_action": proposed_action,
            "template": template,
            "potential_verb": proposed_verb,
            "created_by": self.__class__.__name__,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
        }

        # 5. Save Proposal to JSONL file
        try:
            # Ensure the output directory exists
            proposal_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured proposal directory exists: {proposal_path.parent}")

            # Append the proposal as a JSON line
            with open(proposal_path, "a", encoding="utf-8") as f:
                json_line = json.dumps(proposal_record, ensure_ascii=False)
                f.write(json_line + "\n")

            logger.info(f"Language extension proposal saved to: {proposal_path}")
            message = "Language extension proposal registered successfully."
            status = "success"

        except OSError as e:
            logger.error(f"Failed to create proposal directory {proposal_path.parent}: {e}", exc_info=True)
            message = f"Failed to create proposal directory: {e}"
            status = "error"
        except IOError as e:
            logger.error(f"Failed to write proposal to {proposal_path}: {e}", exc_info=True)
            message = f"Failed to write proposal file: {e}"
            status = "error"
        except Exception as e:
            logger.error(f"An unexpected error occurred saving the proposal: {e}", exc_info=True)
            message = f"Unexpected error saving proposal: {e}"
            status = "error"

        # 6. Return Result
        return {
            "status": status,
            "message": message,
            "proposed_action": proposed_action if status == "success" else None,
            "original_command": invalid_command
        }

# --- Example Usage/Testing (Optional) ---
# async def main():
#     # Setup logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     import time # Add import for timing

#     # Create dummy workspace
#     test_workspace = Path("./temp_evolver_workspace")
#     test_workspace.mkdir(exist_ok=True)
#     proposal_dir = test_workspace / "a3x/a3lang/data"
#     # Don't create the dir here, let the fragment do it

#     # Mock Context
#     class MockContext(FragmentContext):
#          def __init__(self, workspace):
#               self.workspace_root = workspace

#     # Instantiate Fragment
#     fragment = A3LangEvolverFragment() # Assuming no complex init args
#     context = MockContext(str(test_workspace.resolve()))

#     # --- Test Case 1 --- 
#     print("--- Test Case 1 --- ")
#     invalid_cmd1 = "sintetizar prompt com base em memoria episodica"
#     result1 = await fragment.execute(context, invalid_command=invalid_cmd1)
#     print(f"Execution result 1: {result1}")

#     # --- Test Case 2 --- 
#     print("\n--- Test Case 2 --- ")
#     invalid_cmd2 = "analisar logs para encontrar erros"
#     result2 = await fragment.execute(context, invalid_command=invalid_cmd2)
#     print(f"Execution result 2: {result2}")
    
#     # --- Test Case 3 --- 
#     print("\n--- Test Case 3 --- ")
#     invalid_cmd3 = "summarize discussion"
#     result3 = await fragment.execute(context, invalid_command=invalid_cmd3)
#     print(f"Execution result 3: {result3}")

#     # Check proposal file content
#     proposal_file_path = proposal_dir / "proposed_extensions.jsonl"
#     if proposal_file_path.exists():
#         print(f"\n--- Content of {proposal_file_path} ---")
#         with open(proposal_file_path, 'r') as f:
#             print(f.read())
#         # Clean up
#         # proposal_file_path.unlink()
#         # proposal_dir.rmdir()
#         # Path(test_workspace / "a3x/a3lang").rmdir()
#         # Path(test_workspace / "a3x").rmdir()
#     else:
#         print(f"Proposal file {proposal_file_path} was not created.")
    
#     # Clean up workspace
#     # test_workspace.rmdir()

# if __name__ == "__main__":
#     import asyncio
#     import time # Ensure time is imported
#     import datetime # Ensure datetime is imported
#     asyncio.run(main()) 