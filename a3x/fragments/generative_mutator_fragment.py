# a3x/fragments/mutator_fragment.py
import logging
import json
import datetime
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Awaitable

# --- Core A3X Imports ---
from a3x.fragments.base import BaseFragment, FragmentContext, FragmentDef # Use real imports
from a3x.fragments.registry import fragment # Use real decorator
from a3x.core.tool_registry import ToolRegistry 

# --- Remove Placeholder Imports/Definitions ---
# try:
#     from a3x.fragments.base import BaseFragment, FragmentContext
#     # Placeholder for fragment registration decorator
#     # from a3x.fragment_registry import fragment_decorator
#     # Placeholder for the actual skill/tool import
#     # from a3x.skills.code_generation import generate_code_variation
# except ImportError as e:
#     print(f"[MutatorFragment] Warning: Could not import core A3X components ({e}). Using placeholders.")
#     # Define placeholders if imports fail
#     class FragmentContext:
#         workspace_root: Optional[str] = None
#         post_message_handler: Optional[Callable[..., Awaitable[None]]] = None
#
#     class BaseFragment:
#         _a3x_fragment_name: str = "placeholder_fragment"
#         def __init__(self, *args, **kwargs): pass
#         def get_name(self) -> str: return self._a3x_fragment_name
#         async def post_chat_message(self, message_type: str, content: Dict, target_fragment: Optional[str] = None):
#              print(f"Placeholder post_chat_message: Type={message_type}, Target={target_fragment}, Content={content}")
#
#
#     # --- Placeholder Registration ---
#     def fragment_decorator(name, trigger_phrases):
#         def decorator(cls):
#             print(f"Placeholder: Registering fragment {name} with triggers {trigger_phrases}")
#             cls._a3x_fragment_name = name
#             cls._a3x_trigger_phrases = trigger_phrases
#             return cls
#         return decorator
#     # --- End Placeholder ---
#
#     # --- Placeholder Code Variation Skill ---
#     async def generate_code_variation(base_code: str, base_name: str, strategy: str) -> str:
#         """Placeholder for LLM-based code variation generation."""
#         print(f"Placeholder Skill: Generating variation for '{base_name}' using '{strategy}' strategy.")
#         variation_comment = f"# <<<< Variation generated from {base_name} using strategy '{strategy}' >>>>"
#         # Attempt simple renaming (heuristic)
#         new_class_name = f"{base_name.replace('_fragment', '')}MutVariationFragment" # Example naming
#         modified_code = re.sub(r"class\s+\w+\(BaseFragment\):", f"class {new_class_name}(BaseFragment):", base_code, count=1)
#         # Add a comment
#         modified_code = f"{variation_comment}\n{modified_code}\n{variation_comment}\n"
#         return modified_code
#     # --- End Placeholder ---


logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_NUM_VARIATIONS = 3
DEFAULT_STRATEGY = "llm_guided" # Example strategy
MUTATION_HISTORY_FILE = "a3x/a3net/data/mutation_history.jsonl"
FRAGMENT_DIR = "a3x/fragments"

# --- Fragment Registration (Use actual A3X mechanism) ---
# @register_fragment(name="mutator", trigger_phrases=["mutar fragmento", "corrigir erro automaticamente"])
# Remove trigger, not a valid argument for @fragment
@fragment(
    name="generative_mutator", 
    description="Gera variações (mutações) de um fragmento existente.",
    capabilities=["code_mutation", "fragment_generation"]
)
# --- End Placeholder ---

class GenerativeMutatorFragment(BaseFragment):
    """
    Generates variations (mutations) of an existing fragment's source code.
    It locates the base fragment file, uses a code generation skill to create
    N variations, saves them as new fragment files, logs the event, and
    optionally triggers a system-wide evaluation.
    
    Relies on the AggregateEvaluatorFragment -> PerformanceManagerFragment pipeline
    for subsequent comparison and potential promotion/archiving of mutations.
    
    A3L Command Structure (Example):
    mutar fragmento FragmentName [gerando <N>] [usando <strategy>]
    """
    # ... (restante da classe)

# --- Fragment Definition ---
GenerativeMutatorFragmentDef = FragmentDef(
    name="GenerativeMutator",
    description="Gera variações (mutações) de um fragmento existente.",
    fragment_class=GenerativeMutatorFragment,
    skills=["mutar_fragmento"],
    managed_skills=["mutar_fragmento"],
    prompt_template="Gere variações do fragmento base informado utilizando a estratégia especificada."
)

class GenerativeMutatorFragment(BaseFragment):
    """
    Generates variations (mutations) of an existing fragment's source code.

    It locates the base fragment file, uses a code generation skill to create
    N variations, saves them as new fragment files, logs the event, and
    optionally triggers a system-wide evaluation.

    Relies on the AggregateEvaluatorFragment -> PerformanceManagerFragment pipeline
    for subsequent comparison and potential promotion/archiving of mutations.

    A3L Command Structure (Example):
    mutar fragmento FragmentName [gerando <N>] [usando <strategy>]
    """

    def __init__(self, fragment_def: FragmentDef, tool_registry: ToolRegistry):
        super().__init__(fragment_def, tool_registry)
        # Tool registry is likely stored in self._tool_registry by BaseFragment/ManagerFragment
        # If not, uncomment the line below:
        # self._tool_registry = tool_registry
        logger.info(f"[{self.get_name()}] Initialized.")

    async def execute(self, ctx: FragmentContext, fragment_base_name: str, num_variations: int = DEFAULT_NUM_VARIATIONS, strategy: str = DEFAULT_STRATEGY, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes the fragment mutation process.

        Args:
            ctx: The fragment execution context.
            fragment_base_name: The name of the fragment to mutate (e.g., "MyFragment").
                               Assumes file is at a3x/fragments/my_fragment.py
            num_variations: The number of variations to generate.
            strategy: The mutation strategy to use (passed to the generation skill).
            **kwargs: Additional arguments.

        Returns:
            A dictionary containing the status, a message, and a list of
            generated fragment names/files.
        """
        logger.info(f"Starting mutation process for fragment '{fragment_base_name}' ({num_variations} variations, strategy: '{strategy}').")

        if not ctx.workspace_root:
            logger.error("Workspace root not found in context. Cannot locate fragment files.")
            return {"status": "error", "message": "Workspace root missing from context."}

        workspace_path = Path(ctx.workspace_root)
        fragment_dir_path = workspace_path / FRAGMENT_DIR
        history_path = workspace_path / MUTATION_HISTORY_FILE

        # Convert fragment name (e.g., "MyFragment") to filename stem (e.g., "my_fragment")
        # Simple heuristic: lowercase and add underscores before caps (excluding first char)
        base_filename_stem = re.sub(r'(?<!^)(?=[A-Z])', '_', fragment_base_name).lower()
        base_fragment_file = fragment_dir_path / f"{base_filename_stem}_fragment.py" # Convention

        # 1. Locate and Read Base Fragment Source Code
        try:
            if not base_fragment_file.is_file():
                 logger.error(f"Base fragment file not found at: {base_fragment_file}")
                 return {"status": "error", "message": f"Base fragment file not found: {base_fragment_file}"}

            original_code = base_fragment_file.read_text(encoding="utf-8")
            logger.info(f"Successfully read base fragment code from: {base_fragment_file}")

        except IOError as e:
            logger.error(f"Error reading base fragment file {base_fragment_file}: {e}", exc_info=True)
            return {"status": "error", "message": f"IOError reading base fragment file: {e}"}

        # 2. Generate Variations
        generated_mutations = []
        errors_generating = []
        if not hasattr(self, '_tool_registry') or not self._tool_registry:
             logger.error(f"[{self.get_name()}] ToolRegistry not available. Cannot generate variations.")
             return {"status": "error", "message": "ToolRegistry not available in fragment.", "suggested_a3l_commands": []}

        for i in range(num_variations):
            variation_number = i + 1
            logger.info(f"Generating variation {variation_number}/{num_variations}...")
            try:
                modification_prompt = (
                    f"Generate a plausible variation of the following code. "
                    f"Apply a modification strategy described as: '{strategy}'. "
                    f"Consider the original fragment name '{fragment_base_name}'. "
                    f"Return ONLY the complete, modified code block for the potential new fragment, "
                    f"without any explanations or markdown formatting."
                )

                modify_result = await self._tool_registry.execute_tool(
                    "modify_code",
                    {
                        "modification": modification_prompt,
                        "code_to_modify": original_code,
                        "language": "python" # Assuming python fragments
                    }
                    # Context for the skill is likely handled by execute_tool
                )

                modify_status = modify_result.get("status")
                modify_data = modify_result.get("data", {})
                mutated_code = ""

                if modify_status == "success":
                    mutated_code = modify_data.get("modified_code")
                    if not mutated_code:
                         raise ValueError("modify_code skill succeeded but returned empty code.")
                    logger.info(f"Variation {variation_number} generated successfully by modify_code.")
                elif modify_status == "no_change":
                     error_msg = f"modify_code skill reported no change for variation {variation_number}. Strategy: '{strategy}'."
                     logger.warning(error_msg)
                     errors_generating.append(error_msg)
                     continue # Skip this variation
                else: # error or unexpected status
                     error_msg = f"modify_code skill failed for variation {variation_number}. Status: {modify_status}. Error: {modify_data.get('message', 'Unknown error')}"
                     logger.error(error_msg)
                     errors_generating.append(error_msg)
                     continue # Skip this variation

                # Define new fragment name and filename
                # Ensure the *new* fragment name reflects it's a mutation
                new_fragment_name = f"{fragment_base_name}_mut_{variation_number}"
                new_filename_stem = f"{base_filename_stem}_mut_{variation_number}"
                new_fragment_file = fragment_dir_path / f"{new_filename_stem}_fragment.py"

                # **Critical Step:** Modify class name and potentially registration within the code
                # This basic regex assumes 'class FragmentName(BaseFragment):' structure
                # and registration like '@fragment_decorator(name="...", ...)'
                final_mutated_code = re.sub(
                    rf"class\s+{fragment_base_name}\(BaseFragment\):",
                    f"class {new_fragment_name}(BaseFragment):",
                    mutated_code, count=1
                )
                # Adjust registration name if decorator exists
                final_mutated_code = re.sub(
                    rf'@fragment_decorator\(name="{fragment_base_name}"',
                    f'@fragment_decorator(name="{new_fragment_name}"',
                    final_mutated_code, count=1
                )
                # (More robust parsing might be needed for complex cases)


                # Save the new fragment file
                new_fragment_file.parent.mkdir(parents=True, exist_ok=True)
                new_fragment_file.write_text(final_mutated_code, encoding="utf-8")

                generated_mutations.append({
                    "mutation_name": new_fragment_name,
                    "mutation_file": str(new_fragment_file.relative_to(workspace_path))
                })
                logger.info(f"Saved variation {variation_number} to {new_fragment_file}")

            except Exception as e:
                error_msg = f"Error generating or saving variation {variation_number}: {e}"
                logger.error(error_msg, exc_info=True)
                errors_generating.append(error_msg)

        if not generated_mutations:
             return {"status": "error", "message": f"Failed to generate any mutations. Errors: {errors_generating}", "suggested_a3l_commands": []}

        # 3. Log Mutation Event
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds'),
            "base_fragment_name": fragment_base_name,
            "base_fragment_file": str(base_fragment_file.relative_to(workspace_path)),
            "strategy": strategy,
            "num_requested": num_variations,
            "generated_mutations": generated_mutations,
            "generation_errors": errors_generating,
            "triggered_by": self.get_name()
        }
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(history_path, "a", encoding="utf-8") as f:
                 json_line = json.dumps(log_entry, ensure_ascii=False)
                 f.write(json_line + "\\n")
            logger.info(f"Mutation event logged to: {history_path}")
        except IOError as e:
             logger.error(f"Failed to log mutation history to {history_path}: {e}", exc_info=True)
             # Continue, but report the logging failure

        # 4. Suggest Evaluation via A3L
        suggested_a3l_commands = []
        for mutation in generated_mutations:
            mutation_name = mutation.get("mutation_name")
            if mutation_name:
                 # Command to evaluate the new mutation
                 eval_cmd = f"avaliar fragmento '{mutation_name}'" 
                 suggested_a3l_commands.append(eval_cmd)
            
        # Optionally, also suggest evaluating the original fragment for comparison
        # suggested_a3l_commands.append(f"avaliar fragmento '{fragment_base_name}'")
        
        logger.info(f"Generated {len(suggested_a3l_commands)} A3L suggestions for evaluating mutations.")

        # 5. Return Result with Suggestions
        return {
            "status": "success" if not errors_generating else "warning",
            "message": f"Successfully generated {len(generated_mutations)} mutations for '{fragment_base_name}'.",
            "generated_mutations": generated_mutations,
            "generation_errors": errors_generating,
            "suggested_a3l_commands": suggested_a3l_commands
        }

# Example Usage (for testing purposes, if run directly)
if __name__ == '__main__':
    import asyncio

    logging.basicConfig(level=logging.INFO)

    # --- Mock Context ---
    class MockContext(FragmentContext):
        def __init__(self, root):
            self.workspace_root = root
            self.post_message_handler = self.mock_post_message

        async def mock_post_message(self, message_type: str, content: Dict, target_fragment: Optional[str] = None):
            print(f"[Mock Post Message] Type: {message_type}, Target: {target_fragment}, Content: {content}")

    # --- Mock Base Fragment for Testing ---
    # Create a dummy fragment file to mutate
    mock_root = Path("./temp_mutator_test_ws")
    mock_root.mkdir(exist_ok=True)
    mock_frag_dir = mock_root / FRAGMENT_DIR
    mock_frag_dir.mkdir(exist_ok=True)
    mock_data_dir = mock_root / "a3x/a3net/data"
    mock_data_dir.mkdir(parents=True, exist_ok=True)

    base_frag_name = "ExampleSimple"
    base_frag_file = mock_frag_dir / f"{base_frag_name.lower()}_fragment.py"
    base_frag_code = """
# a3x/fragments/example_simple_fragment.py
import logging
from a3x.fragments.base import BaseFragment, FragmentContext
# @fragment_decorator(name="ExampleSimple", trigger_phrases=["do simple task"]) # Assume decorator exists
class ExampleSimple(BaseFragment):
    async def execute(self, ctx: FragmentContext, **kwargs):
        logging.info("ExampleSimple executed.")
        return {"status": "success", "message": "Simple task done."}
"""
    base_frag_file.write_text(base_frag_code, encoding="utf-8")
    print(f"Created mock base fragment: {base_frag_file}")

    # --- Instantiate and Run ---
    mock_ctx = MockContext(str(mock_root.resolve()))
    mock_fdef = FragmentDef(name="mutator_test") # Mock FragmentDef instance
    mock_registry = ToolRegistry()
    mutator = GenerativeMutatorFragment(mock_fdef, mock_registry) # Pass mock def and registry

    async def run_test():
        result = await mutator.execute(mock_ctx, fragment_base_name=base_frag_name, num_variations=2)
        print("\\n--- Execution Result ---")
        print(json.dumps(result, indent=2))

        # Check created files
        print("\\n--- Generated Files ---")
        for mut in result.get("generated_mutations", []):
            mut_file = mock_root / mut['mutation_file']
            if mut_file.exists():
                print(f"\n--- Content of {mut['mutation_file']} ---")
                print(mut_file.read_text(encoding='utf-8'))
            else:
                print(f"File not found: {mut_file}")

        # Check log file
        history_file = mock_root / result.get("history_log", "")
        if history_file.exists():
             print(f"\\n--- Content of {result.get('history_log', '')} ---")
             print(history_file.read_text(encoding='utf-8'))


    asyncio.run(run_test())

    # Cleanup (optional)
    # import shutil
    # shutil.rmtree(mock_root)
    # print(f"Cleaned up {mock_root}") 