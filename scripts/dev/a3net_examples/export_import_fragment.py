# Example script to test exporting and importing .a3xfrag fragment packages.

import asyncio
from pathlib import Path
import shutil # For cleaning up test directories

from a3x.a3net.core.memory_bank import MemoryBank
from a3x.a3net.integration.a3lang_interpreter import interpret_a3l_line

# Updated import path after moving a3net into a3x
# from a3x.a3net.integration.a3x_bridge import handle_directive, MEMORY_BANK

# --- Setup Test Environment ---
TEST_MEMORY_DIR = "./temp_export_import_memory"
TEST_REPO_DIR = "./temp_export_import_repo"

def cleanup_dirs():
    """Remove temporary directories."""
    if Path(TEST_MEMORY_DIR).exists():
        shutil.rmtree(TEST_MEMORY_DIR)
        print(f"Cleaned up {TEST_MEMORY_DIR}")
    if Path(TEST_REPO_DIR).exists():
        shutil.rmtree(TEST_REPO_DIR)
        print(f"Cleaned up {TEST_REPO_DIR}")

async def setup_test_fragment():
    """Creates a dummy fragment to export."""
    from a3x.a3net.core.neural_language_fragment import NeuralLanguageFragment
    memory_bank = MemoryBank(save_dir=TEST_MEMORY_DIR, export_dir=TEST_REPO_DIR)
    dummy_frag = NeuralLanguageFragment(
        fragment_id="export_test_frag", input_dim=5, num_classes=2
    )
    memory_bank.save("export_test_frag", dummy_frag)
    print("Dummy fragment 'export_test_frag' created for export.")
    return memory_bank # Return the initialized bank

async def main():
    # Clean up previous runs
    cleanup_dirs()
    
    # Create a fragment and get the memory bank instance
    memory_bank = await setup_test_fragment()
    
    # --- Example Directives ---
    export_path = Path(TEST_REPO_DIR) / "exported_frag.a3xfrag"
    export_directive_str = f"exportar fragmento 'export_test_frag' para '{str(export_path)}'"
    # Assume import loads it as 'imported_frag'
    import_directive_str = f"importar fragmento de '{str(export_path)}' como 'imported_frag'"
    
    directives_to_run = [
        export_directive_str,
        import_directive_str
    ]

    print("--- Running Export/Import Test Directives ---")
    for i, cmd_str in enumerate(directives_to_run):
        print(f"\n[Step {i+1}] Executing: {cmd_str}")
        directive_dict = interpret_a3l_line(cmd_str)
        if directive_dict:
             print(f"  Interpreted: {directive_dict}")
             # --- Cannot call handle_directive directly anymore ---
             print("  <<< SKIPPING EXECUTION (handle_directive moved) >>>")
             # result = await handle_directive(directive_dict, memory_bank=memory_bank)
             # print(f"  Result: {result}")
             # --- --------------------------------------------- ---
        else:
            print(f"  ERROR: Could not interpret directive: {cmd_str}")
            
    print("\n--- Checking Memory Bank after Import (Manual Check Recommended) ---")
    # Verify if 'imported_frag' exists (manual check needed as execution was skipped)
    # loaded_imported = memory_bank.load("imported_frag")
    # if loaded_imported:
    #     print("  Fragment 'imported_frag' found in MemoryBank after import test.")
    # else:
    #     print("  ERROR: Fragment 'imported_frag' NOT found after import test (or execution skipped)." )
    print("  (Execution was skipped, manual check of memory bank content needed if desired)")

    print("\n--- Export/Import Test Finished ---")
    # cleanup_dirs() # Optional: clean up immediately after run

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Test interrupted.")
    finally:
        cleanup_dirs() # Ensure cleanup happens even on interruption 