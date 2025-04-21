# Example script to test exporting and importing .a3xfrag fragment packages.

import sys
from pathlib import Path

# Adjust path to import from parent directory if needed (when running as script)
# sys.path.append(str(Path(__file__).resolve().parents[2])) 

# Updated import path after moving a3net into a3x
from a3x.a3net.integration.a3x_bridge import handle_directive, MEMORY_BANK

if __name__ == '__main__':
    print("--- Running A³Net Fragment Export/Import Example ---")

    # --- Configuration ---
    fragment_to_test = "frag_reflector"
    default_export_dir = "a3x_repo" # Should match MemoryBank default
    expected_export_path = Path(default_export_dir) / f"{fragment_to_test}.a3xfrag"

    # --- Ensure the target fragment exists in memory/disk for export ---
    # This relies on the fragment being trained/saved by another script run
    # e.g., python -m a3x.a3net.examples.test_reflective_fragment
    print(f"\nChecking if fragment '{fragment_to_test}' is loadable for export...")
    if MEMORY_BANK.load(fragment_to_test) is None:
        print(f"Error: Fragment '{fragment_to_test}' cannot be loaded. Please run its training script first.")
        print("Example: python -m a3x.a3net.examples.test_reflective_fragment")
        print("\n--- Export/Import Example Aborted ---")
        exit()
    else:
        print(f"Fragment '{fragment_to_test}' loaded successfully, proceeding with export.")
        # MEMORY_BANK.list() # Optionally list cached fragments

    # --- Step 1: Export the Fragment --- 
    print(f"\n--- Testing Export Directive for '{fragment_to_test}' --- ")
    export_directive = {
        "type": "export_fragment",
        "fragment_id": fragment_to_test
        # No path specified, should use default export dir
    }

    print(f"Sending export directive: {export_directive}")
    export_result = handle_directive(export_directive)

    print("\nExport Result:")
    print(export_result)

    exported_path_str = None
    if export_result and export_result.get("status") == "success":
        print("Export successful!")
        exported_path_str = export_result.get("path")
        if exported_path_str:
            print(f"Fragment exported to: {exported_path_str}")
            # Basic check if the path matches expected default
            if Path(exported_path_str).resolve() != expected_export_path.resolve():
                 print(f"Warning: Export path {exported_path_str} differs from expected default {expected_export_path}")
        else:
            print("Warning: Export successful but path not returned in result.")
    else:
        print("Export failed.")
        print("\n--- Export/Import Example Finished (Export Failed) ---")
        exit()
        
    # --- Step 2: Import the Fragment --- 
    print(f"\n--- Testing Import Directive from '{exported_path_str}' --- ")
    
    # Clear the fragment from memory cache first to force loading from package
    if fragment_to_test in MEMORY_BANK.fragments:
         del MEMORY_BANK.fragments[fragment_to_test]
         print(f"Removed '{fragment_to_test}' from memory cache to test import from file.")

    import_directive = {
        "type": "import_fragment",
        "path": exported_path_str # Use the path returned by the export step
    }

    print(f"Sending import directive: {import_directive}")
    import_result = handle_directive(import_directive)

    print("\nImport Result:")
    print(import_result)

    if import_result and import_result.get("status") == "success":
        print("Import successful!")
        # Verify it's back in the memory bank
        if fragment_to_test in MEMORY_BANK.fragments:
             print(f"Fragment '{fragment_to_test}' is now loaded in the MemoryBank cache.")
        else:
             print(f"Warning: Import reported success, but '{fragment_to_test}' not found in MemoryBank cache afterwards.")
    else:
        print("Import failed.")

    print("\n--- Export/Import Example Finished --- ") 