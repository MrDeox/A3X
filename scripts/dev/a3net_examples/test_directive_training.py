# Example script to test the A³X bridge directive handling for training.

import asyncio
from a3x.a3net.core.memory_bank import MemoryBank
# from a3x.a3net.integration.a3x_bridge import handle_directive
from a3x.a3net.integration.a3lang_interpreter import interpret_a3l_line

# --- Setup (Assume MemoryBank is initialized elsewhere or mock it) ---
# For testing, create a mock or temporary MemoryBank
MEMORY_BANK = MemoryBank(save_dir="./temp_memory_test", export_dir="./temp_repo_test")

async def main():
    # --- Example Directives ---
    create_directive_str = "criar fragmento 'test_train_frag' tipo 'neural' input_dim 10 num_classes 2 task_name 'binary_classification'"
    train_directive_str = "treinar fragmento 'test_train_frag' task_name 'binary_classification' epochs 5"
    ask_directive_str = "ask fragmento 'test_train_frag' com [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]"

    directives_to_run = [
        create_directive_str,
        train_directive_str,
        ask_directive_str
    ]

    print("--- Running Test Directives ---")
    for i, cmd_str in enumerate(directives_to_run):
        print(f"\n[Step {i+1}] Executing: {cmd_str}")
        directive_dict = interpret_a3l_line(cmd_str)
        if directive_dict:
             print(f"  Interpreted: {directive_dict}")
             # --- Cannot call handle_directive directly anymore ---
             # It needs to be passed in or called via the message queue system
             print("  <<< SKIPPING EXECUTION (handle_directive moved) >>>")
             # result = await handle_directive(directive_dict, memory_bank=MEMORY_BANK)
             # print(f"  Result: {result}")
             # --- --------------------------------------------- ---
        else:
            print(f"  ERROR: Could not interpret directive: {cmd_str}")
            
    print("\n--- Test Finished ---")

if __name__ == "__main__":
    # Basic setup for running async main
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Test interrupted.") 