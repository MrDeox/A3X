# Example script to test the 'ask' directive using a NeuralLanguageFragment.

import torch
import random # For generating the random input list
import asyncio
from a3x.a3net.core.memory_bank import MemoryBank
from a3x.a3net.integration.a3lang_interpreter import interpret_a3l_line

# Updated import path after moving a3net into a3x
# from a3x.a3net.integration.a3x_bridge import handle_directive, MEMORY_BANK

# --- Setup (Assume MemoryBank is initialized and fragment exists) ---
# For testing, create a mock or temporary MemoryBank and potentially a dummy fragment
MEMORY_BANK = MemoryBank(save_dir="./temp_memory_test", export_dir="./temp_repo_test")

# Example: Create a dummy fragment for testing if needed
async def setup_dummy_fragment():
    from a3x.a3net.core.neural_language_fragment import NeuralLanguageFragment
    dummy_frag = NeuralLanguageFragment(
        fragment_id="test_ask_frag", 
        input_dim=10, # Example dim
        num_classes=3  # Example classes
    )
    MEMORY_BANK.save("test_ask_frag", dummy_frag)
    print("Dummy fragment 'test_ask_frag' created.")

async def main():
    # Ensure dummy fragment exists (optional, depends on test setup)
    # await setup_dummy_fragment() 
    
    # --- Example Directive ---
    # Note: 'ask' directive parsing/handling might differ now
    ask_directive_str = "ask fragmento 'test_ask_frag' com [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]"
    # Alternative format if needed: ask_directive_str = "ASK fragment_id='test_ask_frag' input_data=[...]"

    print("--- Running Ask Test Directive ---")
    print(f"Executing: {ask_directive_str}")
    directive_dict = interpret_a3l_line(ask_directive_str)
    if directive_dict:
         print(f"  Interpreted: {directive_dict}")
         # --- Cannot call handle_directive directly anymore ---
         print("  <<< SKIPPING EXECUTION (handle_directive moved) >>>")
         # result = await handle_directive(directive_dict, memory_bank=MEMORY_BANK)
         # print(f"  Result: {result}")
         # -----------------------------------------------
    else:
        print(f"  ERROR: Could not interpret directive: {ask_directive_str}")
            
    print("\n--- Ask Test Finished ---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Test interrupted.") 