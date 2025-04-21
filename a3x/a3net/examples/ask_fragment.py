# Example script to test the 'ask' directive using a NeuralLanguageFragment.

import torch
import random # For generating the random input list

# Updated import path after moving a3net into a3x
from a3x.a3net.integration.a3x_bridge import handle_directive, MEMORY_BANK

if __name__ == '__main__':
    print("--- Running A³Net 'ask' Directive Example ---")

    # --- Ensure the target fragment exists ---
    # This example assumes 'frag_decisor' was trained and saved previously
    # (e.g., by running test_language_fragment.py)
    target_fragment_id = "frag_decisor"
    print(f"\nChecking if target fragment '{target_fragment_id}' exists in MemoryBank...")
    if MEMORY_BANK.load(target_fragment_id) is None:
        print(f"Error: Fragment '{target_fragment_id}' not found. Please run the corresponding training script first.")
        print("Example: python -m a3net.examples.test_language_fragment")
        print("\n--- A³Net 'ask' Directive Example Aborted ---")
        exit()
    else:
         # Reload clears the internal print flag, so list again after loading
         print(f"Fragment '{target_fragment_id}' found.") 
         print(f"Current fragments in bank: {MEMORY_BANK.list()}")

    # --- Prepare Input Data ---
    input_dim = 128 # Must match the expected input dim of frag_decisor
    # Generate a random list of floats
    random_input_list = [random.random() for _ in range(input_dim)]
    # Alternative using torch:
    # input_tensor = torch.randn(input_dim)
    # random_input_list = input_tensor.tolist()
    print(f"\nGenerated random input vector (list) with {len(random_input_list)} elements.")

    # --- Construct the Directive ---
    ask_directive = {
        "type": "ask",
        "fragment_id": target_fragment_id,
        "input": random_input_list
    }
    
    print(f"\nConstructed 'ask' directive:")
    # Print only first few elements of input for brevity
    print(f"  type: {ask_directive['type']}")
    print(f"  fragment_id: {ask_directive['fragment_id']}")
    print(f"  input: [{str(ask_directive['input'][:5])[:-1]} ...]") # Show first 5

    # --- Call the Handler ---
    print("\nCalling handle_directive...")
    result = handle_directive(ask_directive)

    # --- Print Result ---
    print("\nResult received from handle_directive:")
    print(result)

    print("\n--- A³Net 'ask' Directive Example Finished ---") 