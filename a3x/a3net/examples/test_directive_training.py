# Example script to test the A³X bridge directive handling for training.

# Updated import path after moving a3net into a3x
from a3x.a3net.integration.a3x_bridge import handle_directive

if __name__ == '__main__':
    print("--- Running A³Net Directive Training Example ---")

    # Define the directive dictionary
    directive = {
        "type": "train_fragment",
        "goal": "example training with neural fragment", # Corrected typo
        "context_id": "simulated_context_run_01",
        "input_dim": 128,   # Standard input dimension from FragmentCell
        "output_dim": 1,    # Standard output dimension from FragmentCell
        "hidden_dim": 64,   # Default hidden dimension
        "epochs": 10,       # Number of training epochs
        "learning_rate": 0.001 # Learning rate for Adam optimizer
    }

    print(f"\nSending directive:\n{directive}\n")

    # Call the handler function
    handle_directive(directive)

    print("\n--- A³Net Directive Training Example Finished ---") 