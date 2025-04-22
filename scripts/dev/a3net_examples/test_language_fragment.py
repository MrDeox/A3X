# Example script to test the NeuralLanguageFragment.

import torch
from typing import List, Tuple

# Updated import paths after moving a3net into a3x
from a3x.a3net.core.neural_language_fragment import NeuralLanguageFragment
from a3x.a3net.integration.a3x_bridge import MEMORY_BANK

if __name__ == '__main__':
    print("--- Running A³Net Neural Language Fragment Example ---")

    # --- Generate Synthetic Classification Data ---
    num_samples = 300
    input_dim = 128
    num_classes = 3 # Corresponds to SIM, NÃO, REAVALIAR
    
    print(f"\nGenerating {num_samples} synthetic samples...")
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(num_samples):
        # Random input vector
        x = torch.randn(input_dim)
        # Random target class index (0, 1, or 2)
        # Create as a tensor, train_on expects tensors and will convert to long
        y = torch.randint(0, num_classes, (1,)) # Shape [1]
        dataset.append((x, y))
        
    print(f"Generated dataset with {len(dataset)} samples.")
    print(f"First sample target type: {dataset[0][1].dtype}, shape: {dataset[0][1].shape}")

    # --- Instantiate the Fragment ---
    fragment_id = "frag_decisor"
    description = "Fragmento que responde sim/não/reavaliar"
    print(f"\nInstantiating NeuralLanguageFragment:")
    print(f"  ID: {fragment_id}")
    print(f"  Desc: {description}")
    
    language_fragment = NeuralLanguageFragment(
        fragment_id=fragment_id,
        description=description,
        input_dim=input_dim,
        num_classes=num_classes # Uses default labels {0: SIM, 1: NÃO, 2: REAVALIAR}
    )

    # --- Train the Fragment ---
    print("\nStarting training...")
    try:
        language_fragment.train_on(dataset=dataset, epochs=10, learning_rate=0.001)
        print("Training finished.")
        training_successful = True
    except Exception as e:
        print(f"Training failed: {e}")
        training_successful = False

    # --- Save Fragment to Memory Bank (if training was successful) ---
    if training_successful:
        print(f"\nAttempting to save fragment '{fragment_id}' to MemoryBank...")
        try:
            MEMORY_BANK.save(fragment_id, language_fragment)
            print(f"Fragment '{fragment_id}' saved successfully.")
        except Exception as e:
            print(f"Error saving fragment '{fragment_id}': {e}")
    else:
        print("\nSkipping fragment save due to training failure.")

    # --- Test Prediction (Optional: Can still test even if save failed) ---
    print("\nTesting prediction...")
    test_input = torch.randn(1, input_dim) 
    print(f"Generated test input with shape: {test_input.shape}")
    
    # Get the prediction
    predicted_label = language_fragment.predict(test_input)
    
    print(f"\nPredicted Label for test input: {predicted_label}")

    print("\n--- A³Net Neural Language Fragment Example Finished ---") 