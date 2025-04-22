# Example script to test the ReflectiveLanguageFragment.

import torch
from typing import List, Tuple

# Updated import paths after moving a3net into a3x
from a3x.a3net.core.reflective_language_fragment import ReflectiveLanguageFragment
from a3x.a3net.integration.a3x_bridge import MEMORY_BANK

if __name__ == '__main__':
    print("--- Running A³Net Reflective Language Fragment Example ---")

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
        y = torch.randint(0, num_classes, (1,)) # Shape [1]
        dataset.append((x, y))
        
    print(f"Generated dataset with {len(dataset)} samples.")
    print(f"First sample target type: {dataset[0][1].dtype}, shape: {dataset[0][1].shape}")

    # --- Instantiate the Reflective Fragment ---
    fragment_id = "frag_reflector" # New ID for this fragment
    description = "Fragmento que responde sim/não/reavaliar e explica o porquê"
    print(f"\nInstantiating ReflectiveLanguageFragment:")
    print(f"  ID: {fragment_id}")
    print(f"  Desc: {description}")
    
    reflective_fragment = ReflectiveLanguageFragment(
        fragment_id=fragment_id,
        description=description,
        input_dim=input_dim,
        num_classes=num_classes # Uses default labels
    )

    # --- Train the Fragment ---
    # Uses the train_on method inherited from NeuralLanguageFragment
    print("\nStarting training...")
    training_successful = False
    try:
        reflective_fragment.train_on(dataset=dataset, epochs=10, learning_rate=0.001)
        print("Training finished.")
        training_successful = True
    except Exception as e:
        print(f"Training failed: {e}")
        training_successful = False

    # --- Save Fragment to Memory Bank (if training was successful) ---
    if training_successful:
        print(f"\nAttempting to save fragment '{fragment_id}' to MemoryBank...")
        try:
            MEMORY_BANK.save(fragment_id, reflective_fragment)
            print(f"Fragment '{fragment_id}' saved successfully.")
        except Exception as e:
            print(f"Error saving fragment '{fragment_id}': {e}")
    else:
        print("\nSkipping fragment save due to training failure.")

    # --- Test Prediction ---
    print("\nTesting prediction after training...")
    # Generate a new random input vector (add batch dimension)
    test_input = torch.randn(1, input_dim) 
    print(f"Generated test input with shape: {test_input.shape}")
    
    # Get the prediction dictionary (includes explanation)
    prediction_result = reflective_fragment.predict(test_input)
    
    print(f"\nPrediction Result Dictionary:")
    print(prediction_result)
    print(f"\n  Output: {prediction_result.get('output')}")
    print(f"  Explanation: {prediction_result.get('explanation')}")

    print("\n--- A³Net Reflective Language Fragment Example Finished ---") 