import torch
import torch.nn as nn
import torch.optim as optim
from typing import Iterable, Tuple

def train_fragment_cell(
    cell: nn.Module, 
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]], 
    epochs: int, 
    optimizer: optim.Optimizer,
    loss_fn: nn.Module = nn.MSELoss()
):
    """Performs a basic training loop for a given cell (Module)."""
    cell.train() # Set the module to training mode
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for i, (inputs, targets) in enumerate(dataset):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = cell(inputs)
            
            # Calculate loss
            try:
                loss = loss_fn(outputs, targets)
            except Exception as e:
                # Add more context to shape mismatch errors
                print(f"Error calculating loss (batch {i}): {e}")
                print(f"  Output shape: {outputs.shape}, Output dtype: {outputs.dtype}")
                print(f"  Target shape: {targets.shape}, Target dtype: {targets.dtype}")
                # Re-raise the exception or continue to next batch
                # For now, let's continue to potentially process other batches
                print(f"Skipping batch {i} due to loss calculation error.")
                continue 

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}")

    print("Training finished.") 