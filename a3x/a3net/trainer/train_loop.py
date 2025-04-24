import torch
import torch.nn as nn
import torch.optim as optim
from typing import Iterable, Tuple, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def save_training_history(history: List[float], fragment_id: str):
    """Save training history to a JSON file."""
    history_dir = Path("data/training_history")
    history_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = history_dir / f"{fragment_id}_{timestamp}.json"
    
    with open(history_file, 'w') as f:
        json.dump({
            "fragment_id": fragment_id,
            "timestamp": timestamp,
            "loss_history": history
        }, f, indent=2)
    
    logger.info(f"Saved training history to {history_file}")

def train_fragment_cell(
    cell: nn.Module, 
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]], 
    epochs: int, 
    optimizer: optim.Optimizer,
    loss_fn: nn.Module = nn.MSELoss(),
    early_stopping: Optional[EarlyStopping] = None,
    fragment_id: str = "unknown_fragment"
):
    """Performs a basic training loop for a given cell (Module)."""
    cell.train() # Set the module to training mode
    logger.info(f"Starting training for {epochs} epochs...")
    
    loss_history = []
    best_loss = float('inf')
    
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
                logger.error(f"Error calculating loss (batch {i}): {e}")
                logger.error(f"  Output shape: {outputs.shape}, Output dtype: {outputs.dtype}")
                logger.error(f"  Target shape: {targets.shape}, Target dtype: {targets.dtype}")
                logger.error(f"Skipping batch {i} due to loss calculation error.")
                continue 

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        loss_history.append(avg_epoch_loss)
        
        logger.info(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}")
        
        # Early stopping check
        if early_stopping and early_stopping(avg_epoch_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
            
        # Update best loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
    
    # Save training history
    save_training_history(loss_history, fragment_id)
    
    logger.info("Training finished.")
    return loss_history 