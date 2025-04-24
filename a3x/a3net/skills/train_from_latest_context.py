import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import joblib
import asyncio

from ..core.context_store import ContextStore
from ..core.dummy_fragment import DummyFragmentCell
from ..trainer.dataset_builder import build_dataset_from_context
from ..trainer.train_loop import train_fragment_cell, EarlyStopping

logger = logging.getLogger(__name__)

async def train_from_latest_context(
    context_store: ContextStore,
    fragment_id: str,
    max_samples: int = 100,
    epochs: int = 10,
    patience: int = 5
) -> Dict[str, Any]:
    """
    Train a fragment using the latest context entries.
    
    Args:
        context_store: The context store to get training data from
        fragment_id: ID of the fragment to train
        max_samples: Maximum number of samples to use
        epochs: Number of training epochs
        patience: Patience for early stopping
        
    Returns:
        Dictionary with training results
    """
    try:
        # 1. Get or create fragment
        model_path = Path(f"models/{fragment_id}.joblib")
        if model_path.exists():
            fragment = joblib.load(model_path)
            logger.info(f"Loaded existing fragment {fragment_id}")
        else:
            fragment = DummyFragmentCell(fragment_id=fragment_id)
            logger.info(f"Created new fragment {fragment_id}")
        
        # 2. Build dataset from context
        dataset = await build_dataset_from_context(
            fragment_id=fragment_id,
            context_store=context_store,
            model=None,  # We'll use random embeddings for testing
            max_samples=max_samples
        )
        
        if not dataset:
            raise ValueError(f"No training data found for fragment {fragment_id}")
            
        logger.info(f"Built dataset with {len(dataset)} samples")
        
        # 3. Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True
        )
        
        # 4. Set up training
        optimizer = optim.Adam(fragment.parameters())
        loss_fn = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=patience)
        
        # 5. Train
        logger.info(f"Starting training for {epochs} epochs")
        loss_history = train_fragment_cell(
            cell=fragment,
            dataset=dataloader,
            epochs=epochs,
            optimizer=optimizer,
            loss_fn=loss_fn,
            early_stopping=early_stopping,
            fragment_id=fragment_id
        )
        
        # 6. Save model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fragment, model_path)
        logger.info(f"Saved trained model to {model_path}")
        
        return {
            "success": True,
            "fragment_id": fragment_id,
            "num_samples": len(dataset),
            "final_loss": loss_history[-1] if loss_history else None,
            "model_path": str(model_path)
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "fragment_id": fragment_id
        } 