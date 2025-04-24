import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import logging

from ..core.dummy_fragment import DummyFragmentCell
from .dataset_builder import create_or_update_dataset_jsonl
from .train_loop import train_fragment_cell

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_dummy_dataset(num_samples: int = 100) -> List[Dict[str, str]]:
    """Generate a dummy dataset for testing."""
    examples = []
    for i in range(num_samples):
        # Generate random text and labels
        text = f"Example text {i} with some random content"
        label = np.random.choice(["SIM", "NÃO", "REAVALIAR"])
        examples.append({
            "text": text,
            "label": label
        })
    return examples

class DummyDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for testing."""
    def __init__(self, examples: List[Dict[str, str]], embedding_model):
        self.examples = examples
        self.embedding_model = embedding_model
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        # Generate random embeddings for testing
        embedding = torch.randn(384)  # Match default input_dim
        # Convert label to tensor
        label_map = {"SIM": 0, "NÃO": 1, "REAVALIAR": 2}
        label = torch.tensor(label_map[example["label"]])
        return embedding, label

def test_training_cycle():
    """Test the complete training cycle."""
    try:
        # 1. Create dummy fragment
        fragment = DummyFragmentCell()
        logger.info("Created dummy fragment")
        
        # 2. Generate and save dataset
        examples = generate_dummy_dataset(100)
        create_or_update_dataset_jsonl("test_training", examples)
        logger.info("Generated and saved dummy dataset")
        
        # 3. Create dataset loader
        # For testing, we'll use random embeddings
        dataset = DummyDataset(examples, None)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 4. Train the fragment
        optimizer = optim.Adam(fragment.parameters())
        loss_fn = nn.CrossEntropyLoss()
        
        logger.info("Starting training...")
        train_fragment_cell(
            cell=fragment,
            dataset=dataloader,
            epochs=10,
            optimizer=optimizer,
            loss_fn=loss_fn
        )
        logger.info("Training completed")
        
        # 5. Save and load model
        model_path = Path("models/test_fragment.joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(fragment, model_path)
        logger.info(f"Saved model to {model_path}")
        
        loaded_fragment = joblib.load(model_path)
        logger.info("Successfully loaded model")
        
        # 6. Test prediction
        test_input = torch.randn(384)  # Random input
        prediction = loaded_fragment.predict(test_input)
        logger.info(f"Test prediction: {prediction}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_training_cycle()
    if success:
        logger.info("Training cycle test completed successfully!")
    else:
        logger.error("Training cycle test failed!") 