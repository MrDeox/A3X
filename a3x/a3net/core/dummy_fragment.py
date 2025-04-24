import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class DummyFragmentCell(nn.Module):
    """A simple neural network fragment for testing purposes."""
    
    def __init__(
        self,
        input_dim: int = 384,  # Default embedding size
        hidden_dim: int = 128,
        output_dim: int = 3,   # Default for 3 classes
        fragment_id: str = "dummy_fragment",
        description: str = "Dummy fragment for testing"
    ):
        super().__init__()
        self.fragment_id = fragment_id
        self.description = description
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1)
            confidence = torch.max(probs, dim=-1).values
            
        return {
            "prediction": pred_class.item(),
            "confidence": confidence.item(),
            "probabilities": probs.tolist()
        } 