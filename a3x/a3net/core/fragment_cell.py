import torch
import torch.nn as nn

class FragmentCell(nn.Module):
    """A simple neural cell representing a computational fragment."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initializes the cell with configurable dimensions."""
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input tensor through the cell's layers."""
        # Add dimension check if needed
        if x.shape[-1] != self.linear1.in_features:
            raise ValueError(f"Input tensor last dimension ({x.shape[-1]}) does not match cell input_dim ({self.linear1.in_features})")
            
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x 