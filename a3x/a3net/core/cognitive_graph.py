import torch
from typing import List
import logging

# A³Net core imports
from .fragment_cell import FragmentCell
from .memory_bank import MemoryBank

logger = logging.getLogger(__name__) # Use standard logging

class CognitiveGraph(torch.nn.Module):
    """Connects multiple FragmentCells loaded from a MemoryBank and orchestrates data flow.
    
    Operates in inference mode (no gradients computed during forward pass).
    """
    def __init__(self, memory_bank: MemoryBank, fragment_ids: List[str]):
        """Initializes the graph by loading specified fragments from the MemoryBank.
        
        Args:
            memory_bank: The MemoryBank instance containing saved fragments.
            fragment_ids: A list of fragment IDs to load and connect sequentially.
        """
        super().__init__()
        
        loaded_fragments: List[FragmentCell] = []
        print(f"[CognitiveGraph] Initializing with fragment IDs: {fragment_ids}")
        for fragment_id in fragment_ids:
            fragment = memory_bank.load(fragment_id)
            if fragment is not None:
                loaded_fragments.append(fragment)
                print(f"[CognitiveGraph] Successfully loaded fragment: {fragment_id}")
            else:
                # Log a warning if a fragment ID is not found
                logger.warning(f"[CognitiveGraph] Fragment ID '{fragment_id}' not found in MemoryBank. Skipping.")
                print(f"[CognitiveGraph] Warning: Fragment ID '{fragment_id}' not found in MemoryBank. Skipping.") # Also print

        if not loaded_fragments:
            logger.warning("[CognitiveGraph] No fragments were loaded. The graph is empty.")
            print("[CognitiveGraph] Warning: No fragments were loaded. The graph is empty.")

        # Use ModuleList to ensure fragments are registered correctly for PyTorch
        self.fragments = torch.nn.ModuleList(loaded_fragments)
        print(f"[CognitiveGraph] Initialization complete. Loaded {len(self.fragments)} fragments.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input tensor sequentially through all loaded fragments in inference mode."""
        if not self.fragments:
            logger.warning("[CognitiveGraph] Forward pass called on an empty graph. Returning input tensor.")
            print("[CognitiveGraph] Forward pass called on an empty graph. Returning input tensor.")
            return x
            
        # Execute the forward pass without calculating gradients
        with torch.no_grad():
            current_tensor = x
            print(f"[CognitiveGraph] Starting forward pass with input shape: {current_tensor.shape}")
            for i, fragment in enumerate(self.fragments):
                try:
                    # Ensure fragment is in evaluation mode
                    fragment.eval() 
                    output_tensor = fragment(current_tensor)
                    print(f"[CognitiveGraph] Passed through fragment {i+1}/{len(self.fragments)}. Output shape: {output_tensor.shape}")
                    current_tensor = output_tensor # Pass output of one to the next
                except Exception as e:
                    logger.error(f"[CognitiveGraph] Error during forward pass at fragment {i+1}: {e}", exc_info=True)
                    print(f"[CognitiveGraph] Error during forward pass at fragment {i+1}: {e}. Aborting forward pass.")
                    # Depending on desired behavior, could return partial result or raise error
                    # For now, return the tensor as it was before the error
                    return current_tensor 
                    
        print(f"[CognitiveGraph] Forward pass completed. Final output shape: {current_tensor.shape}")
        return current_tensor 