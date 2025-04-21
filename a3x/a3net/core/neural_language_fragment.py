import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import logging
import random # For shuffling dataset
import math # For splitting dataset
import asyncio # For async evaluate method

# Corrected relative import path to trainer directory
from ..trainer.train_loop import train_fragment_cell
# Import dataset builder
from ..trainer.dataset_builder import build_dataset_from_context

logger = logging.getLogger(__name__)

class NeuralLanguageFragment(nn.Module):
    """A neural fragment designed for classification tasks based on input vectors.
    
    Example: Classifying user intent (Yes/No/Re-evaluate) based on an embedding.
    """
    
    DEFAULT_ID_TO_LABEL = {0: "SIM", 1: "NÃO", 2: "REAVALIAR"}
    
    def __init__(self, 
                 fragment_id: str, 
                 description: str, 
                 input_dim: int = 128, 
                 hidden_dim: int = 64, 
                 num_classes: int = 3, # Default based on DEFAULT_ID_TO_LABEL
                 id_to_label: Optional[Dict[int, str]] = None,
                 associated_task_name: Optional[str] = None): # <<< Added task association
        """Initializes the NeuralLanguageFragment.

        Args:
            fragment_id: A unique identifier for this fragment.
            description: A textual description of the fragment's purpose.
            input_dim: The dimensionality of the input vectors.
            hidden_dim: The dimensionality of the hidden layer.
            num_classes: The number of output classes.
            id_to_label: Optional dictionary mapping class indices to string labels.
                       Defaults to {0: "SIM", 1: "NÃO", 2: "REAVALIAR"} if num_classes is 3.
            associated_task_name: The specific task name (used for dataset loading). <<< Added
        """
        super().__init__()
        self.fragment_id = fragment_id
        self.description = description
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.associated_task_name = associated_task_name # <<< Store task name

        # --- Internal Sub-network ---
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_classes) # Output layer size matches num_classes

        # --- Class Label Mapping ---
        if id_to_label is not None:
            if len(id_to_label) != num_classes:
                 raise ValueError(f"Length of id_to_label ({len(id_to_label)}) must match num_classes ({num_classes})")
            self.id_to_label = id_to_label
        elif num_classes == 3: # Default mapping only if num_classes is 3
            self.id_to_label = self.DEFAULT_ID_TO_LABEL
        else:
            # Create a generic mapping if no specific one is provided and num_classes isn't 3
            self.id_to_label = {i: f"CLASS_{i}" for i in range(num_classes)}
            logger.warning(f"No id_to_label provided for {num_classes} classes. Using generic labels: {self.id_to_label}")

        print(f"[NeuralLangFrag '{self.fragment_id}'] Initialized: {input_dim}->{hidden_dim}->{num_classes}")
        print(f"[NeuralLangFrag '{self.fragment_id}'] Label map: {self.id_to_label}")
        if self.associated_task_name:
            print(f"[NeuralLangFrag '{self.fragment_id}'] Associated Task: {self.associated_task_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input tensor through the network and returns raw logits."""
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Input tensor last dimension ({x.shape[-1]}) does not match fragment input_dim ({self.input_dim})")
        
        x = self.linear1(x)
        x = self.activation(x)
        logits = self.linear2(x)
        return logits

    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """Predicts the most likely class label and confidence for the input tensor."""
        self.eval() # Ensure model is in evaluation mode
        prediction_result = {}
        with torch.no_grad():
            logits = self(x)
            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=-1)
            # Get the highest probability and its index
            max_probability, predicted_index = torch.max(probabilities, dim=-1)
            
            predicted_index = predicted_index.item() # Assumes batch size 1
            max_probability = max_probability.item() # Assumes batch size 1
            
            # Map index to label string
            label = self.id_to_label.get(predicted_index, "UNKNOWN_CLASS")
            
            prediction_result['output'] = label
            prediction_result['confidence'] = max_probability
            
        return prediction_result # Return dict {output: str, confidence: float}

    async def train_on_task(self, task_name: str, epochs: int = 10, learning_rate: float = 0.001) -> bool:
        """Loads dataset for a task and trains the fragment using train_on.

        Returns:
            bool: True if training was initiated and reported success, False otherwise.
        """
        logger.info(f"[{self.fragment_id}] Received request to train on task '{task_name}' for {epochs} epochs.")
        try:
            # Load/build dataset (synchronous, run in executor)
            loop = asyncio.get_running_loop()
            dataset = await loop.run_in_executor(
                None, 
                build_dataset_from_context, 
                task_name, 
                self.num_classes 
            )

            if not dataset:
                logger.error(f"[{self.fragment_id}] Could not load or build dataset for task '{task_name}'. Training aborted.")
                return False

            # Initiate training (synchronous, run in executor)
            success = await loop.run_in_executor(
                None, 
                self.train_on, 
                dataset, 
                epochs, 
                learning_rate
            )
            
            logger.info(f"[{self.fragment_id}] Training on task '{task_name}' completed. Reported success: {success}")
            return success

        except Exception as e:
            logger.error(f"[{self.fragment_id}] Error during train_on_task for '{task_name}': {e}", exc_info=True)
            return False

    def train_on(self, 
                 dataset: List[Tuple[torch.Tensor, torch.Tensor]], 
                 epochs: int = 10, 
                 learning_rate: float = 0.001):
        """Trains this specific fragment using the provided data.

        Uses CrossEntropyLoss suitable for classification.

        Args:
            dataset: A list of (input_tensor, target_class_index_tensor) tuples.
                     Target tensors should contain class indices (long integers).
            epochs: The number of training epochs.
            learning_rate: The learning rate for the Adam optimizer.
        """
        print(f"[NeuralLangFrag '{self.fragment_id}'] Starting training...")
        logger.info(f"Starting training for NeuralLanguageFragment '{self.fragment_id}'")
        
        # --- Validation for Classification Data ---
        if not dataset:
            print("[NeuralLangFrag] Error: Training dataset is empty.")
            logger.error(f"Training dataset for {self.fragment_id} is empty.")
            return False
            
        # Check target tensor type in the first sample (assuming consistency)
        _, first_target = dataset[0]
        if first_target.dtype != torch.long:
             # Attempt conversion or raise error - CrossEntropyLoss expects Long targets
             try:
                 # Create a new dataset list with converted targets
                 converted_dataset = []
                 for i, (inp, tar) in enumerate(dataset):
                     if tar.dtype != torch.long:
                         # Ensure target is scalar or single element before converting
                         if tar.numel() == 1:
                            converted_dataset.append((inp, tar.long().squeeze())) # Squeeze to make it scalar if needed
                         else:
                             raise TypeError(f"Target tensor at index {i} has multiple elements ({tar.numel()}) and cannot be automatically converted to a class index.")
                     else:
                         converted_dataset.append((inp, tar.squeeze())) # Ensure it's scalar like if already long
                 dataset = converted_dataset # Replace original dataset
                 logger.warning(f"Converted target tensors to torch.long for fragment {self.fragment_id}.")
                 print("[NeuralLangFrag] Warning: Converted target tensors to torch.long.")
             except Exception as e:
                 print(f"[NeuralLangFrag] Error: Target tensors must be torch.long (class indices) for CrossEntropyLoss. Conversion failed: {e}")
                 logger.error(f"Target tensor type mismatch for {self.fragment_id}: {first_target.dtype}. Conversion failed.", exc_info=True)
                 return False
        else:
            # Ensure targets are scalar-like if they are already long
             dataset = [(inp, tar.squeeze()) for inp, tar in dataset]

        # --- Setup Optimizer and Loss ---
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # Use CrossEntropyLoss for multi-class classification (expects raw logits)
        loss_fn = nn.CrossEntropyLoss()
        
        # --- Call Generic Trainer (if suitable) ---
        # Note: train_fragment_cell needs adjustment if it only uses MSELoss.
        # If train_fragment_cell can accept loss_fn, we can use it:
        try:
            print(f"[NeuralLangFrag '{self.fragment_id}'] Using CrossEntropyLoss.")
            # We pass the specific loss function for classification
            train_fragment_cell(
                cell=self, 
                dataset=dataset, 
                epochs=epochs, 
                optimizer=optimizer,
                loss_fn=loss_fn # Pass CrossEntropyLoss
            )
            print(f"[NeuralLangFrag '{self.fragment_id}'] Training finished.")
            logger.info(f"Training completed for NeuralLanguageFragment '{self.fragment_id}'")
            return True # Indicate success
        except Exception as e:
            print(f"[NeuralLangFrag '{self.fragment_id}'] Error during training: {e}")
            logger.exception(f"Error during training for {self.fragment_id}")
            return False # Indicate failure

        # --- Alternative: Internal Training Loop (if train_fragment_cell is unsuitable) ---
        # Uncomment and adapt this section if train_fragment_cell cannot handle CrossEntropyLoss
        # self.train() # Set model to training mode
        # for epoch in range(epochs):
        #     epoch_loss = 0.0
        #     num_batches = 0
        #     for inputs, targets in dataset:
        #         optimizer.zero_grad()
        #         outputs = self(inputs) # Get logits
        #         # Ensure targets are the correct shape for CrossEntropyLoss (usually [N])
        #         # and outputs are [N, C]
        #         if len(inputs.shape) == 1: # If single sample, add batch dim
        #             inputs = inputs.unsqueeze(0)
        #             outputs = outputs.unsqueeze(0)
        #             targets = targets.unsqueeze(0)
        #         
        #         loss = loss_fn(outputs, targets)
        #         loss.backward()
        #         optimizer.step()
        #         epoch_loss += loss.item()
        #         num_batches += 1
        #     avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        #     print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}")
        # print(f"[NeuralLangFrag '{self.fragment_id}'] Training finished.") 

    # <<< NEW EVALUATION METHOD >>>
    async def evaluate(self, task_name: str, test_split_ratio: float = 0.2) -> Dict[str, Any]:
        """Evaluates the fragment's performance on a test split of the task dataset."""
        logger.info(f"[{self.fragment_id}] Starting evaluation on task '{task_name}' (Test Split: {test_split_ratio:.1%})")
        results = {"task_name": task_name, "fragment_id": self.fragment_id, "status": "error", "message": ""}

        try:
            # Load the full dataset (synchronous function, run in executor)
            loop = asyncio.get_running_loop()
            full_dataset = await loop.run_in_executor(
                None, 
                build_dataset_from_context, 
                task_name, 
                self.num_classes
            )

            if not full_dataset:
                results["message"] = f"Could not load or build dataset for task '{task_name}'."
                logger.error(f"[{self.fragment_id}] Evaluation failed: {results['message']}")
                return results

            dataset_size = len(full_dataset)
            if dataset_size < 5: # Arbitrary minimum size for meaningful split
                results["message"] = f"Dataset too small ({dataset_size} samples) for evaluation split."
                logger.warning(f"[{self.fragment_id}] Evaluation warning: {results['message']}")
                # Optionally evaluate on the whole dataset if needed, or just return warning
                results["status"] = "warning_small_dataset" 
                # Fall through to evaluate on whole dataset for now, or return here
                test_set = full_dataset # Use whole dataset for testing if too small to split
                test_set_size = dataset_size
            else:
                # Shuffle and split the dataset
                random.shuffle(full_dataset)
                split_index = math.ceil(dataset_size * (1.0 - test_split_ratio))
                # train_set = full_dataset[:split_index] # Not used in evaluation
                test_set = full_dataset[split_index:]
                test_set_size = len(test_set)
                if test_set_size == 0:
                    results["message"] = "Test set size is 0 after splitting."
                    logger.error(f"[{self.fragment_id}] Evaluation failed: {results['message']}")
                    return results


            logger.info(f"[{self.fragment_id}] Evaluating on {test_set_size} test samples.")

            # --- Perform Evaluation ---
            self.eval() # Set model to evaluation mode
            correct_predictions = 0
            total_loss = 0.0
            loss_fn = nn.CrossEntropyLoss() # Use the same loss for consistency checking

            with torch.no_grad():
                for input_tensor, target_tensor in test_set:
                    try:
                        # Ensure tensors are on the correct device (if using GPU)
                        # device = next(self.parameters()).device
                        # input_tensor = input_tensor.to(device)
                        # target_tensor = target_tensor.to(device)

                        # Add batch dimension if necessary (assuming dataset provides single samples)
                        if len(input_tensor.shape) == 1: 
                            input_tensor = input_tensor.unsqueeze(0)
                        if len(target_tensor.shape) == 0:
                            target_tensor = target_tensor.unsqueeze(0)

                        logits = self(input_tensor)
                        
                        # Calculate loss
                        loss = loss_fn(logits, target_tensor)
                        total_loss += loss.item()

                        # Get prediction
                        predicted_index = torch.argmax(logits, dim=-1)
                        
                        # Compare prediction with true label
                        if predicted_index.item() == target_tensor.item():
                            correct_predictions += 1
                    except Exception as eval_loop_err:
                         logger.error(f"[{self.fragment_id}] Error during evaluation loop for one sample: {eval_loop_err}", exc_info=True)
                         # Continue with next sample? Or fail fast? Let's continue for now.

            # Calculate metrics
            accuracy = (correct_predictions / test_set_size) if test_set_size > 0 else 0.0
            average_loss = (total_loss / test_set_size) if test_set_size > 0 else 0.0

            results.update({
                "status": "success",
                "accuracy": accuracy,
                "average_loss": average_loss,
                "test_set_size": test_set_size,
                "correct_predictions": correct_predictions,
                "message": f"Evaluation successful. Accuracy: {accuracy:.2%}"
            })
            logger.info(f"[{self.fragment_id}] Evaluation completed for task '{task_name}'. Accuracy: {accuracy:.4f}, Avg Loss: {average_loss:.4f}")

        except Exception as e:
            results["message"] = f"Unexpected error during evaluation: {e}"
            logger.exception(f"[{self.fragment_id}] Unexpected error during evaluation for task '{task_name}'")
            results["status"] = "error"

        return results
    # <<< END NEW EVALUATION METHOD >>>

    def generate_reflection_a3l(self) -> str:
        """Generates a human-readable A3L-like string summarizing the fragment."""
        
        parts = [
            f"fragmento '{self.fragment_id}' ({self.__class__.__name__})",
            f"recebe input com dimensão {self.input_dim}",
            f"processa com camada oculta de dimensão {self.hidden_dim}",
            f"e retorna uma das {self.num_classes} classes possíveis"
        ]
        
        if self.id_to_label:
            class_names = list(self.id_to_label.values())
            parts.append(f"com rótulos: {class_names}")
        else:
            parts.append("(sem rótulos definidos)")
            
        if hasattr(self, 'description') and self.description:
            parts.append(f"descrito como: '{self.description}'")
            
        return ", ".join(parts) + "."

# Example usage (if run directly)
if __name__ == '__main__':
    pass # Add pass to fix indentation error
    # Example usage code can be added here later if needed
    # frag = NeuralLanguageFragment('test_frag', 'A test fragment', 10, 5, 3)
    # print(frag.generate_reflection_a3l()) 