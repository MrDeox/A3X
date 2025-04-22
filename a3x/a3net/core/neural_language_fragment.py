import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import logging
import random # For shuffling dataset
import math # For splitting dataset
import asyncio # For async evaluate method
import collections # <<< Add collections for deque
from ..core.context_store import ContextStore # <<< Import ContextStore for type hint
import time # For logging timestamp
import functools # <<< Add functools for partial >>>

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

        # --- State for dynamic stopping ---
        self.recent_val_accuracies = collections.deque(maxlen=3) # Stores last 3 val accuracies
        self.min_accuracy_gain = 0.001 # Threshold to stop training

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

    # --- Helper for dynamic stopping ---
    def _should_stop_training(self, current_val_accuracy: Optional[float]) -> Tuple[bool, str]:
        """Checks if training should stop based on validation accuracy trend."""
        if current_val_accuracy is None:
            return False, "No validation accuracy available."

        should_stop = False
        reason = ""
        
        # Store current accuracy
        self.recent_val_accuracies.append(current_val_accuracy)

        # Check only if we have enough history
        if len(self.recent_val_accuracies) == self.recent_val_accuracies.maxlen:
            # Calculate average gain
            # Example: [0.80, 0.81, 0.815] -> Gains: [0.01, 0.005] -> Avg Gain: 0.0075
            gains = []
            acc_list = list(self.recent_val_accuracies)
            for i in range(len(acc_list) - 1):
                gains.append(acc_list[i+1] - acc_list[i])
            
            if gains: # Ensure we have at least one gain calculation
                avg_gain = sum(gains) / len(gains)
                logger.debug(f"[{self.fragment_id}] Recent Accuracies: {acc_list}, Gains: {gains}, Avg Gain: {avg_gain:.5f}")
                if avg_gain < self.min_accuracy_gain:
                    should_stop = True
                    reason = f"Validation accuracy plateaued (Avg gain {avg_gain:.5f} < {self.min_accuracy_gain:.5f})"
                    logger.info(f"[{self.fragment_id}] {reason}. Suggesting stop.")
            else:
                 logger.debug(f"[{self.fragment_id}] Not enough data points for gain calculation yet ({len(acc_list)} accuracies)." )

        # TODO: Implement optional LLM examiner check here if needed
        # if use_llm_as_examiner and not should_stop:
        #    # ... ask professor ...
        #    if professor_says_stop:
        #        should_stop = True
        #        reason = "LLM examiner indicates mastery."

        if not reason and not should_stop:
             reason = f"Accuracy gain sufficient or insufficient history ({len(self.recent_val_accuracies)}/{self.recent_val_accuracies.maxlen})."

        return should_stop, reason
    # --------------------------------

    async def train_on_task(self, 
                            task_name: str, 
                            #epochs: int = 10, # <<< Replaced by max_epochs
                            max_epochs: int = 50, # <<< Max epochs to run
                            learning_rate: float = 0.001,
                            target_accuracy: Optional[float] = None, 
                            validation_split: float = 0.2,
                            context_store: Optional[ContextStore] = None) -> Dict[str, Any]: # <<< Added context_store
        """Loads dataset, splits, trains with validation and dynamic stopping.
           Logs intermediate validation results to ContextStore.
        """
        # <<< Updated log message >>>
        logger.info(f"[{self.fragment_id}] Train request task='{task_name}', max_epochs={max_epochs}, lr={learning_rate}, target_acc={target_accuracy}, val_split={validation_split}")
        results = {
            "fragment_id": self.fragment_id,
            "task_name": task_name,
            "status": "error", 
            "message": "Training did not start",
            "epochs_run": 0,
            "final_train_loss": None,
            "final_val_loss": None,
            "final_val_accuracy": None,
            "target_met": False
        }

        try:
            # --- 1. Load Full Dataset --- 
            loop = asyncio.get_running_loop()
            logger.info(f"[{self.fragment_id}] Loading full dataset for task '{task_name}'...")
            # <<< Corrected async call using functools.partial (Attempt 3 - Removing split) >>>
            # Create partial function binding only positional arguments for build_dataset_from_context
            build_fn = functools.partial(
                build_dataset_from_context, 
                task_name,         # Positional arg 1
                self.num_classes  # Positional arg 2
                # split='full'      # <<< REMOVED - Not an argument of build_dataset_from_context >>>
            )
            # Run the partial function (which now takes no extra args) in the executor
            full_dataset = await loop.run_in_executor(None, build_fn)
            # <<< End corrected call >>>

            if not full_dataset:
                logger.error(f"[{self.fragment_id}] Could not load or build dataset for task '{task_name}'.")
                results["message"] = "Failed to load dataset"
                return results
            
            # Ensure targets are torch.long and scalar
            try:
                processed_dataset = []
                for i, (inp, tar) in enumerate(full_dataset):
                    if not isinstance(tar, torch.Tensor): tar = torch.tensor(tar)
                    if tar.dtype != torch.long: tar = tar.long()
                    if tar.dim() > 0: tar = tar.squeeze() # Ensure scalar target
                    if tar.dim() != 0: raise ValueError(f"Target at index {i} is not scalar after processing: shape {tar.shape}")
                    processed_dataset.append((inp, tar))
                full_dataset = processed_dataset
                logger.info(f"[{self.fragment_id}] Processed {len(full_dataset)} dataset entries (checked target types).")
            except Exception as proc_err:
                 logger.error(f"[{self.fragment_id}] Error processing dataset targets for task '{task_name}': {proc_err}", exc_info=True)
                 results["message"] = f"Dataset target processing error: {proc_err}"
                 return results

            # --- 2. Split Dataset --- 
            if validation_split <= 0 or validation_split >= 1:
                logger.warning(f"[{self.fragment_id}] Invalid validation_split ({validation_split}). Using full dataset for training, validation disabled.")
                train_dataset = full_dataset
                val_dataset = []
                target_accuracy = None # Disable target accuracy if no validation set
            else:
                # Simple shuffle and split
                random.shuffle(full_dataset)
                split_idx = math.ceil(len(full_dataset) * (1 - validation_split))
                train_dataset = full_dataset[:split_idx]
                val_dataset = full_dataset[split_idx:]
                logger.info(f"[{self.fragment_id}] Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation.")
                
                if not val_dataset and target_accuracy is not None:
                    logger.warning(f"[{self.fragment_id}] Validation set is empty after split. Disabling target_accuracy check.")
                    target_accuracy = None

            if not train_dataset:
                logger.error(f"[{self.fragment_id}] Training dataset is empty after split. Aborting.")
                results["message"] = "Training dataset empty after split"
                return results

            # --- 3. Create Dataloaders --- 
            # Adjust batch size as needed
            batch_size = 16 
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

            # --- 4. Setup Training --- 
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            loss_fn = nn.CrossEntropyLoss()
            device = next(self.parameters()).device # Get model device
            logger.info(f"[{self.fragment_id}] Starting training on device: {device}")

            # --- 5. Training Loop --- 
            for epoch in range(max_epochs):
                results["epochs_run"] = epoch + 1
                self.train() # Set model to training mode
                running_train_loss = 0.0
                train_batches = 0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = self(inputs) # Forward pass (logits)
                    loss = loss_fn(outputs, targets)
                    loss.backward() # Backward pass
                    optimizer.step() # Update weights
                    
                    running_train_loss += loss.item()
                    train_batches += 1
                
                avg_train_loss = running_train_loss / train_batches if train_batches > 0 else 0
                results["final_train_loss"] = avg_train_loss
                log_msg = f"Epoch [{epoch+1}/{max_epochs}] Train Loss: {avg_train_loss:.4f}"

                # --- Validation Step --- 
                current_val_accuracy = None
                current_val_loss = None # Initialize val loss
                if val_loader:
                    self.eval() # Set model to evaluation mode
                    running_val_loss = 0.0
                    correct_preds = 0
                    total_preds = 0
                    val_batches = 0
                    with torch.no_grad():
                        for val_inputs, val_targets in val_loader:
                            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                            val_outputs = self(val_inputs)
                            val_loss = loss_fn(val_outputs, val_targets)
                            running_val_loss += val_loss.item()
                            
                            _, predicted_indices = torch.max(val_outputs, 1)
                            total_preds += val_targets.size(0)
                            correct_preds += (predicted_indices == val_targets).sum().item()
                            val_batches += 1
                    
                    avg_val_loss = running_val_loss / val_batches if val_batches > 0 else 0
                    current_val_accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
                    current_val_loss = avg_val_loss # Assign to variable
                    results["final_val_loss"] = current_val_loss
                    results["final_val_accuracy"] = current_val_accuracy
                    log_msg += f" | Val Loss: {current_val_loss:.4f}, Val Acc: {current_val_accuracy:.4f}"

                    # <<< Log intermediate validation results to ContextStore >>>
                    if context_store:
                        try:
                            log_key = f"training_log:{self.fragment_id}:{task_name}:epoch_{epoch+1}"
                            log_data = {
                                "epoch": epoch + 1,
                                "validation_accuracy": current_val_accuracy,
                                "validation_loss": current_val_loss,
                                "train_loss": avg_train_loss,
                                "timestamp": time.time() # Use time directly
                            }
                            await context_store.set(log_key, log_data)
                            logger.debug(f"[{self.fragment_id}] Logged intermediate results to ContextStore key '{log_key}'")
                        except Exception as cs_log_err:
                            logger.warning(f"[{self.fragment_id}] Failed to log intermediate results to ContextStore: {cs_log_err}", exc_info=True)
                    # <<< End ContextStore logging >>>

                logger.info(f"[{self.fragment_id}] {log_msg}")
                print(f"[{self.fragment_id}] {log_msg}") # Print for visibility

                # --- Combined Stopping Checks --- 
                stop_reason = ""
                # 1. Check Target Accuracy (Explicit Goal)
                if target_accuracy is not None and current_val_accuracy is not None:
                    if current_val_accuracy >= target_accuracy:
                        stop_reason = f"Target accuracy ({target_accuracy:.4f}) reached at epoch {epoch+1}."
                        results["target_met"] = True
                
                # 2. Check Dynamic Stopping (Plateau/Diminishing Returns)
                if not stop_reason:
                     should_stop_dynamic, dynamic_reason = self._should_stop_training(current_val_accuracy)
                     if should_stop_dynamic:
                         stop_reason = dynamic_reason # Use the reason from the helper

                # 3. Break loop if any stop reason is found
                if stop_reason:
                    logger.info(f"[{self.fragment_id}] Stopping training: {stop_reason}")
                    results["status"] = "success"
                    results["message"] = f"Training stopped early at epoch {epoch+1}: {stop_reason}" # Use the specific reason
                    # Ensure final metrics are set before returning
                    results["final_train_loss"] = avg_train_loss 
                    results["final_val_loss"] = current_val_loss
                    results["final_val_accuracy"] = current_val_accuracy
                    break # Exit the training loop
            # --- End Training Loop ---
            
            # <<< Update final status message if loop finished normally >>>
            if results["status"] != "success": # If loop finished without early stopping
                results["status"] = "success"
                results["message"] = f"Training finished after reaching max_epochs ({max_epochs})."
                if target_accuracy is not None and not results.get("target_met", False):
                     # Check if target accuracy was met on the *last* epoch if loop completed
                     if current_val_accuracy is not None and current_val_accuracy >= target_accuracy:
                         results["target_met"] = True
                         results["message"] += f" Target accuracy ({target_accuracy:.4f}) was met on the final epoch."
                     else:
                         results["message"] += f" Target accuracy ({target_accuracy:.4f}) was not met."
                # Ensure final metrics are stored even if loop completes normally
                # results["final_train_loss"] is already updated each epoch
                results["final_val_loss"] = current_val_loss
                results["final_val_accuracy"] = current_val_accuracy

            return results

        except Exception as e:
            logger.error(f"[{self.fragment_id}] Error during train_on_task for '{task_name}': {e}", exc_info=True)
            results["status"] = "error"
            results["message"] = f"Training loop failed: {e}"
            return results

    # Remove or deprecate the old train_on method as logic is moved to train_on_task
    # def train_on(self, ...):
    #    pass 

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