import torch
from typing import Dict, Any
import logging

# Import the parent class
from .neural_language_fragment import NeuralLanguageFragment

logger = logging.getLogger(__name__)

class ReflectiveLanguageFragment(NeuralLanguageFragment):
    """Extends NeuralLanguageFragment to provide explanations for its predictions."""

    def __init__(self, *args, **kwargs):
        """Initializes the ReflectiveLanguageFragment using the parent class initializer."""
        super().__init__(*args, **kwargs)
        print(f"[ReflectLangFrag '{self.fragment_id}'] Initialized (Reflective).")
        # No additional initialization needed specifically for reflection yet

    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """Predicts the most likely class label, confidence, and provides an explanation.
        
        Args:
            x: The input tensor.
            
        Returns:
            A dictionary containing 'output', 'confidence', and 'explanation'.
        """
        # 1. Get the base prediction (output and confidence) from the parent class
        base_prediction = super().predict(x)
        predicted_label = base_prediction['output']
        confidence = base_prediction['confidence']

        # 2. Generate the explanation based on the confidence and potentially other factors
        confidence_percent = confidence * 100
        explanation = f"Respondi {predicted_label} porque essa classe teve {confidence_percent:.1f}% de confiança."

        # --- Optional: Add more detail based on other class probabilities --- 
        # This requires getting the full probability distribution again, slightly inefficient
        # but necessary if we didn't store it from the super().predict call.
        # If performance critical, optimize NeuralLanguageFragment.predict to optionally return probabilities.
        with torch.no_grad():
             logits = self.forward(x)
             probabilities = torch.softmax(logits, dim=-1)
             
             if self.num_classes > 1 and probabilities.shape[-1] > 1:
                # Find the index of the winning class to exclude it
                predicted_index = -1
                for idx, label_name in self.id_to_label.items():
                    if label_name == predicted_label:
                         predicted_index = idx
                         break
                         
                if predicted_index != -1:
                    second_probs = probabilities.clone()
                    second_probs[..., predicted_index] = -float('inf') 
                    second_max_prob, second_index = torch.max(second_probs, dim=-1)
                    second_label = self.id_to_label.get(second_index.item(), "UNKNOWN_CLASS")
                    second_conf_pct = second_max_prob.item() * 100
                    if second_conf_pct > 1.0: 
                        explanation += f" A segunda classe mais provável ({second_label}) teve {second_conf_pct:.1f}%."
                    else:
                         explanation += f" As outras classes tiveram confiança muito baixa."
                else:
                     # This case should ideally not happen if id_to_label is consistent
                     explanation += f" (Não foi possível determinar outras probabilidades.)"
             elif self.num_classes == 1:
                  explanation += f" É a única classe possível."
        # --- End Optional Detail --- 

        logger.info(f"Prediction for fragment '{self.fragment_id}': Label='{predicted_label}', Confidence={confidence_percent:.1f}%")

        # 3. Combine results into the final dictionary
        final_result = base_prediction # Start with {'output': ..., 'confidence': ...}
        final_result['explanation'] = explanation
        
        return final_result

# Note: The train_on method is inherited directly from NeuralLanguageFragment
# and doesn't need to be overridden unless specific reflective training is needed. 