import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from a3x.core.context import Context
from a3x.core.skills import skill

logger = logging.getLogger(__name__)

# Define the directory and filename for the dataset
# Assuming PROJECT_ROOT is accessible or defined globally, otherwise adjust path logic
# Placeholder for project root - this might need adjustment depending on where config is loaded
try:
    from a3x.core.config import PROJECT_ROOT
except ImportError:
    logger.warning("Could not import PROJECT_ROOT from config, using relative path logic.")
    # Define a fallback relative path if needed, e.g., based on this file's location
    # This is just an example, adjust as necessary
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "a3net"
DATASET_PATH = DATASET_DIR / "failure_training.jsonl"

@skill(
    name="generate_training_data_from_failure",
    description="Gera um dataset de exemplos positivos e negativos com base nos logs de falhas simbólicas para treinar o A³Net.",
    parameters={
        "failure_logs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "string", "description": "The symbolic command or action attempted."}, 
                    "error": {"type": "string", "description": "The error message encountered."}, 
                    "context": {"type": "object", "description": "Relevant context at the time of failure."}
                },
                "required": ["step", "error", "context"]
            },
            "description": "An array of symbolic failure log entries."
        },
        "objective": {"type": "string", "description": "The high-level objective the agent was trying to achieve."}, 
        "max_examples": {"type": "integer", "default": 100, "description": "Maximum number of training examples to generate."}
    }
)
async def generate_training_data_from_failure(
    context: Context,
    failure_logs: List[Dict[str, Any]],
    objective: str,
    max_examples: int = 100
) -> Dict[str, Any]:
    """
    Generates a training dataset from symbolic failure logs.

    Iterates through failure logs, extracts relevant information,
    formats it into training examples (positive/negative),
    and saves them to a .jsonl file.

    Args:
        context: The execution context.
        failure_logs: A list of dictionaries, each representing a failure event.
        objective: The overall goal during which the failures occurred.
        max_examples: The maximum number of examples to include in the dataset.

    Returns:
        A dictionary indicating success or failure, the number of examples generated,
        and the path to the dataset file.
    """
    logger.info(f"Generating training data from {len(failure_logs)} failure logs for objective: '{objective}' (max_examples: {max_examples}).")
    
    generated_examples = []

    try:
        # 1. Iterate over failure_logs and extract relevant data
        for i, log_entry in enumerate(failure_logs):
            if len(generated_examples) >= max_examples:
                logger.info(f"Reached max_examples limit ({max_examples}). Stopping data generation.")
                break
                
            step = log_entry.get("step")
            error = log_entry.get("error")
            log_context = log_entry.get("context", {})

            if not step or not error:
                logger.warning(f"Skipping log entry {i} due to missing 'step' or 'error'.")
                continue

            # 2. Construct training examples (input/label format TBD)
            # Placeholder: Simple format - refine based on A³Net requirements
            # Example: Create a negative example for the failed step
            training_example = {
                "input": {
                    "objective": objective,
                    "command": step, 
                    "context": log_context, # Include relevant context
                    # Potentially add previous steps/history if needed
                },
                "label": "failure", # Label this specific step as failure
                "error_details": error # Include error for analysis/specific training
            }
            generated_examples.append(training_example)

            # Placeholder: Logic to potentially generate positive examples 
            # (e.g., if a correction was applied later, or based on successful similar steps)
            # This part needs more definition based on how positive examples are derived.

        logger.info(f"Constructed {len(generated_examples)} training examples.")

        # 3. Ensure dataset directory exists
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

        # 4. Write the dataset to the .jsonl file
        written_count = 0
        with open(DATASET_PATH, 'w', encoding='utf-8') as f:
            for example in generated_examples:
                try:
                    f.write(json.dumps(example) + '\n')
                    written_count += 1
                except Exception as json_err:
                    logger.error(f"Failed to serialize or write example: {example}. Error: {json_err}")
                    # Decide whether to skip or abort
                    continue 

        logger.info(f"Successfully wrote {written_count} examples to {DATASET_PATH}")

        # 5. Return success
        return {
            "status": "success",
            "generated": written_count,
            "path": str(DATASET_PATH)
        }

    except Exception as e:
        logger.exception(f"Error generating training data: {e}")
        return {
            "status": "error",
            "message": f"Failed to generate training data: {e}"
        } 