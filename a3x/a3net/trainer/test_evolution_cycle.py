import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import logging
import json
from datetime import datetime
import asyncio

from ..core.dummy_fragment import DummyFragmentCell
from ..core.context_store import SQLiteContextStore
from ..trainer.dataset_builder import create_or_update_dataset_jsonl
from ..trainer.train_loop import train_fragment_cell, EarlyStopping
from ..skills.train_from_latest_context import train_from_latest_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockEvaluator:
    """Mock evaluator for testing."""
    def evaluate(self, fragment: nn.Module, inputs: torch.Tensor) -> float:
        # Simulate evaluation score
        return np.random.random()

class MockCompetitor:
    """Mock competitor for testing."""
    def compete(self, fragment1: nn.Module, fragment2: nn.Module) -> bool:
        # Simulate competition result
        return np.random.random() > 0.5

async def generate_evolution_context(
    context_store: SQLiteContextStore,
    num_episodes: int = 5
):
    """Generate mock evolution context for testing."""
    for i in range(num_episodes):
        # Simulate a failure episode
        episode = {
            "type": "failure",
            "fragment_id": f"fragment_{i}",
            "error": f"Test error {i}",
            "context": f"Test context {i}",
            "timestamp": datetime.now().isoformat(),
            "text": f"Example text {i} with some random content",
            "label": np.random.choice(["SIM", "NÃO", "REAVALIAR"])
        }
        await context_store.set(f"episode_{i}", episode, tags=[f"fragment_{i}"])

async def test_evolution_cycle():
    """Test the complete neural-symbolic evolution cycle."""
    try:
        # 1. Initialize context store
        context_store = SQLiteContextStore()
        await context_store.initialize()
        logger.info("Initialized context store")
        
        # 2. Generate evolution context
        await generate_evolution_context(context_store)
        logger.info("Generated evolution context")
        
        # 3. Create evaluator and competitor
        evaluator = MockEvaluator()
        competitor = MockCompetitor()
        logger.info("Created evaluator and competitor")
        
        # 4. Test training from context
        fragment_id = "fragment_0"  # Use one of the generated fragments
        training_result = await train_from_latest_context(
            context_store=context_store,
            fragment_id=fragment_id
        )
        
        if not training_result["success"]:
            raise ValueError(f"Training failed: {training_result['error']}")
            
        logger.info(f"Training completed: {training_result}")
        
        # 5. Load trained fragment
        model_path = Path(training_result["model_path"])
        fragment = joblib.load(model_path)
        logger.info("Loaded trained fragment")
        
        # 6. Test evaluation
        test_input = torch.randn(384)  # Random input
        score = evaluator.evaluate(fragment, test_input)
        logger.info(f"Evaluation score: {score}")
        
        # 7. Test competition
        competitor_fragment = DummyFragmentCell()
        winner = competitor.compete(fragment, competitor_fragment)
        logger.info(f"Competition result: {'Original fragment won' if winner else 'Competitor won'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False
    finally:
        # Clean up
        if 'context_store' in locals():
            await context_store.close()

if __name__ == "__main__":
    success = asyncio.run(test_evolution_cycle())
    if success:
        logger.info("Evolution cycle test completed successfully!")
    else:
        logger.error("Evolution cycle test failed!") 