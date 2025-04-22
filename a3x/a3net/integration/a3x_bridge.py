from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import re # Import regex for cleaning goal string
import logging # Added logging
from pathlib import Path # <<< Add Path >>>
import os
import shlex # For parsing command-line like strings
import ast   # For safely evaluating literals
import time # <<< Add time import for timestamp >>>
import asyncio # Added for async handling
import json # Import json for loading dataset
import functools # Added for partial function application
import uuid
import datetime as dt # <<< Alias datetime import

# Corrected A³Net imports (relative within a3x)
from ..trainer.dataset_builder import build_dataset_from_context, get_embedding_model
from ..core.fragment_cell import FragmentCell
from ..trainer.train_loop import train_fragment_cell # <<< Correct function name
from ..core.memory_bank import MemoryBank # <<< Import MemoryBank >>>
from ..core.cognitive_graph import CognitiveGraph # <<< Import CognitiveGraph >>>
from ..core.neural_language_fragment import NeuralLanguageFragment # <<< Import NeuralLanguageFragment >>>
from ..core.reflective_language_fragment import ReflectiveLanguageFragment # <<< Import ReflectiveLanguageFragment >>>
from ..core.professor_llm_fragment import ProfessorLLMFragment # <<< Import ProfessorLLMFragment >>>
from ..core.context_store import ContextStore # <<< Ensure ContextStore is imported >>>

# <<< REMOVED ALL HANDLER IMPORTS >>>
# --- Import new handlers ---
# from .bridge_handlers.handle_train import handle_train_fragment
# from .bridge_handlers.handle_create import handle_create_fragment
# ... (all other handlers removed)
# --- End Import new handlers ---

# Configure logging
logger = logging.getLogger(__name__)

# --- Module-Level State for Last Ask (If still needed by any remaining logic here) ---
LAST_ASK_RESULT: Optional[Dict[str, Any]] = None

# --- Example Registration Function (If needed by remaining logic) ---
async def registrar_exemplo_de_aprendizado(context_store: ContextStore, task_name: str, input_data: str, label_text: str):
    # ... (existing function, seems okay or replaced by data_logger) ...
    pass
# ---------------------------------

# <<< handle_directive function has been MOVED to run.py >>>