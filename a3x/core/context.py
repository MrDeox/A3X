# a3x/core/context.py
import logging
from pathlib import Path
from typing import Optional, Dict, Any

class Context:
    """
    Context object passed to skills, providing access to shared resources.
    """
    def __init__(self,
                 logger: Optional[logging.Logger] = None,
                 workspace_root: Optional[Path] = None,
                 mem: Optional[Dict[str, Any]] = None, # Simple memory store
                 llm_url: Optional[str] = None,
                 tools: Optional[Dict[str, Any]] = None, # Available tools
                 # Add other necessary attributes as identified
                 ):
        self.logger = logger or logging.getLogger(__name__)
        self.workspace_root = workspace_root or Path('.')
        self.mem = mem if mem is not None else {}
        self.llm_url = llm_url
        self.tools = tools if tools is not None else {}
        # Potentially add execute_tool method reference if needed by skills

    # Add methods if Context needs specific functionality 