# a3x/core/types.py
""" Defines common data structures and types used across the A3X core components. """

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class ActionIntent:
    """ Represents a structured request from a fragment to execute a symbolic skill/action. """
    skill_target: str                  # The registered name of the skill/action to execute (e.g., "mutar_fragmento")
    parameters: Dict[str, Any]         # Dictionary containing the necessary parameters for the skill
    reasoning: Optional[str] = None    # Optional: Natural language justification from the fragment
    requested_by: Optional[str] = None # Optional: ID of the fragment making the request (for logging/debugging)

    def __post_init__(self):
        # Basic validation
        if not self.skill_target or not isinstance(self.skill_target, str):
            raise ValueError("ActionIntent requires a non-empty string for skill_target.")
        if not isinstance(self.parameters, dict):
            raise ValueError("ActionIntent requires a dictionary for parameters.")

@dataclass
class PendingRequest:
    """ Represents a request from a fragment for help from another fragment with a specific capability. """
    capability_needed: str             # Description of the required capability (e.g., "data_analysis", "knowledge_lookup")
    details: Optional[str] = None      # Optional: Natural language description of the specific need
    data_reference: Optional[Any] = None # Optional: Reference to relevant data within the SharedTaskContext (e.g., a key in task_data)
    requested_by: Optional[str] = None # Optional: ID of the fragment making the request

    def __post_init__(self):
        if not self.capability_needed or not isinstance(self.capability_needed, str):
            raise ValueError("PendingRequest requires a non-empty string for capability_needed.")

# We can add other common types here later, e.g., for WorkingMemoryEntry if needed separately
# from SharedTaskContext internal_chat_queue structure. 