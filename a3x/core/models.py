from typing import TypedDict, Dict, Any, Literal

class PlanStep(TypedDict):
    """Represents a single, structured step in an execution plan."""
    step_id: int
    description: str
    action_type: Literal['skill', 'fragment']
    target_name: str
    arguments: Dict[str, Any]
    # Add optional fields later if needed, e.g., depends_on: List[int] 