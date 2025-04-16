from typing import List, Optional, Dict, Any

class AgentState:
    """Singleton class to hold the live state of the A³X agent."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentState, cls).__new__(cls)
            # Initialize state variables here
            cls._instance.current_task: Optional[str] = None
            cls._instance.current_plan: Optional[List[str]] = None
            cls._instance.active_fragment_name: Optional[str] = None
            cls._instance.last_step_info: Optional[Dict[str, Any]] = None # e.g., {thought: ..., action: ..., input: ..., observation: ...}
            cls._instance.recent_skills_executed: List[str] = [] # Could be limited in size
            cls._instance.last_heuristic_used: Optional[str] = None
            cls._instance.agent_status: str = "idle" # e.g., idle, planning, thinking, executing, error, finished
            cls._instance.last_error: Optional[str] = None
        return cls._instance

    def reset(self):
        """Resets the state, typically called when a new task starts."""
        self.current_task = None
        self.current_plan = None
        self.active_fragment_name = None
        self.last_step_info = None
        self.recent_skills_executed = []
        self.last_heuristic_used = None
        self.agent_status = "idle"
        self.last_error = None

    def update_step(self, step_info: Dict[str, Any]):
        """Updates state after a step (thought, action, observation)."""
        self.last_step_info = step_info
        if step_info.get("action"):
            self.recent_skills_executed.append(step_info["action"])
            # Optional: limit the size of recent_skills_executed
            if len(self.recent_skills_executed) > 20: # Keep last 20 skills
                self.recent_skills_executed.pop(0)

    # Add other specific update methods as needed, e.g.:
    def set_status(self, status: str):
        self.agent_status = status

    def set_active_fragment(self, name: Optional[str]):
        self.active_fragment_name = name

    def set_current_task(self, task: str):
        self.current_task = task
        self.reset() # Reset other fields when a new task starts
        self.set_status("planning") # Initial status for a new task

    def set_error(self, error_message: str):
        self.last_error = error_message
        self.set_status("error")

# The actual singleton instance used throughout the application
AGENT_STATE = AgentState() 