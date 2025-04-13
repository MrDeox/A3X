import logging
from typing import Any, Dict, Optional, List
import time
import asyncio

class FragmentState:
    """Holds the dynamic state and metrics for a Fragment instance."""
    def __init__(self, name: str, skills: list[str], prompt_template: str):
        self.name = name
        self.base_skills = skills # Store the initial skills
        self.current_skills = skills[:] # Active skills, might be restricted
        self.base_prompt_template = prompt_template # Store the original prompt
        self.current_prompt = prompt_template # Active prompt, might be optimized
        # Overall fragment performance metrics
        self.metrics = {"success_count": 0, "failure_count": 0, "total_time": 0.0, "tasks_processed": 0}
        self.performance_history = [] # Track overall performance over time
        # Per-skill performance metrics
        self.skill_metrics: Dict[str, Dict[str, Any]] = {
            # Example: "read_file": {"success": 5, "failure": 1, "total_time": 1.23}
        }

    def update_metrics(self, success: bool, execution_time: float):
        """Updates the overall fragment performance metrics after a task execution."""
        self.metrics["tasks_processed"] += 1
        if success:
            self.metrics["success_count"] += 1
        else:
            self.metrics["failure_count"] += 1
        self.metrics["total_time"] += execution_time
        self.performance_history.append({
            "timestamp": time.time(),
            "success_rate": self.get_success_rate(),
            "avg_time": self.get_avg_time()
        })
        if len(self.performance_history) > 50:
            self.performance_history.pop(0)

    def update_skill_metrics(self, skill_name: str, metrics: Dict[str, Any]):
        """Updates the metrics for a specific skill execution."""
        if skill_name not in self.skill_metrics:
            self.skill_metrics[skill_name] = {"success": 0, "failure": 0, "total_time": 0.0, "calls": 0}

        self.skill_metrics[skill_name]["calls"] += 1
        if metrics.get("status") == "success":
            self.skill_metrics[skill_name]["success"] += 1
        else:
            self.skill_metrics[skill_name]["failure"] += 1
        self.skill_metrics[skill_name]["total_time"] += metrics.get("duration_seconds", 0.0)

    def get_success_rate(self) -> float:
        """Calculates the overall success rate of the fragment."""
        return self.metrics["success_count"] / self.metrics["tasks_processed"]

    def get_avg_time(self) -> float:
        """Calculates the overall average execution time of the fragment."""
        return self.metrics["total_time"] / self.metrics["tasks_processed"]

    def get_status_summary(self) -> str:
        """Generates a summary string of the fragment's overall status."""
        rate = self.get_success_rate()
        avg_time = self.get_avg_time()
        return f"Fragment '{self.name}': {self.metrics['tasks_processed']} tasks, {rate*100:.1f}% success, {avg_time:.2f}s avg time."

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FragmentOptimizer:
    def __init__(self, fragment_state: FragmentState, config: Optional[Dict] = None):
        self.fragment_state = fragment_state
        # Default thresholds, can be overridden by config
        self.config = {
            "low_success_threshold": 0.5,
            "skill_removal_threshold": 0.3, # Remove skill if success rate below this
            "min_skill_calls_for_removal": 5, # Don't remove skill based on too few calls
            ** (config or {})
        }
        logger.info(f"Optimizer initialized for Fragment '{self.fragment_state.name}' with config: {self.config}")

    async def optimize_if_needed(self) -> bool:
        """Checks performance and triggers optimization steps if thresholds are met."""
        if self.fragment_state.metrics["tasks_processed"] < self.config["min_tasks_for_optimization"]:
            logger.debug(f"Skipping optimization for '{self.fragment_state.name}', not enough tasks processed ({self.fragment_state.metrics['tasks_processed']}/{self.config['min_tasks_for_optimization']}).")
            return False

        success_rate = self.fragment_state.get_success_rate()
        optimized = False

        if success_rate < self.config["low_success_threshold"]:
            logger.warning(f"Low success rate ({success_rate:.2f} < {self.config['low_success_threshold']}) detected for Fragment '{self.fragment_state.name}'. Initiating optimization.")

            # --- Optimization Steps ---
            prompt_updated = await self._rewrite_prompt_based_on_failures()
            if prompt_updated:
                optimized = True
                logger.info(f"Prompt updated for Fragment '{self.fragment_state.name}'.")

            skills_restricted = await self._restrict_underperforming_skills()
            if skills_restricted:
               optimized = True
               logger.info(f"Skills restricted for Fragment '{self.fragment_state.name}'. New list: {self.fragment_state.current_skills}")

        else:
             logger.info(f"Fragment '{self.fragment_state.name}' performance is acceptable ({success_rate:.2f} >= {self.config['low_success_threshold']}). No optimization needed.")

        return optimized

    async def _rewrite_prompt_based_on_failures(self) -> bool:
        """
        (Placeholder) Analyzes recent failures and uses an LLM to rewrite the fragment's prompt.
        Returns True if the prompt was updated.
        """
        logger.info(f"Attempting to rewrite prompt for Fragment '{self.fragment_state.name}'...")
        recent_failures = ["Failed due to ambiguous instruction X", "Tool Y failed unexpectedly"] # Dummy data

        if not recent_failures:
             logger.warning("No recent failure data found to guide prompt rewriting.")
             return False

        meta_prompt = f"""
        The cognitive fragment '{self.fragment_state.name}' is underperforming. Its current prompt is:
        --- CURRENT PROMPT ---
        {self.fragment_state.current_prompt}
        --- END CURRENT PROMPT ---

        Recent failures indicate these issues:
        - {chr(10).join(f'- {fail}' for fail in recent_failures)}

        Rewrite the prompt to be clearer, more robust, and specifically address these failure patterns.
        Focus on {self.fragment_state.name}'s core function. Ensure the output is ONLY the new prompt text.
        """

        await asyncio.sleep(0.5)
        new_prompt_text = self.fragment_state.current_prompt + "\n\n# Auto-Correction Attempt: Added more specific examples based on recent failures."

        if new_prompt_text and new_prompt_text != self.fragment_state.current_prompt:
            self.fragment_state.current_prompt = new_prompt_text
            logger.info("Prompt successfully rewritten.")
            return True
        else:
            logger.warning("LLM failed to generate a new prompt or returned the same prompt.")
            return False

    async def _restrict_underperforming_skills(self) -> bool:
        """
        Analyzes per-skill performance metrics and removes consistently failing ones
        from the fragment's *current* skill list.
        Returns True if skills were restricted.
        """
        logger.info(f"Attempting to restrict skills for Fragment '{self.fragment_state.name}'...")
        skill_performance = self.fragment_state.skill_metrics

        if not skill_performance:
            logger.info("No per-skill metrics available yet. Skipping skill restriction.")
            return False

        initial_skill_count = len(self.fragment_state.current_skills)
        skills_to_remove = set()
        min_calls = self.config["min_skill_calls_for_removal"]

        logger.debug(f"Analyzing skill metrics: {skill_performance}")
        for skill_name, metrics in skill_performance.items():
            if skill_name not in self.fragment_state.current_skills:
                continue # Skip skills already removed or not part of this fragment initially

            calls = metrics.get("calls", 0)
            failures = metrics.get("failure", 0)

            if calls < min_calls:
                logger.debug(f"Skill '{skill_name}' has only {calls}/{min_calls} calls, skipping removal check.")
                continue

            failure_rate = failures / calls if calls > 0 else 0
            success_rate = 1.0 - failure_rate

            logger.debug(f"Skill '{skill_name}': Calls={calls}, Failures={failures}, Success Rate={success_rate:.2f}")

            if success_rate < self.config["skill_removal_threshold"]:
                 # Check if removing this skill would leave the fragment with no skills
                 if len(self.fragment_state.current_skills) - len(skills_to_remove) > 1:
                     logger.warning(f"Marking skill '{skill_name}' for removal from '{self.fragment_state.name}' due to low success rate ({success_rate:.2f} < {self.config['skill_removal_threshold']}).")
                     skills_to_remove.add(skill_name)
                 else:
                      logger.warning(f"Skipping removal of skill '{skill_name}' from '{self.fragment_state.name}' to avoid leaving it with no skills.")

        if skills_to_remove:
            # Update the *current* skills list in the fragment state
            self.fragment_state.current_skills = [s for s in self.fragment_state.current_skills if s not in skills_to_remove]
            logger.info(f"Skills restricted for '{self.fragment_state.name}'. New list: {self.fragment_state.current_skills}")
            return True
        else:
            logger.info(f"No skills met the criteria for removal from '{self.fragment_state.name}'.")
            return False

# ... (rest of the file, e.g., example usage) 