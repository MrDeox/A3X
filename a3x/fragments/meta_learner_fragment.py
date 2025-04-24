import logging
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Counter as TypingCounter # For type hint
from collections import Counter, defaultdict # Adicionado defaultdict
from a3x.fragments.base import BaseFragment, FragmentContext
from a3x.fragments.registry import fragment
from a3x.core.memory.memory_manager import MemoryManager

# Core A3X Imports (adjust based on actual project structure)
try:
    from a3x.core.memory.memory_manager import MemoryManager     # To access episodic memory
    from a3x.fragments.registry import FragmentRegistry # Assuming registry is needed for validation
except ImportError as e:
    print(f"[MetaLearnerFragment] Warning: Could not import core A3X components ({e}). Using placeholders.")
    # Define placeholders if imports fail
    class FragmentContext:
        memory: Optional['MemoryManager'] = None
        workspace_root: Optional[str] = None
    class BaseFragment:
        def __init__(self, *args, **kwargs): pass
    class MemoryManager: # Placeholder with expected method signature
        async def get_recent_episodes(self, limit: int) -> List[Dict[str, Any]]:
            print(f"Placeholder MemoryManager.get_recent_episodes called with limit={limit}")
            # Simulate data
            return [
                {'id': 1, 'context': 'Task Alpha', 'action': 'CodeWriter', 'outcome': 'success', 'timestamp': '2023-10-28T10:00:00', 'metadata': '{"fragment_name": "CodeWriter", "confidence": 0.9, "a3l_command": "criar arquivo X"}'},
                {'id': 2, 'context': 'Task Beta', 'action': 'DataAnalyzer', 'outcome': 'failure', 'timestamp': '2023-10-28T10:01:00', 'metadata': '{"fragment_name": "DataAnalyzer", "error": "Timeout", "a3l_command": "analisar dados Y"}'},
                {'id': 3, 'context': 'Task Alpha', 'action': 'CodeReviewer', 'outcome': 'success', 'timestamp': '2023-10-28T10:02:00', 'metadata': '{"fragment_name": "CodeReviewer", "heuristic_applied": "check_imports"}'},
                {'id': 4, 'context': 'Task Gamma', 'action': 'CodeWriter', 'outcome': 'success', 'timestamp': '2023-10-28T10:03:00', 'metadata': '{"fragment_name": "CodeWriter", "confidence": 0.8}'},
                {'id': 5, 'context': 'Task Beta', 'action': 'Debugger', 'outcome': 'success', 'timestamp': '2023-10-28T10:04:00', 'metadata': '{"fragment_name": "Debugger", "thought": "Found root cause"}'},
            ] * (limit // 5)

logger = logging.getLogger(__name__)

# Constants for analysis
MIN_EPISODES_FOR_ANALYSIS = 10
LOW_PERFORMANCE_THRESHOLD = 0.5 # Success rate below 50%
HIGH_ERROR_RATE_THRESHOLD = 0.3 # More than 30% errors

@fragment(
    name="meta_learner",
    description="Analyzes system learning and proposes improvements",
    category="learning",
    skills=["analyze_learning", "propose_improvements", "validate_changes"]
)
class MetaLearnerFragment(BaseFragment):
    def __init__(self, ctx: FragmentContext):
        super().__init__(ctx)
        # Ensure MemoryManager is properly initialized (assuming it's provided via ctx or instantiated)
        # This line might be redundant if BaseFragment or ctx handles it.
        # Let's assume ctx provides access to the shared MemoryManager instance.
        if hasattr(ctx, 'memory_manager') and isinstance(ctx.memory_manager, MemoryManager):
            self.memory_manager = ctx.memory_manager
        else:
            # Fallback or error, depending on design
            self._logger.warning("MemoryManager not found in context, creating a new instance (might be incorrect).")
            self.memory_manager = MemoryManager() # Or raise error

        # Assuming FragmentRegistry might be needed later for validation
        if hasattr(ctx, 'fragment_registry') and isinstance(ctx.fragment_registry, FragmentRegistry):
            self.fragment_registry = ctx.fragment_registry
        else:
            self._logger.warning("FragmentRegistry not found in context.")
            self.fragment_registry = None

    async def execute(self, num_episodes_to_analyze: int = 100, **kwargs) -> Dict[str, Any]: # Added parameter
        try:
            self._logger.info(f"Executing MetaLearner analysis on last {num_episodes_to_analyze} episodes.")
            # Analyze learning patterns
            learning_analysis = await self._analyze_learning(num_episodes_to_analyze)
            
            # Propose improvements
            improvements = await self._propose_improvements(learning_analysis)
            
            # Validate proposed changes
            validation = await self._validate_changes(improvements)
            
            self._logger.info("MetaLearner execution finished.")
            return {
                "status": "success", # Added status
                "learning_analysis": learning_analysis,
                "improvements": improvements,
                "validation": validation
            }
            
        except Exception as e:
            self._logger.error(f"Error in MetaLearnerFragment: {str(e)}", exc_info=True) # Log traceback
            # raise # Re-raising might hide the error from caller if not handled
            return {"status": "error", "message": str(e)} # Return error status

    async def _analyze_learning(self, num_episodes: int) -> Dict[str, Any]:
        """Fetches recent episodes and analyzes performance patterns."""
        self._logger.info(f"Analyzing learning from {num_episodes} episodes...")
        episodes = await self.memory_manager.get_recent_episodes(limit=num_episodes)
        
        if not episodes:
            self._logger.warning("No episodes found for analysis.")
            return {"message": "No episodes found.", "fragment_stats": {}, "errors": {}}

        fragment_stats = defaultdict(lambda: {"success": 0, "failure": 0, "total": 0, "errors": Counter()})
        
        for episode in episodes:
            try:
                # Assuming metadata contains fragment_name
                metadata_str = episode.get('metadata', '{}')
                if isinstance(metadata_str, str):
                    metadata = json.loads(metadata_str) 
                elif isinstance(metadata_str, dict):
                    metadata = metadata_str # Already a dict
                else:
                    metadata = {}

                fragment_name = metadata.get("fragment_name")
                if not fragment_name:
                    # Try getting from 'action' if metadata is missing
                    action = episode.get('action')
                    if action and isinstance(action, str): # Basic check
                        fragment_name = action # Use action as a fallback
                    else:
                        continue # Skip episode if no fragment identified

                stats = fragment_stats[fragment_name]
                stats["total"] += 1
                outcome = episode.get('outcome', 'unknown').lower()
                
                if outcome == 'success':
                    stats["success"] += 1
                elif outcome == 'failure':
                    stats["failure"] += 1
                    error_detail = metadata.get("error", "Unknown error")
                    stats["errors"][str(error_detail)] += 1 # Ensure error detail is string
                # Ignore 'unknown' outcomes for now
            except json.JSONDecodeError:
                self._logger.warning(f"Could not parse metadata for episode ID {episode.get('id')}: {metadata_str}")
            except Exception as e:
                self._logger.error(f"Error processing episode ID {episode.get('id')}: {e}", exc_info=True)
        
        # Calculate rates
        analysis_results = {"fragment_performance": {}, "common_errors": {}}
        for name, stats in fragment_stats.items():
            if stats["total"] >= MIN_EPISODES_FOR_ANALYSIS:
                success_rate = stats["success"] / stats["total"]
                error_rate = stats["failure"] / stats["total"]
                analysis_results["fragment_performance"][name] = {
                    "success_rate": round(success_rate, 3),
                    "error_rate": round(error_rate, 3),
                    "total_executions": stats["total"],
                    "top_errors": stats["errors"].most_common(3)
                }
                if stats["errors"]:
                    analysis_results["common_errors"][name] = stats["errors"].most_common(3)
            else:
                self._logger.info(f"Skipping analysis for fragment '{name}', not enough executions ({stats['total']}/{MIN_EPISODES_FOR_ANALYSIS}).")

        self._logger.info(f"Learning analysis complete. Found stats for {len(analysis_results['fragment_performance'])} fragments.")
        return analysis_results

    async def _propose_improvements(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Proposes improvements based on the learning analysis."""
        self._logger.info("Generating improvement proposals...")
        proposals = []
        fragment_performance = analysis.get("fragment_performance", {})

        for name, perf in fragment_performance.items():
            # Proposal 1: Mutate low-performing fragments
            if perf["success_rate"] < LOW_PERFORMANCE_THRESHOLD:
                proposals.append({
                    "type": "mutation",
                    "target_fragment": name,
                    "strategy": "improve_performance", # Suggest a strategy
                    "reason": f"Low success rate ({perf['success_rate']:.1%})"
                })
                self._logger.info(f"Proposing mutation for low-performing fragment: {name}")

            # Proposal 2: Correct fragments with high error rates and specific common errors
            elif perf["error_rate"] > HIGH_ERROR_RATE_THRESHOLD and perf["top_errors"]:
                common_error_msg = perf["top_errors"][0][0] # Most common error message
                proposals.append({
                    "type": "correction", # Different type than general mutation
                    "target_fragment": name,
                    "error_context": common_error_msg, # Provide error context
                    "reason": f"High error rate ({perf['error_rate']:.1%}) with common error: {common_error_msg[:50]}..."
                })
                self._logger.info(f"Proposing correction for high-error fragment: {name}")
            
            # Proposal 3: (Conceptual) Adjust Orchestration Heuristic
            # If a fragment consistently fails on specific task types (needs more context in memory)
            # proposals.append({
            #     "type": "orchestration_tuning",
            #     "target_fragment": name,
            #     "suggestion": "Avoid using for task type X, prefer fragment Y",
            #     "reason": "Consistently low success rate on task type X"
            # })

        # Proposal 4: (Conceptual) Suggest new skill/fragment based on common unhandled objectives (needs objective data in memory)

        self._logger.info(f"Generated {len(proposals)} improvement proposals.")
        return proposals

    async def _validate_changes(self, proposals: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Validates the proposed changes (basic checks)."""
        self._logger.info(f"Validating {len(proposals)} proposals...")
        validated_proposals = []
        rejected_proposals = []

        # Requires FragmentRegistry access for validation
        registered_fragments = set(self.fragment_registry.get_all_definitions().keys()) if self.fragment_registry else set()

        for prop in proposals:
            is_valid = True
            rejection_reason = ""

            target = prop.get("target_fragment")
            prop_type = prop.get("type")

            if not target or not prop_type:
                is_valid = False
                rejection_reason = "Missing target_fragment or type"
            elif target not in registered_fragments:
                # Check if it's a potential mutation name (e.g., ends with _mut_X)
                # Basic check, could be improved
                if not (target.startswith("_") or re.match(r".*_mut\\\d+$", target)):
                    is_valid = False
                    rejection_reason = f"Target fragment '{target}' not found in registry."
                else:
                    self._logger.debug(f"Target fragment '{target}' not in registry, assuming it's a mutation or internal.")
            elif prop_type in ["mutation", "correction"] and not prop.get("strategy") and not prop.get("error_context"):
                # Requires strategy for mutation or context for correction
                is_valid = False
                rejection_reason = f"Missing strategy/error_context for type '{prop_type}'"

            # Add more specific validation rules per proposal type here

            if is_valid:
                self._logger.debug(f"Proposal validated: {prop}")
                validated_proposals.append(prop)
            else:
                self._logger.warning(f"Proposal rejected: {prop}. Reason: {rejection_reason}")
                prop["validation_error"] = rejection_reason # Add error to proposal dict
                rejected_proposals.append(prop)

        self._logger.info(f"Validation complete: {len(validated_proposals)} accepted, {len(rejected_proposals)} rejected.")
        return {"validated": validated_proposals, "rejected": rejected_proposals}

# --- Example Usage/Testing (Optional) ---
async def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    import time # Add import for timing

    # Create dummy workspace
    test_workspace = Path("./temp_metalearner_workspace")
    test_workspace.mkdir(exist_ok=True)
    insight_dir = test_workspace / "a3x/a3net/data"
    # Don't create dir here, let the fragment do it.

    # Mock MemoryManager (Use placeholder defined above or create a more elaborate one)
    class MockMemoryManager(MemoryManager):
         async def get_recent_episodes(self, limit: int) -> List[Dict[str, Any]]:
              # More diverse dummy data
              data = [
                 {'id': i, 'context': f'Task {chr(65+i%3)}', 'action': f'Fragment_{i%4}', 'outcome': 'success' if i%2==0 else 'failure', 'timestamp': f'2023-10-28T10:{i:02d}:00', 'metadata': json.dumps({"fragment_name": f'Fragment_{i%4}', "confidence": 0.6 + (i%5)*0.1, "a3l_command": f"do something {i}"}) } for i in range(limit)
              ]
              print(f"MockMemoryManager: Returning {len(data)} dummy episodes.")
              return data

    # Mock Context
    class MockContext(FragmentContext):
         def __init__(self, workspace):
              self.memory = MockMemoryManager()
              self.workspace_root = workspace

    # Instantiate Fragment
    fragment = MetaLearnerFragment() # Assuming BaseFragment doesn't need args
    context = MockContext(str(test_workspace.resolve()))

    # Execute Fragment
    print("--- Executing MetaLearnerFragment ---")
    result = await fragment.execute(num_episodes=30) # Analyze 30 dummy episodes
    print(f"\nExecution Result:")
    print(json.dumps(result, indent=2, default=str))

    # Check insight file content
    insight_file_path = insight_dir / "meta_insights.jsonl"
    if insight_file_path.exists():
         print(f"\n--- Content of {insight_file_path.name} ({insight_file_path.resolve()}) ---")
         print(insight_file_path.read_text(encoding='utf-8'))
    else:
         print(f"Insight file {insight_file_path} was not created.")

if __name__ == '__main__':
    import asyncio
    # Add re import for validation regex
    import re
    asyncio.run(main()) 