# a3x/fragments/evaluator_fragment.py
import logging


import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Counter as TypingCounter
from collections import Counter, defaultdict
from a3x.fragments.base import BaseFragment, FragmentContext, FragmentDef
from a3x.fragments.registry import fragment
from a3x.core.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

# --- Fragment Registration ---
# @fragment_decorator(name="evaluator", trigger_phrases=["avaliar desempenho dos fragmentos"])
# --- End Placeholder ---

@fragment(name="aggregate_evaluator", description="Avalia desempenho agregado dos fragments.")
class AggregateEvaluatorFragment(BaseFragment):
    """
    Evaluates the performance of fragments based on recent execution history,
    categorizes them, generates corresponding A3L suggestions (promote,
    re-evaluate, archive), and saves a summary report.

    Reads from: Episodic Memory via MemoryManager
    Saves report to: a3x/a3net/data/evaluation_summary.jsonl
    Output: List of A3L action suggestions.

    Invoked by A3L: "avaliar desempenho dos fragmentos"
    """

    DEFAULT_EPISODE_COUNT = 100
    # Path relative to workspace root
    REPORT_FILE = "a3x/a3net/data/evaluation_summary.jsonl"

    # Thresholds for categorization
    GOOD_THRESHOLD = 0.80  # >= 80% success
    REGULAR_THRESHOLD = 0.40 # >= 40% and < 80% success
    # < 40% success is considered weak

    MIN_EXECUTIONS_FOR_EVAL = 3 # Minimum times a fragment must run to be evaluated

    def __init__(self, ctx: FragmentContext):
        super().__init__(ctx)
        self.ctx = ctx
        if hasattr(ctx, 'memory_manager') and isinstance(ctx.memory_manager, MemoryManager):
            self.memory_manager = ctx.memory_manager
        else:
            self._logger.error("MemoryManager not found in context for AggregateEvaluatorFragment.")
            self.memory_manager = None # Handle error: execution will likely fail
        
        # Get workspace root for saving report
        self.workspace_root = ctx.workspace_root if hasattr(ctx, 'workspace_root') else None
        if not self.workspace_root:
            # Only log error if doing aggregate evaluation, specific eval doesn't need workspace root
            # self._logger.error("Workspace root not found in context. Cannot save report.")
            pass # Will be checked later if needed

    async def execute(self, fragment_to_evaluate: Optional[str] = None, num_episodes_to_evaluate: int = DEFAULT_EPISODE_COUNT, **kwargs) -> Dict[str, Any]:
        if not self.memory_manager:
            return {"status": "error", "message": "MemoryManager not initialized."}
        
        is_specific_evaluation = fragment_to_evaluate is not None
        
        if not is_specific_evaluation and not self.workspace_root:
            return {"status": "error", "message": "Workspace root not configured for aggregate report saving."}
            
        try:
            log_target = f"fragment '{fragment_to_evaluate}'" if is_specific_evaluation else "all fragments"
            self._logger.info(f"Executing Evaluator analysis for {log_target} on last {num_episodes_to_evaluate} episodes.")
            
            # Get recent episodes for evaluation
            # For specific evaluation, might filter here later if MemoryManager supports it efficiently
            episodes = await self.memory_manager.get_recent_episodes(limit=num_episodes_to_evaluate)
            
            # Evaluate performance (passes fragment_to_evaluate to the evaluation logic)
            performance_results = await self._evaluate_performance(episodes, fragment_to_evaluate)
            
            if is_specific_evaluation:
                # Return only the specific fragment's evaluation
                self._logger.info(f"Specific evaluation for '{fragment_to_evaluate}' finished.")
                return {
                    "status": "success",
                    "evaluation": performance_results # Should contain only the specific fragment
                }
            else:
                # Aggregate evaluation: generate suggestions and save report
                self._logger.info("Aggregate evaluation finished. Generating suggestions and saving report...")
                suggestions = await self._generate_suggestions(performance_results)
                await self._save_report(performance_results, suggestions)
                validation = await self._validate_improvements(suggestions)
                
                self._logger.info("AggregateEvaluatorFragment execution finished.")
                return {
                    "status": "success",
                    "evaluation": performance_results,
                    "suggestions": suggestions,
                    "validation": validation
                }
        except Exception as e:
            self._logger.error(f"Error in AggregateEvaluatorFragment.execute: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

    async def _evaluate_performance(self, episodes: List[Dict[str, Any]], fragment_to_evaluate: Optional[str] = None) -> Dict[str, Any]:
        """Analyzes episodes, calculates metrics, and classifies fragments.
        If fragment_to_evaluate is provided, only evaluates that fragment.
        """
        target_log = f"fragment '{fragment_to_evaluate}'" if fragment_to_evaluate else "all fragments"
        self._logger.info(f"Evaluating performance for {target_log} based on {len(episodes)} episodes...")
        if not episodes:
            return {"message": "No episodes for evaluation", "fragment_classifications": {}}

        fragment_stats = defaultdict(lambda: {"success": 0, "failure": 0, "total": 0})

        for episode in episodes:
            try:
                metadata_str = episode.get('metadata', '{}')
                metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else (metadata_str if isinstance(metadata_str, dict) else {})
                fragment_name = metadata.get("fragment_name") or episode.get('action')
                if not fragment_name:
                    continue
                
                # If specific evaluation, skip non-matching fragments
                if fragment_to_evaluate and fragment_name != fragment_to_evaluate:
                    continue

                stats = fragment_stats[fragment_name]
                stats["total"] += 1
                outcome = episode.get('outcome', 'unknown').lower()
                if outcome == 'success':
                    stats["success"] += 1
                elif outcome == 'failure':
                    stats["failure"] += 1
            except Exception as e:
                self._logger.warning(f"Error processing episode ID {episode.get('id')} during evaluation: {e}")

        results = {"fragment_classifications": {}}
        evaluated_count = 0
        # Iterate through collected stats (will be just one if fragment_to_evaluate was set)
        for name, stats in fragment_stats.items():
            if stats["total"] >= self.MIN_EXECUTIONS_FOR_EVAL:
                success_rate = (stats["success"] / stats["total"]) if stats["total"] > 0 else 0
                classification = "weak" # Default
                if success_rate >= self.GOOD_THRESHOLD:
                    classification = "good"
                elif success_rate >= self.REGULAR_THRESHOLD:
                    classification = "regular"
                
                results["fragment_classifications"][name] = {
                    "classification": classification,
                    "success_rate": round(success_rate, 3),
                    "total_executions": stats["total"]
                }
                evaluated_count += 1
            else:
                self._logger.debug(f"Skipping classification for fragment '{name}', not enough executions ({stats['total']}/{self.MIN_EXECUTIONS_FOR_EVAL}).")
                # Store minimum data even if not classified
                results["fragment_classifications"][name] = {
                    "classification": "insufficient_data",
                    "success_rate": None,
                    "total_executions": stats["total"]
                }
        
        log_msg_suffix = f"fragment '{fragment_to_evaluate}'" if fragment_to_evaluate else f"{evaluated_count} fragments"
        self._logger.info(f"Performance evaluation complete. Results for {log_msg_suffix}.")
        return results

    async def _generate_suggestions(self, performance_results: Dict[str, Any]) -> List[str]:
        """Generates A3L command suggestions based on fragment classifications."""
        self._logger.info("Generating A3L suggestions based on evaluation...")
        suggestions = []
        classifications = performance_results.get("fragment_classifications", {})

        for name, data in classifications.items():
            classification = data.get("classification")
            
            if classification == "good":
                # Suggest promotion (hypothetical command)
                suggestions.append(f"promover fragmento '{name}'")
                self._logger.debug(f"Suggesting promotion for good fragment: {name}")
            elif classification == "regular":
                # Suggest re-evaluation or monitoring
                suggestions.append(f"monitorar fragmento '{name}'") # Hypothetical
                self._logger.debug(f"Suggesting monitoring for regular fragment: {name}")
            elif classification == "weak":
                # Suggest archiving or mutation
                suggestions.append(f"arquivar fragmento '{name}'") # Hypothetical
                # Or maybe mutation:
                # suggestions.append(f"mutar fragmento '{name}' usando 'melhorar_desempenho'")
                self._logger.debug(f"Suggesting archiving/mutation for weak fragment: {name}")
        
        self._logger.info(f"Generated {len(suggestions)} A3L suggestions.")
        return suggestions

    async def _validate_improvements(self, suggestions: List[str]) -> Dict[str, Any]:
        """Basic validation placeholder for generated A3L suggestions."""
        self._logger.info(f"Validating {len(suggestions)} A3L suggestions...")
        # For now, just basic structural check (similar to EvolutionPlanner)
        is_valid = True
        errors = []
        if not isinstance(suggestions, list):
            is_valid = False
            errors.append("Suggestions is not a list.")
        else:
            for i, cmd in enumerate(suggestions):
                if not isinstance(cmd, str) or not cmd.strip():
                    is_valid = False
                    errors.append(f"Suggestion {i+1} is not a non-empty string: '{cmd}'")
        
        result = {"valid": is_valid, "errors": errors}
        if not is_valid:
            self._logger.warning(f"Suggestion validation failed: {errors}")
        else:
            self._logger.info("Suggestions validation successful (basic checks passed).")
        return result

    async def _save_report(self, performance: Dict[str, Any], suggestions: List[str]):
        """Saves the evaluation summary to a JSONL file."""
        if not self.workspace_root:
            self._logger.error("Cannot save report, workspace root is unknown.")
            return
        
        report_path = Path(self.workspace_root) / self.REPORT_FILE
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            report_entry = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "evaluation": performance,
                "suggestions": suggestions
            }
            with report_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(report_entry, ensure_ascii=False) + "\n")
            self._logger.info(f"Evaluation report saved to {report_path}")
        except Exception as e:
            self._logger.error(f"Failed to save evaluation report to {report_path}: {e}", exc_info=True)

# --- Fragment Definition ---
AggregateEvaluatorFragmentDef = FragmentDef(
    name="AggregateEvaluator",
    description="Agrega e avalia o desempenho de múltiplos fragmentos após mutações ou ciclos evolutivos.",
    fragment_class=AggregateEvaluatorFragment,
    skills=["agregar_avaliacoes"],
    managed_skills=["agregar_avaliacoes"],
    prompt_template="Avalie o desempenho agregado dos fragmentos informados após mutações ou ciclos evolutivos."
)

# --- Example Usage/Testing (Optional) ---
# async def main():
#     # Setup logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     import time

#     # Create dummy workspace
#     test_workspace = Path("./temp_evaluator_workspace")
#     test_workspace.mkdir(exist_ok=True)
#     report_dir = test_workspace / "a3x/a3net/data"
#     # Let the fragment create the dir

#     # Mock MemoryManager (Use placeholder defined above)

#     # Mock Context
#     class MockContext(FragmentContext):
#          def __init__(self, workspace):
#               self.memory = MemoryManager() # Use the placeholder with diverse data
#               self.workspace_root = workspace

#     # Instantiate Fragment
#     fragment = AggregateEvaluatorFragment()
#     context = MockContext(str(test_workspace.resolve()))

#     # Execute Fragment
#     print("--- Executing EvaluatorFragment ---")
#     result = await fragment.execute(context, num_episodes=50) # Analyze 50 dummy episodes
#     print(f"\nExecution Result:")
#     print(json.dumps(result, indent=2))

#     # Check report file content
#     report_file_path = report_dir / "evaluation_summary.jsonl"
#     if report_file_path.exists():
#         print(f"\n--- Content of {report_file_path} ---")
#         with open(report_file_path, 'r') as f:
#             # Read and print each JSON line
#             for line in f:
#                  try:
#                      print(json.dumps(json.loads(line), indent=2))
#                  except json.JSONDecodeError:
#                      print(f"Invalid JSON line: {line.strip()}")

#         # Clean up
#         # report_file_path.unlink()
#         # report_dir.rmdir()
#         # Path(test_workspace / "a3x/a3net").rmdir()
#         # Path(test_workspace / "a3x").rmdir()
#     else:
#         print(f"Report file {report_file_path} was not created.")

#     # Clean up workspace
#     # test_workspace.rmdir()

# if __name__ == "__main__":
#     import asyncio
#     import time
#     import datetime
#     asyncio.run(main()) 