# a3x/fragments/competitor_fragment.py
import logging
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Core A3X Imports ---
from a3x.fragments.base import BaseFragment, FragmentContext # Use real imports
from a3x.fragments.registry import fragment # Use real decorator

logger = logging.getLogger(__name__)

# --- Constants ---
MUTATION_HISTORY_FILE = "a3x/a3net/data/mutation_history.jsonl"
EVALUATION_SUMMARY_FILE = "a3x/a3net/data/evaluation_summary.jsonl"
COMPETITION_LOG_FILE = "a3x/a3net/data/fragment_competitions.jsonl"
DEFAULT_PERFORMANCE_METRIC = "success_rate"
MIN_PERFORMANCE_VALUE = -1.0 # Value assigned if metric is missing or null

# --- Fragment Registration ---
@fragment(name="competidor", description="Avalia a competicao entre um fragmento base e suas mutacoes") # Use real decorator
# --- End Registration ---

class CompetitorFragment(BaseFragment):
    """
    Compares the performance of a base fragment against its recently generated mutations.

    Reads the latest mutation event from mutation_history.jsonl and the latest
    performance summary from evaluation_summary.jsonl. It compares the specified
    performance metric (default: success_rate) and identifies the best-performing
    fragment (base or mutation).

    If a mutation performs better than the base fragment, it suggests A3L commands
    to promote the mutation and archive the base.

    Logs the competition details to fragment_competitions.jsonl.

    A3L Trigger: "avaliar competicao de fragmentos"
    """

    async def execute(self, ctx: FragmentContext, metric: str = DEFAULT_PERFORMANCE_METRIC, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes the fragment competition comparison.

        Args:
            ctx: The fragment execution context.
            metric: The performance metric to compare (key within fragment_stats).
            **kwargs: Additional arguments.

        Returns:
            A dictionary containing the status, message, and suggested A3L commands.
        """
        logger.info(f"Starting fragment competition evaluation using metric: '{metric}'")

        if not ctx.workspace_root:
            logger.error("Workspace root not found in context. Cannot locate log files.")
            return {"status": "error", "message": "Workspace root missing from context.", "suggested_a3l_commands": []}

        workspace_path = Path(ctx.workspace_root)
        mutation_log = workspace_path / MUTATION_HISTORY_FILE
        evaluation_log = workspace_path / EVALUATION_SUMMARY_FILE
        competition_log = workspace_path / COMPETITION_LOG_FILE

        # 1. Read Latest Mutation Event
        latest_mutation_event = self._read_last_jsonl_entry(mutation_log)
        if not latest_mutation_event:
            msg = f"Could not read latest mutation event from {mutation_log}"
            logger.error(msg)
            return {"status": "error", "message": msg, "suggested_a3l_commands": []}

        base_fragment_name = latest_mutation_event.get("base_fragment_name")
        mutations = latest_mutation_event.get("generated_mutations", [])
        mutation_names = [m.get("mutation_name") for m in mutations if m.get("mutation_name")]

        if not base_fragment_name or not mutation_names:
             msg = "Latest mutation event lacks base_fragment_name or generated_mutations."
             logger.error(f"{msg} Event: {latest_mutation_event}")
             return {"status": "error", "message": msg, "suggested_a3l_commands": []}

        logger.info(f"Identified latest mutation group: Base='{base_fragment_name}', Mutations={mutation_names}")
        competitors = [base_fragment_name] + mutation_names

        # 2. Read Latest Evaluation Summary
        latest_evaluation = self._read_last_jsonl_entry(evaluation_log)
        if not latest_evaluation:
            msg = f"Could not read latest evaluation summary from {evaluation_log}"
            logger.error(msg)
            return {"status": "error", "message": msg, "suggested_a3l_commands": []}

        fragment_stats = latest_evaluation.get("fragment_stats", {})
        if not fragment_stats:
            msg = "Latest evaluation summary lacks 'fragment_stats'. Cannot compare performance."
            logger.error(msg)
            return {"status": "error", "message": msg, "suggested_a3l_commands": []}

        # 3. Extract Performance Metrics for Competitors
        performance_data: Dict[str, float] = {}
        for name in competitors:
            stats = fragment_stats.get(name)
            if stats and stats.get(metric) is not None:
                performance_data[name] = float(stats[metric])
            else:
                performance_data[name] = MIN_PERFORMANCE_VALUE # Assign low score if missing/null
                logger.warning(f"Metric '{metric}' not found or null for fragment '{name}'. Assigning default low score.")

        # 4. Compare Performance and Select Winner
        if not performance_data:
             msg = "No performance data could be extracted for any competitor."
             logger.error(msg)
             return {"status": "error", "message": msg, "suggested_a3l_commands": []}

        # Find the competitor with the highest score
        winner_name, winning_score = max(performance_data.items(), key=lambda item: item[1])

        logger.info(f"Competition results ({metric}): {performance_data}")
        logger.info(f"Winner: '{winner_name}' with score {winning_score:.4f}")

        # 5. Generate A3L Commands if a Mutation Wins
        suggested_a3l_commands: List[str] = []
        action_taken = "none"
        if winner_name != base_fragment_name and winner_name in mutation_names:
            # Check if the winning score is actually valid (not the default low score)
            if winning_score > MIN_PERFORMANCE_VALUE:
                promote_cmd = f"promover fragmento {winner_name}"
                archive_cmd = f"arquivar fragmento {base_fragment_name}"
                suggested_a3l_commands = [promote_cmd, archive_cmd]
                action_taken = "promote_mutation"
                logger.info(f"Suggesting actions: {suggested_a3l_commands}")
            else:
                 logger.info(f"Mutation '{winner_name}' had the highest score ({winning_score}), but it was the default low score. No action suggested.")
                 action_taken = "mutation_best_but_invalid_score"
        elif winner_name == base_fragment_name:
            logger.info(f"Base fragment '{base_fragment_name}' remains the best performer. No action suggested.")
            action_taken = "base_fragment_best"
        else:
             logger.warning(f"Winner '{winner_name}' is neither the base nor in the mutation list? This should not happen.")
             action_taken = "winner_unknown" # Should not happen

        # 6. Log Competition Details
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds'),
            "base_fragment_name": base_fragment_name,
            "mutation_names": mutation_names,
            "competitors": competitors,
            "performance_metric": metric,
            "performance_data": performance_data,
            "winner": winner_name,
            "winning_score": winning_score,
            "action_taken": action_taken,
            "suggested_a3l_commands": suggested_a3l_commands,
            "source_mutation_event_ts": latest_mutation_event.get("timestamp"),
            "source_evaluation_ts": latest_evaluation.get("timestamp"),
            "triggered_by": self.get_name()
        }
        self._log_competition_event(competition_log, log_entry)

        # 7. Return Result
        message = f"Competition for base '{base_fragment_name}' complete. Winner: '{winner_name}' ({metric}={winning_score:.4f}). "
        if suggested_a3l_commands:
            message += f"Suggested A3L commands: {', '.join(suggested_a3l_commands)}"
        else:
            message += "No promotion/archiving actions suggested."

        return {
            "status": "success",
            "message": message.strip(),
            "winner": winner_name,
            "winning_score": winning_score,
            "suggested_a3l_commands": suggested_a3l_commands
        }

    def _read_last_jsonl_entry(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Reads the last valid JSON object from a JSONL file."""
        if not file_path.is_file():
            logger.warning(f"File not found: {file_path}")
            return None
        try:
            with open(file_path, "rb") as f: # Read in binary mode for seeking
                try:
                    f.seek(-2, 2) # Go to the second last character (hoping it's before last newline)
                    while f.read(1) != b'\n':
                        f.seek(-2, 1)
                        if f.tell() == 0: # Reached start of file
                             f.seek(0, 0)
                             break
                except OSError: # Handle file smaller than buffer or seek error
                    f.seek(0, 0)

                last_line = f.readline().decode("utf-8")
                if not last_line:
                     # Try reading the whole file if readline failed (e.g., single line file)
                     f.seek(0,0)
                     content = f.read().decode("utf-8").strip()
                     if content:
                         last_line = content

            if last_line:
                try:
                    return json.loads(last_line)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode last line from {file_path}: {e}. Line: '{last_line[:100]}...'", exc_info=True)
                    return None
            else:
                logger.warning(f"File {file_path} seems empty or has no valid lines.")
                return None
        except Exception as e:
            logger.error(f"Error reading last line from {file_path}: {e}", exc_info=True)
            return None

    def _log_competition_event(self, file_path: Path, log_entry: Dict[str, Any]):
        """Appends a JSON entry to the competition log file."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "a", encoding="utf-8") as f:
                json_line = json.dumps(log_entry, ensure_ascii=False)
                f.write(json_line + "\n")
            logger.info(f"Competition event logged to: {file_path}")
        except IOError as e:
            logger.error(f"Failed to log competition event to {file_path}: {e}", exc_info=True) 