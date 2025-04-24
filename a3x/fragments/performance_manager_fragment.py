import logging
import jsonlines
import re
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

# Assuming BaseFragment is importable from a core module
try:
    # <<< MODIFIED: Import FragmentContext >>>
    from a3x.fragments.base import BaseFragment, FragmentContext, FragmentDef
except ImportError:
    # Fallback if the exact path is different or for testing
    # <<< MODIFIED: Update placeholder to include FragmentContext >>>
    class FragmentContext: pass 
    class BaseFragment:
        def __init__(self, ctx: Optional[FragmentContext] = None, **kwargs): # Adjusted signature
            self.context = ctx or {}
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.debug(f"BaseFragment initialized with context: {ctx}, kwargs: {kwargs}")

        # <<< MODIFIED: Update placeholder execute signature >>>
        async def execute(self, context: FragmentContext, **kwargs) -> Dict[str, Any]: 
            raise NotImplementedError
        
        def get_name(self) -> str: # Add placeholder get_name
            return self.__class__.__name__

# <<< ADDED Import for ActionIntent >>>
from a3x.core.types import ActionIntent

logger = logging.getLogger(__name__)


EVALUATION_SUMMARY_PATH = "a3x/a3net/data/evaluation_summary.jsonl"
# Regex to identify mutations: captures base name and mutation number
MUTATION_PATTERN = re.compile(r"^(.*?)_mut_(\d+)$")

# <<< ADDED Import for decorator >>>
from .registry import fragment 

# <<< ADDED Decorator >>>
@fragment(
    name="performance_manager",
    description="Analyzes evaluation results and generates A3L commands for promotion/archival.",
    category="Management",
    capabilities=["performance_analysis", "fragment_reorganization"]
)
class PerformanceManagerFragment(BaseFragment):
    """
    Analyzes evaluation results to identify high-performing mutations
    and generate A3L commands for promotion and archival.
    """

    # --- Fragment Definition ---
    def __init__(self, ctx: FragmentContext):
        super().__init__(ctx=ctx)
        # Access config from ctx if needed for eval file path
        self.evaluation_file = ctx.config.get("performance_manager", {}).get("evaluation_file", EVALUATION_SUMMARY_PATH)
        logger.info(f"[{self.get_name()}] Initialized. Evaluation file: {self.evaluation_file}")

    def _parse_fragment_name(self, fragment_name: str) -> Tuple[Optional[str], Optional[int]]:
        """Parses a fragment name to identify if it's a mutation and its base."""
        match = MUTATION_PATTERN.match(fragment_name)
        if match:
            base_name = match.group(1)
            mutation_num = int(match.group(2))
            return base_name, mutation_num
        return None, None # Not a mutation

    async def execute(self, context: FragmentContext) -> Dict[str, Any]:
        """
        Reads evaluation scores, compares base fragments with mutations,
        and generates ActionIntents for reorganization.
        """
        # <<< MODIFIED: Get logger from context >>>
        logger = context.logger 
        logger.info(f"[{self.get_name()}] Executing...")
        latest_scores: Dict[str, float] = {}
        fragment_families: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"base_score": None, "mutations": {}})

        # Use workspace_root from context if available
        eval_file_path = self.evaluation_file
        if context.workspace_root:
            eval_file_path = context.workspace_root / self.evaluation_file

        try:
            # <<< MODIFIED: Use eval_file_path >>>
            with jsonlines.open(eval_file_path, mode='r') as reader:
                for evaluation in reader:
                    fragment_name = evaluation.get("fragment")
                    score = evaluation.get("score")

                    if fragment_name and isinstance(score, (int, float)):
                        # Store the latest score found for each fragment
                        latest_scores[fragment_name] = float(score)
                else:
                        logger.warning(f"Skipping invalid/incomplete evaluation record: {evaluation}")

        except FileNotFoundError:
            logger.error(f"Evaluation summary file not found: {eval_file_path}")
            # <<< MODIFIED: Return structure consistent with Fragment execution >>>
            return {"status": "error", "message": f"Evaluation file not found: {eval_file_path}", "task_complete": True} 
        except Exception as e:
            logger.exception(f"Error reading evaluation summary file: {eval_file_path}")
            # <<< MODIFIED: Return structure consistent with Fragment execution >>>
            return {"status": "error", "message": f"Error reading evaluation file: {e}", "task_complete": True}

        # Group fragments into families (base + mutations)
        for fragment_name, score in latest_scores.items():
            base_name, _ = self._parse_fragment_name(fragment_name)
            if base_name:
                # It's a mutation, add it to its family's mutations
                fragment_families[base_name]["mutations"][fragment_name] = score
            else:
                # It's potentially a base fragment
                # Ensure the base fragment itself is in the families dict if it exists
                if fragment_name not in fragment_families:
                     fragment_families[fragment_name] # Initialize if not seen via mutation
                fragment_families[fragment_name]["base_score"] = score


        # Decide actions for each family
        for base_name, family_data in fragment_families.items():
            base_score = family_data.get("base_score")
            mutations = family_data.get("mutations", {})

            if not mutations:
                # No mutations for this base, nothing to compare
                logger.debug(f"No mutations found for base fragment '{base_name}'. Skipping comparison.")
                continue

            if base_score is None:
                logger.warning(f"Base fragment '{base_name}' has mutations {list(mutations.keys())} but no evaluation score found. Cannot compare.")
                # Decide if we should archive mutations without a base score? For now, skip.
                continue

            # Find the best mutation
            best_mutation_name: Optional[str] = None
            best_mutation_score: float = -float('inf')
            for mut_name, mut_score in mutations.items():
                if mut_score > best_mutation_score:
                    best_mutation_score = mut_score
                    best_mutation_name = mut_name

            # Decision logic
            if best_mutation_name and best_mutation_score > base_score:
                # Promote best mutation. 
                logger.info(f"[{self.get_name()}] Promoting '{best_mutation_name}' (Score: {best_mutation_score}) over base '{base_name}' (Score: {base_score}).")
                
                # <<< MODIFIED: Set ActionIntent instead of generating A3L >>>
                if context.shared_task_context:
                    intent = ActionIntent(
                        requested_by=self.get_name(), # Use fragment's name
                        skill_target="promover_fragmento",
                        parameters={"fragment_name": best_mutation_name}
                    )
                    context.shared_task_context.set_action_intent(intent)
                    logger.info(f"[{self.get_name()}] Set ActionIntent to promote '{best_mutation_name}'.")
                    
                    # Stop processing further actions in this run and return status indicating action requested
                    return {
                        "status": "waiting_for_action", # Indicate an action was requested
                        "reasoning": f"Requested promotion of '{best_mutation_name}' (Score: {best_mutation_score:.3f}) over base '{base_name}' (Score: {base_score:.3f}) via ActionIntent.",
                        "task_complete": False # Task is not complete, waiting for action
                    }
                else:
                    logger.error(f"[{self.get_name()}] SharedTaskContext not found in FragmentContext. Cannot set ActionIntent.")
                    return {"status": "error", "message": "SharedTaskContext unavailable, cannot request action.", "task_complete": True}
                
                # <<< REMOVED: A3L command generation and summary message append for promotion >>>
                # a3l_commands.append(f"promover fragmento '{best_mutation_name}'")
                # a3l_commands.append(f"arquivar fragmento '{base_name}'") # Also removed archival for now
                # summary_messages.append(f"'{best_mutation_name}' (Score: {best_mutation_score:.3f}) promoted, replacing base '{base_name}' (Score: {base_score:.3f}).")
                
                # <<< REMOVED: Archival of other mutations for simplicity >>>
                # for mut_name in mutations:
                #     if mut_name != best_mutation_name:
                #         logger.info(f"Archiving other mutation '{mut_name}' for base '{base_name}'.")
                #         a3l_commands.append(f"arquivar fragmento '{mut_name}'")

            else:
                # Base is better or equal, archive all mutations (For now, just log, no ActionIntent)
                logger.info(f"[{self.get_name()}] Base fragment '{base_name}' (Score: {base_score}) outperforms mutations. Archiving mutations (skipped in this step): {list(mutations.keys())}.")
                # summary_messages.append(f"Base '{base_name}' (Score: {base_score:.3f}) kept. Archiving mutations: {list(mutations.keys())}.")
                # for mut_name in mutations:
                #     a3l_commands.append(f"arquivar fragmento '{mut_name}'")

        # <<< MODIFIED: Final return if no action was triggered >>>
        logger.info(f"[{self.get_name()}] Performance management analysis complete. No promotion action triggered in this pass.")
        return {
            "status": "success",
            "message": "Analysis complete. No immediate promotion action taken.",
            # "a3l_commands": a3l_commands, # Removed
            # "summary": final_summary # Removed
            "task_complete": True # Mark task as complete if no action was needed
        }

# Example Usage (Conceptual) - Keep commented out
# async def main():
#     # Setup logger
#     logging.basicConfig(level=logging.INFO)
#     # Create dummy evaluation file
#     dummy_data = [
#         {"fragment": "fragA", "score": 0.70, "label": "ok"},
#         {"fragment": "fragB", "score": 0.85, "label": "good"},
#         {"fragment": "fragA_mut_1", "score": 0.75, "label": "better"},
#         {"fragment": "fragC", "score": 0.60}, # Missing label
#         {"fragment": "fragB_mut_1", "score": 0.82, "label": "worse"},
#         {"fragment": "fragA_mut_2", "score": 0.65, "label": "worse"},
#         {"fragment": "fragB_mut_2", "score": 0.88, "label": "best"},
#         {"fragment": "fragD"}, # Missing score
#         {"fragment": "fragA", "score": 0.73, "label": "latest_A"}, # Newer score for A
#         {"fragment": "fragE", "score": 0.90, "label": "good"},
#         {"fragment": "fragE_mut_1", "score": 0.91, "label": "better"},
#     ]
#     with jsonlines.open(EVALUATION_SUMMARY_PATH, mode='w') as writer:
#         writer.write_all(dummy_data)
#
#     manager = PerformanceManagerFragment()
#     result = await manager.execute()
#     print(json.dumps(result, indent=2))
#
# if __name__ == "__main__":
#     import asyncio
#     import json
#     asyncio.run(main()) 
#     # Expected output structure (exact commands depend on logic):
#     # {
#     #   "status": "success",
#     #   "a3l_commands": [
#     #     "promover fragmento 'fragA_mut_1'",
#     #     "arquivar fragmento 'fragA'",
#     #     "arquivar fragmento 'fragA_mut_2'",
#     #     "promover fragmento 'fragB_mut_2'",
#     #     "arquivar fragmento 'fragB'",
#     #     "arquivar fragmento 'fragB_mut_1'",
#     #     "promover fragmento 'fragE_mut_1'",
#     #     "arquivar fragmento 'fragE'"
#     #   ],
#     #   "summary": "'fragA_mut_1' (Score: 0.750) promoted, replacing base 'fragA' (Score: 0.730). | 'fragB_mut_2' (Score: 0.880) promoted, replacing base 'fragB' (Score: 0.850). | 'fragE_mut_1' (Score: 0.910) promoted, replacing base 'fragE' (Score: 0.900)."
#     # }