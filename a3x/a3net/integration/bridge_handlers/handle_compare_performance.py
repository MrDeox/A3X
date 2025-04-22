import logging
import json
from typing import Dict, Optional, Any, Callable, Awaitable

# Assuming these are available in the environment
from a3x.a3net.core.context_store import ContextStore
from a3x.a3net.integration.a3lang_interpreter import interpret_a3l_line
# <<< CORRECTED IMPORT PATH for utility >>>
from ...modules.utils import append_to_log 

logger = logging.getLogger(__name__)

async def handle_compare_performance(
    directive: Dict[str, Any],
    context_store: Optional[ContextStore],
    post_message_handler: Optional[Callable[..., Awaitable[None]]]
) -> Optional[Dict[str, Any]]:
    """Handles the 'comparar_desempenho' directive logic."""

    fragment_id = directive.get("fragment_id")
    task_name = directive.get("task_name")
    origin = directive.get("_origin", "Unknown Compare Origin")

    if not fragment_id or not task_name:
        logger.error("[A3X Bridge Handler - Compare] 'fragment_id' or 'task_name' missing.")
        return {"status": "error", "message": "'fragment_id' or 'task_name' missing for comparar_desempenho"}
    
    if not context_store:
         logger.error("[A3X Bridge Handler - Compare] ContextStore not available.")
         return {"status": "error", "message": "ContextStore unavailable"}

    if not post_message_handler:
         logger.error("[A3X Bridge Handler - Compare] post_message_handler not available.")
         return {"status": "error", "message": "post_message_handler missing"}

    logger.info(f"[A3X Bridge Handler - Compare] Comparing performance for fragment '{fragment_id}' on task '{task_name}' (Origin: {origin})...")

    try:
        # --- 1. Find evaluation results --- 
        eval_keys = await context_store.find_keys_by_tag(f"evaluation_{fragment_id}_{task_name}", limit=10)
        eval_results = []
        for key in eval_keys:
            value_str = await context_store.get_value(key)
            if value_str:
                try:
                    eval_data = json.loads(value_str)
                    if isinstance(eval_data, dict) and 'timestamp' in eval_data:
                         eval_results.append({"key": key, "data": eval_data})
                except json.JSONDecodeError:
                    logger.warning(f"[A3X Bridge Handler - Compare] Failed to decode JSON for eval key '{key}'")
        
        eval_results.sort(key=lambda x: x['data'].get('timestamp', 0), reverse=True)

        if len(eval_results) < 2:
            logger.warning(f"[A3X Bridge Handler - Compare] Less than 2 evaluation results found for '{fragment_id}' on task '{task_name}'. Cannot compare.")
            return {"status": "skipped", "message": "Not enough evaluation history to compare"}

        # --- 2. Compare latest two --- 
        latest_eval = eval_results[0]["data"]
        previous_eval = eval_results[1]["data"]
        latest_accuracy = latest_eval.get("accuracy")
        previous_accuracy = previous_eval.get("accuracy")

        comparison_summary = f"Latest Eval (Key: {eval_results[0]['key']}): {latest_eval}. Previous Eval (Key: {eval_results[1]['key']}): {previous_eval}."
        logger.info(f"[A3X Bridge Handler - Compare] {comparison_summary}")

        # --- 3. Decision Logic --- 
        decision = "keep"
        reason = ""
        performance_threshold = 0.1 
        min_accuracy_threshold = 0.3 

        if latest_accuracy is not None and previous_accuracy is not None:
            accuracy_drop = previous_accuracy - latest_accuracy
            if accuracy_drop > performance_threshold:
                decision = "delete"
                reason = f"Performance decreased significantly (Drop: {accuracy_drop:.2f} > Threshold: {performance_threshold:.2f})"
            elif latest_accuracy < min_accuracy_threshold:
                decision = "delete"
                reason = f"Performance below minimum threshold (Accuracy: {latest_accuracy:.2f} < Threshold: {min_accuracy_threshold:.2f})"
            else:
                reason = f"Performance stable or improved (Latest: {latest_accuracy:.2f}, Previous: {previous_accuracy:.2f})"
        elif latest_accuracy is not None and latest_accuracy < min_accuracy_threshold:
             decision = "delete"
             reason = f"Performance below minimum threshold (Accuracy: {latest_accuracy:.2f} < Threshold: {min_accuracy_threshold:.2f}) - No previous data."
        else:
             reason = "Could not compare accuracy (missing data). Keeping fragment."
             logger.warning(f"[A3X Bridge Handler - Compare] Could not compare accuracy for '{fragment_id}'. Keeping.")

        # --- 4. Action --- 
        logger.info(f"[A3X Bridge Handler - Compare] Decision for '{fragment_id}': {decision.upper()}. Reason: {reason}")
        append_to_log(f"# [COMPARAÇÃO] Fragmento '{fragment_id}' Tarefa '{task_name}': {decision.upper()}. Razão: {reason}")

        if decision == "delete":
            logger.warning(f"[A3X Bridge Handler - Compare] Triggering deletion of fragment '{fragment_id}'.")
            delete_directive_str = f"delete fragment '{fragment_id}'"
            delete_directive = interpret_a3l_line(delete_directive_str)
            if delete_directive:
                delete_directive["_origin"] = f"AdaptiveCompetition after comparing {fragment_id} on {task_name}"
                await post_message_handler(
                    message_type="a3l_command",
                    content=delete_directive,
                    target_fragment="Executor"
                )
                logger.info(f"[A3X Bridge Handler - Compare] Delete directive for '{fragment_id}' enqueued.")
                return {"status": "success", "message": f"Comparison complete. Fragment '{fragment_id}' marked for deletion.", "decision": "delete"}
            else:
                logger.error(f"[A3X Bridge Handler - Compare] Failed to create delete directive for '{fragment_id}'.")
                return {"status": "warning", "message": f"Comparison decided deletion, but failed to create directive for '{fragment_id}'.", "decision": "delete_failed"}
        else:
             return {"status": "success", "message": f"Comparison complete. Fragment '{fragment_id}' kept.", "decision": "keep"}

    except ImportError as imp_err:
        logger.error(f"[A3X Bridge Handler - Compare] Failed to import necessary function: {imp_err}")
        return {"status": "error", "message": f"ImportError during comparison handling: {imp_err}"}
    except Exception as e:
        logger.error(f"[A3X Bridge Handler - Compare] Exception during performance comparison: {e}", exc_info=True)
        return {"status": "error", "message": f"Comparison loop failed: {e}"} 