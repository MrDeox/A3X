"""
Skill para automatizar o ciclo completo de simulação, aprendizado e refinamento
do prompt da skill simulate_decision_reflection.
"""

import logging
import json
import os
import datetime
from typing import Dict, Any, Optional

# Core imports
from a3x.core.tools import skill

# Direct imports of helper skill functions
from a3x.skills.core.call_skill_by_name import call_skill_by_name
from a3x.skills.core.append_to_file_path import append_to_file_path

# Assume SkillContext provides access to logger

HISTORY_LOG_DIR = os.path.join("memory", "learning_history")
HISTORY_LOG_FILE = os.path.join(HISTORY_LOG_DIR, "auto_prompt_evolution.jsonl")
DEFAULT_USER_INPUT = "Vale a pena sair do meu emprego atual para empreender com IA?"
APPLY_REFINEMENT_SKILL = "apply_prompt_refinement_from_logs" # Name of the skill that does the refinement

logger = logging.getLogger(__name__)

@skill(
    name="auto_improve_simulation_prompt",
    description="Executa um ciclo completo de simulação->aprendizado->refinamento para melhorar o prompt da skill simulate_decision_reflection.",
    parameters={
        "user_input": (Optional[str], None) # Opcional – se não for passado, use um padrão
    }
)
async def auto_improve_simulation_prompt(ctx, user_input: Optional[str] = None) -> Dict[str, Any]:
    """
    Orchestrates the simulate, learn, and refine cycle for the simulation prompt
    using directly imported helper skill functions.

    Args:
        ctx: The skill execution context.
        user_input: Optional input question for the simulation. Uses a default if None.

    Returns:
        A dictionary summarizing the results of the auto-improvement cycle.
    """
    ctx.logger.info("--- Starting Auto Prompt Improvement Cycle --- ")
    start_time_absolute = datetime.datetime.now(datetime.timezone.utc)
    ctx.logger.info(f"[TIMER] Cycle started at: {start_time_absolute.isoformat()}")
    last_timestamp = start_time_absolute

    # ---> Initial LLM Readiness Check <---
    # llm_is_ready = await wait_for_llm_ready(ctx, timeout=90)
    # if not llm_is_ready:
    #     error_msg = "Initial LLM readiness check failed. Aborting auto-improvement cycle."
    #     ctx.logger.error(error_msg)
    #     try:
    #         await _log_cycle_history(ctx, start_time_absolute, user_input or DEFAULT_USER_INPUT,
    #                                "[Simulation not run]", "[Learning not run]",
    #                                "[Refinement not run]", "[Not Applied]", error_msg)
    #     except Exception as log_err:
    #         ctx.logger.error(f"Failed to log history after LLM readiness failure: {log_err}")
    #     return {"status": "error", "error": error_msg}
    # -----------------------------------

    # --- Step 1: Determine Input --- #
    simulation_input = user_input if user_input else DEFAULT_USER_INPUT
    ctx.logger.info(f"Using simulation input: '{simulation_input}'")

    # Storage for results
    simulated_reflection = "[Simulation not run or failed]"
    insights = "[Learning not run or failed]"
    new_prompt = "[Refinement not run or failed]"
    refinement_status = "[Refinement not run or failed]"
    final_status = "Ciclo concluído com erros."

    # --- Helper: Use directly imported skill functions --- #
    async def run_dependent_skill(skill_name: str, skill_args: Dict = {}, required: bool = True):
        ctx.logger.info(f"Calling '{skill_name}' via imported call_skill_by_name...")
        try:
            # Directly use the imported function
            result = await call_skill_by_name(ctx, skill_name=skill_name, skill_args=skill_args)

            if "error" in result:
                ctx.logger.error(f"Call to '{skill_name}' failed: {result['error']}")
                if required:
                    raise ValueError(f"Required dependent skill '{skill_name}' failed: {result['error']}")
            else:
                 ctx.logger.info(f"Call to '{skill_name}' successful.")
            return result
        except Exception as e:
            # Catch potential errors if call_skill_by_name itself fails unexpectedly
            ctx.logger.exception(f"Unexpected error calling call_skill_by_name for '{skill_name}':")
            if required:
                raise ValueError(f"Critical failure during call_skill_by_name execution for '{skill_name}'.") from e
            return {"error": f"Unexpected error calling call_skill_by_name: {e}"}

    try:
        # --- Step 2: Simulate Reflection --- #
        step2_start_time = datetime.datetime.now(datetime.timezone.utc)
        ctx.logger.info(f"[TIMER] Before Simulate Reflection. Elapsed since start: {(step2_start_time - start_time_absolute).total_seconds():.2f}s. Since last step: {(step2_start_time - last_timestamp).total_seconds():.2f}s")
        last_timestamp = step2_start_time
        sim_result = await run_dependent_skill(
            skill_name="simulate_decision_reflection",
            skill_args={"user_input": simulation_input}
        )
        # Assuming the simulation skill returns {"simulated_reflection": "..."} on success
        simulated_reflection = sim_result.get("simulated_reflection", simulated_reflection)

        # --- Step 3: Learn from Logs --- #
        step3_start_time = datetime.datetime.now(datetime.timezone.utc)
        ctx.logger.info(f"[TIMER] Before Learn from Logs. Elapsed since start: {(step3_start_time - start_time_absolute).total_seconds():.2f}s. Since last step: {(step3_start_time - last_timestamp).total_seconds():.2f}s")
        last_timestamp = step3_start_time
        learn_result = await run_dependent_skill(
            skill_name="learn_from_reflection_logs",
            skill_args={"n_logs": 5} # Or use offset? Stick to n_logs=5 for now
        )
        insights = learn_result.get("insights_from_logs", insights)

        # --- Step 4: Apply Refinement --- #
        step4_start_time = datetime.datetime.now(datetime.timezone.utc)
        ctx.logger.info(f"[TIMER] Before Apply Refinement. Elapsed since start: {(step4_start_time - start_time_absolute).total_seconds():.2f}s. Since last step: {(step4_start_time - last_timestamp).total_seconds():.2f}s")
        last_timestamp = step4_start_time
        refine_result = await run_dependent_skill(
            skill_name=APPLY_REFINEMENT_SKILL,
            skill_args={} # apply_prompt_refinement needs no args currently
        )
        new_prompt = refine_result.get("new_prompt", new_prompt)
        refinement_status = refine_result.get("status", refinement_status)

        # Determine final status based on refinement outcome
        if refinement_status == "Prompt atualizado com sucesso.":
            final_status = "Ciclo concluído e prompt atualizado."
        else:
            # Include the error from refine_result if available
            refine_error = refine_result.get("error", "Unknown refinement error")
            final_status = f"Ciclo concluído, mas refinamento do prompt falhou: {refine_error}"
            ctx.logger.warning(f"Prompt refinement step failed: {refine_error}")

    except Exception as cycle_error:
        ctx.logger.exception("Error during auto-improvement cycle execution:")
        final_status = f"Ciclo interrompido por erro: {cycle_error}"
        # insights, new_prompt, etc., will retain their default error state or last value

    # --- Call the refactored logging function ---
    await _log_cycle_history(ctx, start_time_absolute, simulation_input,
                             simulated_reflection, insights, refinement_status,
                             new_prompt, final_status)

    # --- Step 6: Return Summary --- #
    summary_result = {
        "status": final_status,
        "simulation_input": simulation_input,
        "insights_snippet": insights[:200] + ("..." if len(insights) > 200 else ""),
        "refinement_status": refinement_status,
        "new_prompt_applied_snippet": new_prompt[:200] + ("..." if len(new_prompt) > 200 else "") if refinement_status == "Prompt atualizado com sucesso." else "[Not Applied]"
    }
    end_time_absolute = datetime.datetime.now(datetime.timezone.utc)
    total_duration = (end_time_absolute - start_time_absolute).total_seconds()
    ctx.logger.info(f"[TIMER] --- Auto Prompt Improvement Cycle Finished: {final_status} --- Total Duration: {total_duration:.2f}s")
    return summary_result

# --- Step 5: Log History (Refactored into helper) --- #
async def _log_cycle_history(ctx, start_time, sim_input, reflection, insights, refinement_status, new_prompt, final_status):
    try:
        step5_start_time = datetime.datetime.now(datetime.timezone.utc)
        ctx.logger.info(f"[TIMER] Before Log History. Elapsed since start: {(step5_start_time - start_time).total_seconds():.2f}s.")
        ctx.logger.info(f"Logging auto-improvement cycle results to {HISTORY_LOG_FILE}...")
        log_entry = {
            "timestamp": start_time.isoformat(),
            "end_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "user_input": sim_input,
            "reflection": reflection, # Could be large
            "insights": insights, # Could be large
            "refinement_status": refinement_status,
            "new_prompt_applied": new_prompt if refinement_status == "Prompt atualizado com sucesso." else "[Not Applied]",
            "final_status_cycle": final_status
        }
        log_text = json.dumps(log_entry, ensure_ascii=False)

        # Use directly imported append_to_file_path function
        write_result = await append_to_file_path(ctx, path=HISTORY_LOG_FILE, text=log_text)
        if write_result.get("status") != "ok":
            ctx.logger.error(f"Failed to append to history log using skill function: {write_result.get('error')}")

    except Exception as log_e:
        ctx.logger.exception(f"Failed to prepare or write auto-improvement history log entry:")

# TODO:
# - Consider making n_logs configurable.
# - Error handling could be more granular.
# - Assumes ctx provides skill registry for call_skill_by_name to function.
# - NOTE: This version directly imports helper functions, creating module coupling.
#   A more robust solution involves context injection or a dedicated skill caller in ctx. 