# core/execution_logic.py
import logging
from typing import Dict, Any, List, AsyncGenerator

# Assuming necessary imports from core modules will be added
from core.tool_executor import execute_tool
# Need to import CerebrumXAgent for type hinting if agent is passed
# from .cerebrumx import CerebrumXAgent # Circular import? Pass necessary methods/attributes instead.
# Consider passing agent methods/attributes directly or using a Protocol

logger = logging.getLogger(__name__)


async def _execute_actual_plan_step(
    agent, step_objective: str, context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes a single step from the generated plan by identifying the core tool
    and invoking the base ReactAgent's run loop for that specific sub-objective.

    Args:
        agent: The agent instance (provides access to memory, tools, run method).
        step_objective (str): The specific objective for this step.
        context (Dict[str, Any]): Current context (may not be directly used by base run).

    Returns:
        Dict[str, Any]: Result dictionary from the base run loop execution.
                       Format: {"status": "success/error", "message": "..."}
    """
    logger.info(
        f"Executing Step Objective: '{step_objective}' using base ReactAgent logic..."
    )
    try:
        # Use the run method inherited from ReactAgent (or overridden if needed)
        # The base 'run' should handle the Thought-Action-Observation loop for this step
        final_step_result = {}
        # ReactAgent.run returns an AsyncGenerator
        async for result_chunk in agent.run(objective=step_objective):
            # In a simple execution, we might just care about the final result.
            # The base `run` should yield intermediate steps if needed,
            # but for step execution, we often wait for the final outcome.
            if isinstance(result_chunk, str):  # Handle final answer string directly
                final_step_result = {"status": "success", "message": result_chunk}
                # TODO: Ensure ReactAgent.run consistently yields a dict for final status
            elif isinstance(result_chunk, dict) and result_chunk.get(
                "status"
            ):  # Check for final status dict
                final_step_result = result_chunk
            # else: process intermediate thoughts/actions if needed

        if not final_step_result:
            logger.warning(
                f"Base agent run for step '{step_objective}' finished without yielding a final result dict or string."
            )
            final_step_result = {
                "status": "error",
                "message": "Step execution finished with unknown state.",
            }

        # Store result in short-term memory
        memory_content = f"Execution result for '{step_objective}': {final_step_result.get('status', 'N/A')} - {final_step_result.get('message', '')}"
        agent.add_history_entry(
            "assistant", memory_content
        )  # Use agent's history method
        logger.info(
            f"Step '{step_objective}' execution completed. Status: {final_step_result.get('status')}"
        )
        return final_step_result

    except Exception as e:
        logger.exception(f"Exception during execution of step '{step_objective}':")
        error_result = {
            "status": "error",
            "message": f"Exception during step execution: {e}",
        }
        # Store error in short-term memory
        memory_content = (
            f"Execution result for '{step_objective}': error - Exception: {e}"
        )
        agent.add_history_entry("assistant", memory_content)
        return error_result


async def execute_plan_with_reflection(
    agent, plan: List[str], context: Dict[str, Any]
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Executes the plan step-by-step, incorporating simulation and reflection.

    Args:
        agent: The agent instance (provides access to methods like _simulate_step, _plan_hierarchically, tools, memory etc.).
        plan (List[str]): The list of plan steps (objectives).
        context (Dict[str, Any]): The current context.

    Yields:
        Dict[str, Any]: Dictionaries representing simulation, pre-reflection, and execution results.
            - {"type": "simulation", "step_index": int, "content": dict}
            - {"type": "pre_reflection", "step_index": int, "content": dict}
            - {"type": "execution_step", "step_index": int, "result": dict}
            - {"type": "modification_trigger", "step_index": int, "reason": str}
            - {"type": "replan", "content": list}
    """
    step_index = 0
    max_steps = (
        len(plan) * 2
    )  # Limit steps to prevent infinite loops if modification keeps failing
    executed_step_count = 0
    execution_results = []  # Store results for final reflection phase

    # original_plan = list(plan) # Keep a copy for context if needed

    while step_index < len(plan) and executed_step_count < max_steps:
        current_step_objective = plan[step_index]
        logger.info(
            f"--- Processing Step {step_index + 1}/{len(plan)}: {current_step_objective[:80]}... ---"
        )
        executed_step_count += 1

        # 1. Simulate
        simulation = await agent._simulate_step(current_step_objective, context)
        yield {"type": "simulation", "step_index": step_index, "content": simulation}

        # 2. Reflect on Simulation (Pre-execution Reflection)
        reflection_input = {
            "step": current_step_objective,
            "simulated_outcome": simulation.get(
                "simulated_outcome", "Simulation failed or missing outcome."
            ),
            "context": context,
        }
        # Default reflection outcome in case of error calling the tool
        step_reflection = {
            "status": "error",
            "decision": "skip",
            "justification": "Default skip due to reflection error calling tool",
        }
        try:
            step_reflection = await execute_tool(
                tool_name="reflect_plan_step",
                action_input=reflection_input,
                tools_dict=agent.tools,
                agent_logger=logger,  # Use execution_logic logger
                agent_memory=agent._memory,
            )
        except Exception as reflect_err:
            logger.exception(
                f"Error executing reflect_plan_step tool for step {step_index}: {current_step_objective}"
            )
            step_reflection["justification"] = (
                f"Reflection tool execution failed: {reflect_err}"
            )

        yield {
            "type": "pre_reflection",
            "step_index": step_index,
            "content": step_reflection,
        }

        # 3. Decide and Act (Execute, Skip, Modify)
        step_decision = step_reflection.get("decision", "unknown")
        step_justification = step_reflection.get("justification", "N/A")
        step_result = None

        if step_decision == "execute":
            logger.info(
                f"--- Executing Step {step_index + 1}/{len(plan)} based on reflection: {current_step_objective[:80]}... ---"
            )
            step_result = await _execute_actual_plan_step(
                agent, current_step_objective, context
            )
            step_index += 1  # Move to next step only on successful execution or skip

        elif step_decision == "skip":
            logger.info(
                f"--- Skipping Step {step_index + 1}/{len(plan)} based on reflection: {current_step_objective[:80]}... Reason: {step_justification} ---"
            )
            step_result = {
                "status": "skipped",
                "action": "step_skipped",
                "data": {
                    "message": f"Step skipped due to reflection. Reason: {step_justification}"
                },
            }
            step_index += 1  # Move to next step

        elif step_decision == "modify":
            logger.warning(
                f"--- Plan Modification Required for Step {step_index + 1} based on reflection: {current_step_objective[:80]}... Reason: {step_justification} ---"
            )
            yield {
                "type": "modification_trigger",
                "step_index": step_index,
                "reason": step_justification,
            }

            # Signal back to the agent/caller that replanning is needed.
            logger.error(
                "Replanning required but not implemented within execution_logic. Signaling modification and skipping step."
            )
            step_result = {
                "status": "skipped",
                "action": "step_skipped_modification_required",
                "data": {
                    "message": f"Modification required but replanning must be handled by caller. Reason: {step_justification}"
                },
            }
            step_index += (
                1  # Skip the problematic step to avoid loop within this function
            )

        else:  # Unknown decision
            logger.error(
                f"--- Skipping Step {step_index + 1}/{len(plan)} due to unknown reflection decision ('{step_decision}'): {current_step_objective[:80]}... Reason: {step_justification} ---"
            )
            step_result = {
                "status": "skipped",
                "action": "step_skipped_unknown_decision",
                "data": {
                    "message": f"Step skipped due to unknown reflection decision: {step_decision}. Reason: {step_justification}"
                },
            }
            step_index += 1  # Move to next step

        if step_result:
            execution_results.append(step_result)
            yield {
                "type": "execution_step",
                "step_index": step_index - 1
                if step_decision != "modify"
                else step_index,
                "result": step_result,
            }
        else:
            logger.error(
                f"No step result generated for step index {step_index}. This indicates a logic error."
            )
            execution_results.append(
                {
                    "status": "error",
                    "action": "internal_loop_error",
                    "data": {"message": "No result for step."},
                }
            )
            step_index += 1  # Prevent infinite loop

    logger.info(
        f"Finished executing plan steps. Total steps processed in loop: {executed_step_count}"
    )


# Potential helper functions related to execution might go here
