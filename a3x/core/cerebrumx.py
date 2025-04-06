# core/cerebrumx.py
import logging
from typing import Dict, Any, List, AsyncGenerator, Optional

# Import base agent and other necessary core components
# from core.agent import ReactAgent
from a3x.core.agent import ReactAgent
# from core.tools import get_tool_descriptions  # <<< ADDED import
from a3x.core.tools import get_tool_descriptions  # <<< ADDED import
# from core.tool_executor import execute_tool  # <<< ADDED import
from a3x.core.tool_executor import execute_tool  # <<< ADDED import

# <<< ADDED Import for new execution logic >>>
# from core.execution_logic import execute_plan_with_reflection
from a3x.core.execution_logic import execute_plan_with_reflection
# Potentially import memory, reflection components later

# Initialize logger for this module
cerebrumx_logger = logging.getLogger(__name__)


class CerebrumXAgent(ReactAgent):  # Inheriting from ReactAgent for now
    """
    Agente Autônomo Adaptável Experimental com ciclo cognitivo CerebrumX.
    Expande o ReactAgent com percepção, planejamento hierárquico, simulação e reflexão.
    """

    def __init__(self, system_prompt: str, llm_url: Optional[str] = None):
        """Inicializa o Agente CerebrumX."""
        super().__init__(system_prompt, llm_url)
        self.initial_perception = (
            None  # Store initial perception for potential replanning
        )
        cerebrumx_logger.info("[CerebrumX INIT] Agente CerebrumX inicializado.")
        # TODO: Initialize specific CerebrumX components (e.g., perception handler, planner, simulator, reflector)

    async def run_cerebrumx_cycle(
        self, initial_perception: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executa o ciclo cognitivo completo do CerebrumX.
        Yields dictionaries representing each step of the cognitive cycle.
        """
        cerebrumx_logger.info(
            f"--- Iniciando Ciclo CerebrumX --- Perception inicial: {str(initial_perception)[:100]}..."
        )
        self.initial_perception = (
            initial_perception  # Store for potential use in replanning
        )

        # --- 1. Percepção ---
        processed_perception = self._perceive(initial_perception)
        yield {"type": "perception", "content": processed_perception}

        # --- 2. Recuperação de Contexto ---
        context = await self._retrieve_context(processed_perception)
        yield {"type": "context_retrieval", "content": context}

        # --- 3. Planejamento Hierárquico ---
        plan = await self._plan_hierarchically(processed_perception, context)
        yield {"type": "planning", "content": plan}

        # --- 4-5. Dynamic Execution Loop (Delegated) ---
        cerebrumx_logger.info(
            "--- Iniciando Execução do Plano com Simulação e Reflexão ---"
        )
        execution_results = []  # Collect execution results for final reflection
        # modification_needed = False # F841

        async for execution_event in execute_plan_with_reflection(self, plan, context):
            # Pass through the yields from the execution logic
            yield execution_event

            # Collect execution results
            if execution_event.get("type") == "execution_step":
                execution_results.append(execution_event.get("result"))
            # Handle replanning trigger if needed
            elif execution_event.get("type") == "modification_trigger":
                # modification_needed = True # F841
                step_index = execution_event.get("step_index")
                reason = execution_event.get("reason", "Unknown reason")
                cerebrumx_logger.warning(
                    f"Modification trigger received for step {step_index}. Reason: {reason}. Attempting to replan."
                )

                # --- Attempt Re-planning ---
                # Create a new context/objective for the planner
                replanning_perception_data = {
                    # Use the original processed perception + modification info
                    "processed": processed_perception.get(
                        "processed", "Original Objective Missing"
                    ),
                    "modification_request": {
                        "failed_step_index": step_index,
                        "failed_step": plan[step_index]
                        if step_index < len(plan)
                        else "N/A",
                        "reason": reason,
                        "current_plan": plan,
                    },
                }

                new_plan = await self._plan_hierarchically(
                    replanning_perception_data, context
                )  # Call planner again

                if (
                    new_plan != plan and new_plan
                ):  # Check if plan actually changed and is not empty
                    cerebrumx_logger.info(
                        f"Re-planning successful. New plan generated with {len(new_plan)} steps. Restarting execution loop with new plan."
                    )
                    plan = new_plan  # Replace the old plan
                    yield {"type": "replan", "content": plan}  # Signal the replan
                    # Restart the execution loop with the new plan
                    execution_results = []  # Reset results for the new plan
                    # modification_needed = ( # F841
                    #     False  # Reset flag, allow execution_logic to skip the step
                    # )
                    pass # Flag is not actually used, just log and let logic proceed
                else:
                    cerebrumx_logger.error(
                        "Re-planning failed or did not change the plan. Execution will continue skipping modified step."
                    )
                    # modification_needed = ( # F841
                    #     False  # Reset flag, allow execution_logic to skip the step
                    # )
                    pass # Flag is not actually used, just log and let logic proceed

        # --- 6. Reflexão (Pós-Execução Geral) ---
        cerebrumx_logger.info("--- Iniciando Reflexão Pós-Execução ---")
        reflection = await self._reflect(processed_perception, plan, execution_results)
        yield {"type": "reflection", "content": reflection}

        # --- 7. Aprendizagem (Atualização da Memória) ---
        cerebrumx_logger.info("--- Iniciando Aprendizagem/Atualização da Memória ---")
        await self._learn(reflection)
        yield {"type": "learning_update", "status": "success"}

        cerebrumx_logger.info("--- Ciclo CerebrumX Concluído ---")

        # Determine final answer based on execution results or reflection
        final_answer = "CerebrumX cycle completed."  # Default
        if reflection and reflection.get("summary"):  # Prioritize reflection summary
            final_answer = reflection.get("summary")
        elif execution_results:  # Fallback to last step result if reflection is minimal
            last_result = execution_results[-1]
            final_answer = f"Last step status: {last_result.get('status')}. Message: {last_result.get('data', {}).get('message', 'N/A')}"
        # We might need a dedicated 'final_answer' tool call based on reflection

        yield {"type": "final_answer", "content": final_answer}

    # Placeholder methods for the new cycle stages
    def _perceive(self, perception_input: Any) -> Dict[str, Any]:
        cerebrumx_logger.info("Processing perception...")
        # TODO: Implementar lógica de percepção (extrair dados relevantes do input)
        # Store initial perception if not already done
        if self.initial_perception is None:
            self.initial_perception = perception_input
        # Process input - simple pass-through for now
        if isinstance(perception_input, dict) and "processed" in perception_input:
            return perception_input  # Already processed (e.g., during replanning)
        return {"processed": perception_input}

    async def _retrieve_context(
        self, processed_perception: Dict[str, Any]
    ) -> Dict[str, Any]:
        cerebrumx_logger.info("Retrieving context from memory...")

        semantic_match = "No semantic match found."
        query = processed_perception.get("processed", "")

        if query and hasattr(self._memory, "retrieve_relevant_context"):
            try:
                # Call the actual memory retrieval method
                memory_result = await self._memory.retrieve_relevant_context(
                    query=query,
                    max_results=5,  # Default max results, consider making configurable
                )
                # Assuming memory_result is a dict or similar structure containing matches
                # Adapt based on the actual return format of retrieve_relevant_context
                if (
                    isinstance(memory_result, dict)
                    and "semantic_match" in memory_result
                ):
                    semantic_match = memory_result["semantic_match"]
                elif isinstance(memory_result, str):  # Handle simple string return
                    semantic_match = memory_result
                elif memory_result:  # Handle other non-empty results
                    semantic_match = str(memory_result)
                else:
                    cerebrumx_logger.info(
                        "Memory retrieval returned no relevant semantic context."
                    )

            except Exception:
                cerebrumx_logger.exception(
                    f"Error retrieving semantic context from memory for query '{query[:50]}...'"
                )
                semantic_match = "Error during semantic memory retrieval."
        elif not hasattr(self._memory, "retrieve_relevant_context"):
            cerebrumx_logger.warning(
                "_memory object does not have 'retrieve_relevant_context' method."
            )
            semantic_match = "Memory retrieval method not available."
        else:
            semantic_match = "No query provided for semantic search."

        # Combine with short-term history
        retrieved = {
            "short_term_history": self.get_history(),  # Get current conversation history
            "semantic_match": semantic_match,  # Use the retrieved or placeholder match
        }
        # TODO: Add retrieval from episodic memory based on relevance or time
        return {"retrieved_context": retrieved}

    async def _plan_hierarchically(
        self, perception: Dict[str, Any], context: Dict[str, Any]
    ) -> List[str]:
        cerebrumx_logger.info("Generating hierarchical plan using planning skill...")

        # 1. Get available tool descriptions (needed by the planner skill)
        tool_desc = get_tool_descriptions()

        # 2. Prepare input for the hierarchical_planner skill
        objective = perception.get("processed")
        # Add modification info if present
        if "modification_request" in perception:
            objective += f"\n[REPLANNING CONTEXT: Need to modify plan. Failed Step: {perception['modification_request'].get('failed_step')}. Reason: {perception['modification_request'].get('reason')}]"

        planner_input = {
            "objective": objective,
            "available_tools": tool_desc,
            "context": context,  # Pass the retrieved context
        }

        # 3. Execute the planner skill using execute_tool
        planner_result = {
            "status": "error",
            "data": {"message": "Planner execution failed."},
        }  # Default
        try:
            planner_result = await execute_tool(
                tool_name="hierarchical_planner",
                action_input=planner_input,
                tools_dict=self.tools,  # Agent has access to the tools registry via self.tools
                agent_logger=cerebrumx_logger,  # Use the module-level logger for this agent's context
                agent_memory=self._memory,  # Pass agent's memory
            )
        except Exception as e:
            cerebrumx_logger.exception(
                "Exception occurred while executing the planner tool."
            )
            planner_result["data"]["message"] = f"Exception calling planner: {e}"

        # 4. Process the result
        if planner_result.get("status") == "success":
            plan_list = planner_result.get("data", {}).get("plan", [])
            if isinstance(plan_list, list) and plan_list:
                cerebrumx_logger.info(
                    f"Hierarchical plan generated successfully ({len(plan_list)} steps)."
                )
                return plan_list
            else:
                cerebrumx_logger.warning(
                    "Planner skill succeeded but returned an empty or invalid plan list."
                )
                objective_str = str(perception.get("processed", "Fallback objective"))
                return [f"Address objective directly: '{objective_str[:50]}...'"]
        else:
            error_msg = planner_result.get("data", {}).get(
                "message", "Unknown planning error"
            )
            cerebrumx_logger.error(f"Hierarchical planner skill failed: {error_msg}")
            objective_str = str(perception.get("processed", "Fallback objective"))
            return [
                f"Address objective directly (planning failed): '{objective_str[:50]}...'"
            ]

    async def _simulate_step(
        self, plan_step: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        cerebrumx_logger.info(f"Simulating step: {plan_step[:50]}...")
        # --- Use the simulate_step skill ---
        simulation_input = {"step": plan_step, "context": context}
        simulation_result = {  # Default error result
            "status": "error",
            "simulated_outcome": "Exception during simulation call",
            "confidence": "N/A",
        }
        try:
            tool_result = await execute_tool(
                tool_name="simulate_step",
                action_input=simulation_input,
                tools_dict=self.tools,
                agent_logger=cerebrumx_logger,
                agent_memory=self._memory,
            )

            if tool_result.get("status") == "success":
                simulation_result = {
                    "simulated_outcome": tool_result.get(
                        "simulated_outcome",
                        "Simulation success, but no outcome text provided.",
                    ),
                    "confidence": tool_result.get("confidence", "N/A"),
                }
            else:
                error_msg = tool_result.get(
                    "error_message", "Simulation skill failed without specific message."
                )
                cerebrumx_logger.error(
                    f"Simulation skill failed for step '{plan_step[:50]}...': {error_msg}"
                )
                simulation_result["simulated_outcome"] = (
                    f"Simulation Failed: {error_msg}"
                )

        except Exception as e:
            cerebrumx_logger.exception(
                f"Exception calling simulate_step skill for step '{plan_step[:50]}...':"
            )
            # Keep default error message
            simulation_result["simulated_outcome"] = (
                f"Exception during simulation call: {e}"
            )

        return simulation_result  # Always return a dict

    # <<< REMOVED _execute_plan_step method (logic moved to execution_logic.py) >>>
    # async def _execute_plan_step(self, step_objective: str, context: Dict[str, Any]) -> Dict[str, Any]:
    #     ...

    async def _reflect(
        self,
        perception: Dict[str, Any],
        plan: List[str],
        execution_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        cerebrumx_logger.info("Reflecting on overall execution...")

        objective = perception.get("processed", "Unknown objective")
        learnings = []
        success_count = 0
        total_steps_attempted = len(execution_results)
        plan_completed = total_steps_attempted == len(plan)

        # Analyze each step result
        for i, result in enumerate(execution_results):
            step_description = (
                plan[i] if i < len(plan) else f"Step {i + 1} (description missing)"
            )
            step_summary = f"Step {i + 1} ('{step_description[:30]}...')"
            status = result.get("status", "unknown")
            message = result.get("message", "No message provided.")

            if status == "success":
                success_count += 1
                learnings.append(
                    {
                        "type": "success",
                        "step_index": i,
                        "step_description": step_description,
                        "content": f"{step_summary}: Completed successfully. Result: {str(message)[:100]}...",
                    }
                )
            elif status == "skipped":
                learnings.append(
                    {
                        "type": "skipped",
                        "step_index": i,
                        "step_description": step_description,
                        "content": f"{step_summary}: Skipped. Reason: {str(message)[:100]}...",
                    }
                )
            else:  # error or unknown
                learnings.append(
                    {
                        "type": "failure",
                        "step_index": i,
                        "step_description": step_description,
                        "content": f"{step_summary}: Failed. Reason: {str(message)[:100]}...",
                    }
                )

        # Calculate final stats
        success_rate = (
            (success_count / total_steps_attempted) * 100
            if total_steps_attempted > 0
            else 0
        )

        # Generate overall assessment
        if not execution_results:
            summary = f"Objective '{objective[:50]}...': No steps were executed."
            overall_outcome = "unknown"
        elif success_rate == 100 and plan_completed:
            summary = f"Objective '{objective[:50]}...': Plan executed successfully. All {total_steps_attempted} steps completed."
            overall_outcome = "success"
            # Optionally add a summary learning point
            # learnings.append({"type": "summary", "content": "All plan steps executed successfully."})
        elif success_count > 0:
            summary = f"Objective '{objective[:50]}...': Plan partially executed. {success_count}/{total_steps_attempted} steps successful ({success_rate:.0f}%)."
            overall_outcome = "partial_success"
        else:  # 0 successes
            summary = f"Objective '{objective[:50]}...': Plan execution failed. {success_count}/{total_steps_attempted} steps successful ({success_rate:.0f}%)."
            overall_outcome = "failure"

        return {
            "assessment": summary,
            "success_rate": success_rate / 100.0,
            "overall_outcome": overall_outcome,  # Added field: success, failure, partial_success
            "learnings": learnings,  # List of dicts with structured info per step
        }

    async def _learn(self, reflection_output: Dict[str, Any]):
        cerebrumx_logger.info("Updating memory based on reflection...")

        learnings = reflection_output.get("learnings", [])
        overall_assessment = reflection_output.get(
            "assessment", "No assessment provided."
        )

        if not learnings:
            cerebrumx_logger.info(
                "No specific step learnings identified in this cycle."
            )
            # Optionally, still log the overall assessment
            log_entry = {"type": "cycle_summary", "assessment": overall_assessment}
        else:
            cerebrumx_logger.info(
                f"{len(learnings)} potential learning points identified."
            )
            # Process each learning point
            for learning in learnings:
                log_entry = {
                    "type": learning.get("type", "unknown_learning"),
                    "step_index": learning.get("step_index", -1),
                    "step_description": learning.get("step_description", "N/A"),
                    "content": learning.get("content", "N/A"),
                }
                # Attempt to save to memory
                if hasattr(self._memory, "add_episodic_record"):
                    try:
                        # Adapt the data format as needed by add_episodic_record
                        await self._memory.add_episodic_record(data=log_entry)
                        cerebrumx_logger.debug(
                            f"Saved learning to episodic memory: {log_entry['type']} for step {log_entry['step_index']}"
                        )
                    except Exception:
                        cerebrumx_logger.exception(
                            f"Failed to save learning point to episodic memory: {log_entry}"
                        )
                else:
                    cerebrumx_logger.warning(
                        "Cannot save learning to memory: _memory object lacks 'add_episodic_record' method."
                    )

        # Log the overall assessment separately for clarity
        cerebrumx_logger.info(f"Overall Execution Assessment: {overall_assessment}")


# Exemplo de como poderia ser chamado (requereria ajustes na CLI)
# async def main():
#     # Carregar prompt, etc.
#     system_prompt = "Você é CerebrumX..."
#     agent = CerebrumXAgent(system_prompt=system_prompt)
#     initial_input = "Resuma as notícias de hoje sobre IA."
#     async for output in agent.run_cerebrumx_cycle(initial_input):
#         print(output)

# if __name__ == "__main__":
#     asyncio.run(main())
