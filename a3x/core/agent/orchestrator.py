    # --- Main Orchestration Method (Logic from Agent.run_task) ---
    async def orchestrate(self, objective: str, max_steps: Optional[int] = None) -> Dict:
        """Orchestrates task execution by delegating to Fragments."""
        log_prefix = "[TaskOrchestrator]"
        self.logger.info(f"{log_prefix} Starting task orchestration: {objective}")

        # --- Initialization ---
        task_id = str(uuid.uuid4())
        shared_task_context = SharedTaskContext(task_id=task_id, initial_objective=objective)
        self.logger.info(f"{log_prefix} Initialized SharedTaskContext with ID: {task_id}")
        notify_new_task(objective)

        full_task_history: List[Dict[str, str]] = [] # Stores detailed steps from fragments
        orchestration_history: List[Tuple[str, str]] = [] # High-level orchestrator decisions
        current_step = 0
        max_orchestration_steps = max_steps or 20 # Define a max loop count
        self.logger.info(f"{log_prefix} Maximum orchestration steps allowed: {max_orchestration_steps}")
        final_result: Dict = {}
        task_completed_successfully = False # Flag to indicate successful completion

        # --- Orchestration Loop ---
        while current_step < max_orchestration_steps and not task_completed_successfully:
            current_step += 1
            self.logger.info(f"{log_prefix} Orchestration step {current_step}/{max_orchestration_steps}")

            # 1. Determine Component and Sub-task
            component_name, sub_task = await self._get_next_step_delegation(
                objective=objective, 
                history=orchestration_history, # Pass high-level history for decisions
                shared_task_context=shared_task_context
            )

            if not component_name or not sub_task:
                error_msg = "Failed to get delegation decision from LLM."
                self.logger.error(f"{log_prefix} {error_msg}")
                final_result = {
                    "status": "error", 
                    "message": error_msg,
                    "final_answer": error_msg,
                    "shared_task_context": shared_task_context,
                    "full_history": full_task_history
                }
                notify_task_error(task_id, "Delegation Failed", final_result)
                break # Exit loop on delegation failure

            # Add delegation decision to high-level history
            orchestration_history.append((f"Step {current_step}: Delegate to {component_name}", sub_task))
            notify_fragment_selection(component_name)

            # --- Check for explicit termination before execution (e.g., if LLM delegates directly to a non-existent 'End' component) ---
            # This is a potential future improvement, currently handled by validation in _get_next_step_delegation
            # if component_name == "End": # Or some other termination signal
            #     self.logger.info(f"{log_prefix} Received termination signal from LLM.")
            #     # Decide how to finalize: assume success if no prior errors?
            #     final_result = { ... success state ... }
            #     task_completed_successfully = True
            #     notify_task_completion(task_id, final_result)
            #     break

            # 2. Execute the Fragment Task
            fragment_success, fragment_result = await self._execute_fragment_task(
                component_name, sub_task, shared_task_context
            )
            
            # Add fragment's detailed history to the main history
            if isinstance(fragment_result, dict) and "full_history" in fragment_result:
                 full_task_history.extend(fragment_result["full_history"])
            elif isinstance(fragment_result, dict) and "history" in fragment_result:
                 self.logger.warning(f"{log_prefix} Fragment {component_name} returned 'history' instead of 'full_history'. Appending anyway.")
                 full_task_history.extend(fragment_result["history"])

            # 3. Process Fragment Result
            # --- Modify this block --- 
            # Check specifically if the FinalAnswerProvider ran successfully
            if component_name == "FinalAnswerProvider" and fragment_success:
                 self.logger.info(f"{log_prefix} Final answer provided by FinalAnswerProvider. Task complete.")
                 # Extract the actual answer content from the fragment result
                 final_answer_content = fragment_result.get("final_answer", fragment_result.get("message", "Final answer process finished."))
                 final_result = {
                     "status": "success",
                     "message": f"Orchestration completed by {component_name}.",
                     "final_answer": final_answer_content, # Use the extracted answer
                     "full_history": full_task_history,
                     "shared_task_context": shared_task_context
                 }
                 task_completed_successfully = True # Set flag to exit loop
                 notify_task_completion(task_id, final_result)
                 # DO NOT break here - let the loop condition handle the exit

            # Handle other fragments providing a final answer (less common scenario)
            elif fragment_success and isinstance(fragment_result, dict) and fragment_result.get("final_answer"):
                self.logger.warning(f"{log_prefix} Final answer received unexpectedly from {component_name}: {fragment_result['final_answer']}")
                final_result = {
                    "status": "success",
                    "message": f"Orchestration completed unexpectedly by {component_name}.",
                    "final_answer": fragment_result["final_answer"],
                    "full_history": full_task_history,
                    "shared_task_context": shared_task_context
                }
                task_completed_successfully = True # Set flag to exit loop
                notify_task_completion(task_id, final_result)
                # DO NOT break here - let the loop condition handle the exit

            elif not fragment_success:
                 # Fragment execution failed (Keep existing logic with break)
                 error_msg = fragment_result.get("message", f"Fragment {component_name} failed without a specific message.")
                 self.logger.error(f"{log_prefix} Fragment {component_name} execution failed: {error_msg}")
                 final_result = {
                     "status": "error", 
                     "message": f"Error during {component_name} execution: {error_msg}",
                     "final_answer": f"Error during {component_name} execution: {error_msg}",
                     "full_history": full_task_history,
                     "shared_task_context": shared_task_context
                 }
                 notify_task_error(task_id, f"{component_name} Failed", final_result)
                 break # Exit loop immediately on fragment failure
            else:
                 # Fragment completed sub-task (Keep existing logic)
                 if isinstance(fragment_result, dict) and fragment_result.get("context_updates"):
                      shared_task_context.update_data(fragment_result["context_updates"])
                      self.logger.info(f"{log_prefix} Updated shared context from {component_name} result.")
                 self.logger.info(f"{log_prefix} {component_name} completed sub-task. Continuing orchestration.")

        # --- End of Orchestration Loop ---

        # --- Modify the conditions for handling loop exit --- 
        # Handle loop exit due to max steps (only if not already successfully completed)
        if not task_completed_successfully and not final_result and current_step >= max_orchestration_steps:
            self.logger.warning(f"{log_prefix} Task exceeded max steps ({max_orchestration_steps}). Objective: {objective}")
            final_result = {
                "status": "error", 
                "message": f"Task exceeded maximum steps ({max_orchestration_steps}).",
                "final_answer": f"Task exceeded maximum steps ({max_orchestration_steps}).",
                "full_history": full_task_history,
                "shared_task_context": shared_task_context
            }
            add_episodic_record(f"Orchestrator reached max steps ({max_orchestration_steps}) for objective: {objective}", "orchestrator_max_steps_reached", json.dumps({"last_history": full_task_history[-3:]}), {"status": "max_steps_error"})
            notify_task_error(task_id, "Max Steps Exceeded", final_result)

        # Ensure some result is set if loop finished unexpectedly (only if not successfully completed)
        if not task_completed_successfully and not final_result:
             self.logger.error(f"{log_prefix} Orchestration loop finished without a definitive result. Objective: {objective}")
             final_result = {
                 "status": "error", 
                 "message": "Orchestration finished without a result.",
                 "final_answer": "Orchestration finished without a result.",
                 "full_history": full_task_history,
                 "shared_task_context": shared_task_context
            }
             add_episodic_record(f"Orchestrator finished unknown state for objective: {objective}", "orchestrator_unknown_end_state", json.dumps({"last_history": full_task_history[-3:]}), {"status": "unknown_error"})
             notify_task_error(task_id, "Unknown Orchestration End", final_result)

        # --- Finalization ---
        # Invoke learning cycle regardless of success/failure if needed
        await self._invoke_learning_cycle(objective, full_task_history, final_result.get("status", "unknown"), shared_task_context)
        self.logger.info(f"{log_prefix} Recorded final episodic memory state for task {task_id}")

        if "full_history" not in final_result:
            final_result["full_history"] = full_task_history
            
        self.logger.info(f"{log_prefix} Orchestration finished for objective '{objective}'. Final Status: {final_result.get('status')}. Result Keys: {list(final_result.keys())}")
        return final_result 