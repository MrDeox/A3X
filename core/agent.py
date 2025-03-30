import logging
import re
import json
import datetime
import os
import sys
import traceback
from typing import Tuple, Optional, Dict, Any, List

import requests

# Local imports
from core.config import MAX_REACT_ITERATIONS, MAX_HISTORY_TURNS, LLAMA_SERVER_URL, LLAMA_DEFAULT_HEADERS, MAX_META_DEPTH, MAX_TOKENS_FALLBACK, CONTEXT_SIZE
from core.tools import TOOLS, get_tool_descriptions
# Removed memory skill import as it's not directly used here anymore
from core.db_utils import save_agent_state, load_agent_state
from core.prompts import get_react_prompt_v2 # Use v2 prompt
# <<< ADDED IMPORT >>>
from core.format_recovery import parse_react_output
from core.utils import agent_logger, setup_agent_logger, history_manager, tool_executor, llm_client, agent_parser, agent_reflector
# <<< REMOVE unused imports >>>
# from core import agent_error_handler, agent_autocorrect

# <<< IMPORT NEW INTERFACE >>>
from .llm_interface import call_llm

# Initialize logger
agent_logger = logging.getLogger(__name__)

# Constante para ID do estado do agente
AGENT_STATE_ID = 1

# --- Classe ReactAgent ---
class ReactAgent:
    def __init__(self, llm_url: str, system_prompt: str):
        """Inicializa o Agente ReAct."""
        self.llm_url = llm_url
        self.system_prompt = system_prompt
        self.tools = TOOLS  # Carrega as ferramentas
        self._history = [] # Histórico de Thought, Action, Observation
        self._memory = load_agent_state(AGENT_STATE_ID) # Carrega estado/memória inicial
        self.max_iterations = MAX_REACT_ITERATIONS
        self._current_plan = None # <<< Initialize plan >>>
        # Removed _last_error_type, not needed with new handlers
        agent_logger.info(f"[ReactAgent INIT] Agente inicializado. Memória carregada: {list(self._memory.keys())}")

    # --- run (Refatorado para iterar sobre o plano) ---
    async def run(self, objective: str) -> str:
        """Executa o ciclo ReAct (agora orientado por plano) para atingir o objetivo."""
        # Initialize potential final response
        final_response: Optional[str] = None
        log_prefix_base = "[ReactAgent]"
        self._current_plan = None # Ensure plan is reset
        plan_to_execute: List[str] = [] # Plan steps to iterate over

        try:
            # --- Setup History --- 
            self._history = []
            self._history.append(f"Human: {objective}")
            agent_logger.info(f"{log_prefix_base} Objetivo Inicial: '{objective[:100]}...'" )

            # --- Planning Phase --- 
            agent_logger.info("--- Generating Plan ---")
            tool_desc = get_tool_descriptions()
            generated_plan = await planner.generate_plan(objective, tool_desc, agent_logger, self.llm_url)
            
            if generated_plan:
                plan_to_execute = generated_plan
                plan_str = json.dumps(plan_to_execute, indent=2, ensure_ascii=False)
                agent_logger.info(f"Plan Generated:\n{plan_str}")
            else:
                agent_logger.warning("Failed to generate a plan. Proceeding with the original objective as a single-step plan.")
                plan_to_execute = [objective] # Fallback: Treat original objective as a 1-step plan
            
            agent_logger.info("--- Starting Plan Execution ---")

            # --- Plan Execution Loop --- 
            current_step_index = 0
            total_iterations = 0
            max_total_iterations = self.max_iterations # Use configured max iterations

            while current_step_index < len(plan_to_execute) and total_iterations < max_total_iterations:
                current_step = plan_to_execute[current_step_index]
                agent_logger.info(f"--- Executing Plan Step {current_step_index + 1}/{len(plan_to_execute)} --- Total Iterations: {total_iterations + 1}/{max_total_iterations}")

                should_break, final_response_from_cycle = await self._execute_react_cycle(
                    cycle_num=total_iterations + 1, 
                    log_prefix_base=log_prefix_base,
                    current_step_objective=current_step, # Pass step as the objective for this cycle
                    objective=objective, # Pass overall objective
                    plan=plan_to_execute, # Pass the plan
                    current_step_index=current_step_index # Pass step index
                )
                
                total_iterations += 1 

                if should_break:
                    final_response = final_response_from_cycle # Update final response (could be success or error)
                    agent_logger.info(f"{log_prefix_base} Execution loop breaking due to cycle result (Final Answer or Error). Step Index: {current_step_index}")
                    break # Exit the while loop
                else:
                    # If the cycle completed normally (no break), move to the next step
                    agent_logger.info(f"{log_prefix_base} Step {current_step_index + 1} completed. Moving to next step.")
                    current_step_index += 1

            # --- After Loop --- 
            if final_response is None: # Check if loop finished without a final answer or break
                if total_iterations >= max_total_iterations:
                    # Reached max iteration limit during plan execution
                    agent_logger.warning(f"{log_prefix_base} Maximum total iterations ({max_total_iterations}) reached during plan execution.")
                    last_observation = self._history[-1] if self._history and self._history[-1].startswith("Observation:") else "No observation."
                    final_response = f"Erro: Maximum total iterations ({max_total_iterations}) reached. Last observation: {last_observation}"
                elif current_step_index >= len(plan_to_execute):
                    # Completed all plan steps (and did *not* hit max iterations)
                    agent_logger.warning(f"{log_prefix_base} Plan execution completed, but no final answer action was explicitly returned.")
                    last_observation = self._history[-1] if self._history and self._history[-1].startswith("Observation:") else "No observation."
                    final_response = f"Plan completed. Last observation: {last_observation}" 
                else:
                    # Should not happen if loop logic is correct
                    agent_logger.error(f"{log_prefix_base} Loop exited unexpectedly without setting final_response.")
                    final_response = "Erro: Loop concluído inesperadamente."

            # --- End of ReAct Cycle Loop ---

        except Exception as main_loop_err:
            agent_logger.exception(f"{log_prefix_base} Exceção inesperada no loop principal ReAct: {main_loop_err}")
            final_response = f"Erro Interno Crítico no Agente: {main_loop_err}"

        finally:
            # --- Save State --- 
            agent_logger.info(f"{log_prefix_base} Salvando estado final do agente (ID: {AGENT_STATE_ID}).")
            save_agent_state(AGENT_STATE_ID, self._memory)
            agent_logger.info(f"{log_prefix_base} Ciclo ReAct finalizado. Retornando: '{final_response[:100]}...'" if final_response else "[No Final Response]")

        return final_response if final_response is not None else "Erro: O agente finalizou sem uma resposta definida."

    # --- _execute_react_cycle (Integrate Reflector) ---
    async def _execute_react_cycle(
        self,
        cycle_num: int,
        log_prefix_base: str,
        current_step_objective: str, # Renamed for clarity
        objective: str, # Overall objective
        plan: List[str], # Current plan
        current_step_index: int # Current step index
    ) -> Tuple[bool, Optional[str]]:
        """
        Executes a single ReAct cycle: Prompt LLM -> Parse Response -> Execute Action -> Reflect -> Decide.
        Returns (should_break_loop, response_on_break)
        """
        log_prefix = f"{log_prefix_base} Cycle {cycle_num}/{len(plan) if plan else '?'}"
        agent_logger.info(f"\n{log_prefix} (Step Objective: '{current_step_objective[:60]}...' Inicio: {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]})" )

        # --- START ADDED DEBUG LOGGING (Keep for now) ---
        agent_logger.debug(f"{log_prefix} --- Start Cycle Input ---")
        agent_logger.debug(f"{log_prefix} Objective: {current_step_objective}")
        try:
            history_to_log = self._history[-(MAX_HISTORY_TURNS * 2):]
            history_log_str = json.dumps(history_to_log, indent=2, ensure_ascii=False)
            agent_logger.debug(f"{log_prefix} Relevant History (Last ~{MAX_HISTORY_TURNS} turns):\n{history_log_str}")
        except Exception as log_err:
            agent_logger.error(f"{log_prefix} Failed to serialize history for logging: {log_err}")
        agent_logger.debug(f"{log_prefix} --- End Cycle Input ---")
        # --- END ADDED DEBUG LOGGING ---

        start_cycle_time = datetime.datetime.now()
        # 1. Prepare Prompt
        prompt = get_react_prompt_v2(
            objective=objective,
            plan=plan,
            current_step_index=current_step_index,
            history=self._history,
            tools_descriptions=self._get_tools_description(),
            scratchpad="", # Initial scratchpad is empty for the first LLM call in a cycle
            memory=self._memory,
        )

        # 2. Call LLM
        agent_logger.info(f"{log_prefix} Calling LLM...")
        llm_response_raw = await llm_client.generate_text(
            prompt=prompt,
            temperature=0.1, # Low temp for more deterministic action
            max_tokens=MAX_TOKENS_FALLBACK, # Use fallback max tokens
            stop=["Observation:"],
            agent_logger=agent_logger,
            context_size=CONTEXT_SIZE # Pass context size
        )

        if not llm_response_raw:
            agent_logger.error(f"{log_prefix} LLM call failed. Stopping plan.")
            # Reflect on the LLM failure
            observation_dict = {"status": "error", "action": "llm_failed", "data": {"message": "LLM did not return a response."}}
            decision, _ = await agent_reflector.reflect_on_observation(
                objective=objective, plan=plan, current_step_index=current_step_index,
                action_name="_call_llm", action_input={},
                observation_dict=observation_dict,
                history=self._history, memory=self._memory, agent_logger=agent_logger,
                agent_instance=self
            )
            return True, "Erro: Falha na chamada ao LLM."

        agent_logger.info(f"{log_prefix} LLM Response (Raw):\n{llm_response_raw}")
        self._history.append(f"LLM Response:\n{llm_response_raw}") # Add raw response for context

        # 3. Parse LLM Response using the new function
        agent_logger.info(f"{log_prefix} Parsing LLM response...")
        parsed_output = parse_react_output(llm_response_raw)

        if "error" in parsed_output:
            parse_error_str = parsed_output["error"]
            agent_logger.error(f"{log_prefix} Parsing Failed: {parse_error_str}")
            observation_dict = {"status": "error", "action": "parsing_failed", "data": {"message": parse_error_str}}
            observation_str = f"Observation: {json.dumps(observation_dict)}" # Keep observation simple
            self._history.append(observation_str)
            # Call Reflector for Parse Error
            decision, _ = await agent_reflector.reflect_on_observation(
                objective=objective, plan=plan, current_step_index=current_step_index,
                action_name="_parse_llm", action_input={},
                observation_dict=observation_dict,
                history=self._history, memory=self._memory, agent_logger=agent_logger,
                agent_instance=self
            )
            agent_logger.error(f"{log_prefix} Stopping plan due to LLM response parsing error (Reflector decision: {decision})")
            return True, f"Erro: Falha ao processar resposta do LLM ({parse_error_str})"
        
        # Extract successfully parsed components
        thought = parsed_output.get("thought", "N/A") # Handle missing thought gracefully
        action_name = parsed_output.get("action")
        action_input = parsed_output.get("action_input", {}) # Default to empty dict if missing

        if not action_name:
            missing_action_error = "Formato ReAct inválido: Chave 'Action:' obrigatória está ausente ou mal formatada."
            agent_logger.error(f"{log_prefix} Parsing Failed: {missing_action_error}")
            observation_dict = {"status": "error", "action": "parsing_failed", "data": {"message": missing_action_error}}
            observation_str = f"Observation: {json.dumps(observation_dict)}"
            self._history.append(observation_str)
            # Call Reflector
            decision, _ = await agent_reflector.reflect_on_observation(
                 objective=objective, plan=plan, current_step_index=current_step_index,
                 action_name="_parse_llm", action_input={},
                 observation_dict=observation_dict,
                 history=self._history, memory=self._memory, agent_logger=agent_logger,
                 agent_instance=self
            )
            agent_logger.error(f"{log_prefix} Stopping plan due to missing 'Action:' (Reflector decision: {decision})")
            return True, f"Erro: {missing_action_error}"

        agent_logger.info(f"{log_prefix} Thought: {thought}")
        agent_logger.info(f"{log_prefix} Ação Decidida: {action_name}, Input: {action_input}")

        # 4. Executar Ação ou Finalizar e obter observation_dict
        observation_dict: Optional[Dict] = None
        observation_str: Optional[str] = None
        final_answer_text: Optional[str] = None

        if action_name == "final_answer":
            final_answer_text = action_input.get("answer", "Finalizado sem resposta específica.")
            agent_logger.info(f"{log_prefix} Ação Final: {final_answer_text}")
            # Create observation dict for reflector
            observation_dict = {"status": "success", "action": "final_answer", "data": {"answer": final_answer_text}}
            observation_str = f"Final Answer: {final_answer_text}" # History uses slightly different format

        elif action_name in self.tools:
            tool_result = await tool_executor.execute_tool(
                tool_name=action_name,
                action_input=action_input,
                tools_dict=self.tools,
                agent_logger=agent_logger
            )
            observation_dict = tool_result # Tool result is already a dict
            # Format observation string for history
            try:
                observation_str = f"Observation: {json.dumps(observation_dict, ensure_ascii=False)}"
            except TypeError as json_err:
                agent_logger.error(f"{log_prefix} Failed to serialize tool_result to JSON: {json_err}. Result: {tool_result}")
                observation_dict = {"status": "error", "action": "internal_error", "data": {"message": f"Failed to serialize tool result: {json_err}"}}
                observation_str = f"Observation: {json.dumps(observation_dict)}"
            # Save code to memory if the action was execute_code (this seems misplaced, should probably be done in reflector or executor)
            # if action_name == "execute_code":
            #     self._memory['last_code'] = action_input.get('code')
            
        else: # Tool not found
            agent_logger.warning(f"{log_prefix} Tool '{action_name}' not found.")
            observation_dict = {"status": "error", "action": "tool_not_found", "data": {"message": f"A ferramenta '{action_name}' não existe."}}
            observation_str = f"Observation: {json.dumps(observation_dict)}"
        
        # Append observation to history AFTER potential tool execution
        if observation_str: 
            self._history.append(observation_str)

        # 5. Reflect on Observation
        if observation_dict:
            agent_logger.info(f"{log_prefix} Calling Reflector (Duration: {(datetime.datetime.now() - start_cycle_time).total_seconds():.3f}s)")
            # <<< AWAIT >>>
            decision, new_plan = await agent_reflector.reflect_on_observation(
                objective=objective, # Overall objective
                plan=plan, 
                current_step_index=current_step_index,
                action_name=action_name, # Action attempted
                action_input=action_input,
                observation_dict=observation_dict,
                history=self._history, # History *includes* the latest observation
                memory=self._memory,
                agent_logger=agent_logger,
                agent_instance=self # <<< Pass agent instance >>>
            )
            agent_logger.info(f"[Reflector] Decision: {decision}")
        else:
            # Should not happen if logic above is correct
            agent_logger.error(f"{log_prefix} Observation dictionary was None, cannot reflect. Stopping.")
            decision = "stop_plan"
            new_plan = None

        # 6. Process Reflector Decision
        should_break_loop = False
        response_on_break = None

        if decision == "continue_plan":
            should_break_loop = False
        elif decision == "plan_complete":
            should_break_loop = True
            response_on_break = final_answer_text if final_answer_text is not None else observation_dict.get("data", {}).get("answer", "Plan Complete.")
        elif decision == "stop_plan":
            should_break_loop = True
            error_detail = observation_dict.get("data", {}).get("message", "Stopped by Reflector")
            response_on_break = f"Erro: Plano interrompido pelo Reflector. Detalhe: {error_detail}"
        elif decision == "replace_step_and_retry":
             agent_logger.warning(f"[Reflector] Decision '{decision}' requires plan modification - not fully implemented. Treating as stop_plan.")
             # TODO: Implement plan update logic here (update plan list, maybe reset index?)
             # self._current_plan = new_plan # Need to modify agent's plan directly?
             should_break_loop = True # Stop for now
             response_on_break = "Erro: Replanning/Retry não implementado."
        elif decision == "retry_step":
             agent_logger.warning(f"[Reflector] Decision '{decision}' not implemented. Treating as stop_plan.")
             should_break_loop = True # Stop for now
             response_on_break = "Erro: Retry Step não implementado."
        elif decision == "ask_user":
             agent_logger.warning(f"[Reflector] Decision '{decision}' not implemented. Treating as stop_plan.")
             should_break_loop = True # Stop for now
             response_on_break = "Erro: Ask User não implementado."
        else:
            agent_logger.error(f"{log_prefix} Decisão desconhecida do Reflector: '{decision}'. Parando por segurança.")
            should_break_loop = True
            response_on_break = f"Erro: Decisão desconhecida do Reflector: {decision}"

        # 7. Trim History (Still do this at the end of the cycle)
        self._history = history_manager.trim_history(self._history, MAX_HISTORY_TURNS, agent_logger)

        return should_break_loop, response_on_break

    # <<< REMOVED METHOD: _execute_tool >>>
    # <<< REMOVED METHOD: _trim_history >>>
