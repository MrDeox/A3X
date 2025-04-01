import logging
import asyncio # Added asyncio
from typing import Optional # Added for type hinting compatibility if needed later
import json # Import json for potentially pretty-printing action input

# Assuming project_root setup is handled elsewhere or core modules are in PYTHONPATH
try:
    from core.cerebrumx import CerebrumXAgent # <<< ADD
    from core.llm_interface import call_llm # Added call_llm import
except ImportError:
    call_llm = None # Placeholder if import fails

# Logger setup (consider passing logger instance or using a shared config)
logger = logging.getLogger(__name__) # Or get logger configured in interface.py

async def handle_agent_interaction(agent: CerebrumXAgent, command: str, conversation_history: list): # <<< UPDATE TYPE HINT
    """Gerencia a interação com o agente CerebrumX, exibindo passos do ciclo cognitivo com print().""" # <<< UPDATE DOCSTRING
    logger.info(f"Processing command with CerebrumX: '{command}'") # <<< UPDATE LOG
    # Replace Panel with simple print
    print(f"--- User Input ---")
    print(f"> {command}")
    print("-" * 18) # Separator

    final_response = ""
    agent_outcome = None

    # Replace console.print with simple print
    print("CerebrumX está iniciando o ciclo cognitivo...") # Feedback inicial updated
    try:
        # --- Iterar sobre os passos do ciclo CerebrumX ---
        async for step_output in agent.run_cerebrumx_cycle(initial_perception=command): # <<< CALL run_cerebrumx_cycle
            step_type = step_output.get("type")
            step_content = step_output.get("content")

            # --- Handle CerebrumX Cycle Steps --- <<< NEW HANDLERS
            if step_type == "perception":
                print("--- 🧠 Perception ---")
                print(f"Processed: {json.dumps(step_content, indent=2, ensure_ascii=False)}")
                print("-" * 18)
            elif step_type == "context_retrieval":
                print("--- 💾 Context Retrieval ---")
                print(f"Retrieved: {json.dumps(step_content, indent=2, ensure_ascii=False)}")
                print("-" * 25)
            elif step_type == "planning":
                print("--- 🗺️ Planning ---")
                plan_steps = step_content if isinstance(step_content, list) else []
                if plan_steps:
                    for i, step in enumerate(plan_steps):
                        print(f"  {i+1}. {step}")
                else:
                    print("  (No plan generated or plan is empty)")
                print("-" * 18)
            elif step_type == "simulation":
                print("--- 🧪 Simulation ---")
                sim_outcomes = step_content if isinstance(step_content, list) else []
                if sim_outcomes:
                    for i, outcome in enumerate(sim_outcomes):
                         print(f"  Step {i+1}: {json.dumps(outcome, indent=4, ensure_ascii=False)}")
                else:
                     print("  (No simulation outcomes)")
                print("-" * 20)
            elif step_type == "execution_step":
                step_index = step_output.get("step_index", "N/A")
                step_result = step_output.get("result", {})
                status = step_result.get('status', 'unknown')
                title = f"--- ▶️ Execution Step {step_index + 1} ---"
                if status == "success":
                     title = f"--- ✅ Execution Step {step_index + 1} (Success) ---"
                elif status == "error":
                     title = f"--- ❌ Execution Step {step_index + 1} (Error) ---"

                print(title)
                print(f"Result: {json.dumps(step_result, indent=2, ensure_ascii=False)}")
                print("-" * len(title))
            elif step_type == "reflection":
                print("--- 🤔 Reflection ---")
                print(f"{json.dumps(step_content, indent=2, ensure_ascii=False)}")
                print("-" * 18)
            elif step_type == "learning_update":
                print("--- 🌱 Learning Update ---")
                print(f"Status: {step_output.get('status', 'unknown')}")
                print("-" * 23)
            # --- End CerebrumX Handlers ---

            # --- REMOVED OLD ReAct Handlers ---
            # elif step_output["type"] == "thought":
            # ... removed ...
            # elif step_output["type"] == "action":
            # ... removed ...
            # elif step_output["type"] == "observation":
            # ... removed ...
            # --- End REMOVED --- 

            elif step_type == "final_answer":
                final_response = step_content
                # Replace Panel with simple print
                print("--- 🏁 Final Answer ---")
                print(final_response)
                print("-" * 22) # Separator
                # Keep outcome structure consistent for potential external callers
                agent_outcome = {"status": "success", "action": "cerebrumx_cycle_completed", "data": {"message": final_response}}
                break # Fim do ciclo
            else:
                 # Fallback para tipos desconhecidos
                 print(f"--- Unknown Step Type: {step_type} ---")
                 print(str(step_output))
                 print("-" * 18) # Separator

        # Se o loop terminar sem final_answer (ex: max iterations or other reason)
        if not final_response:
             final_response = "CerebrumX cycle finished without a final answer."
             # Replace Panel with simple print
             print("--- 🏁 Cycle Finished ---")
             print(final_response)
             print("-" * 24) # Separator
             agent_outcome = agent_outcome or {"status": "finished", "action": "cerebrumx_cycle_ended", "data": {"message": final_response}}

    except Exception as e:
        logger.exception(f"Fatal CerebrumX Error processing command '{command}':") # <<< UPDATE LOG
        final_response = f"Desculpe, ocorreu um erro interno crítico durante o ciclo CerebrumX: {e}"
        agent_outcome = {"status": "error", "action": "cerebrumx_cycle_failed", "data": {"message": str(e)}}
        # Replace Panel with simple print
        print("--- Agent Error ---")
        print(f"Erro do Agente:\n{final_response}")
        print("-" * 17) # Separator

    # History is managed by the agent itself (via memory)
    # No need to append here

# Added stream_direct_llm function
async def stream_direct_llm(prompt: str, llm_url_override: Optional[str] = None):
    """Chama o LLM diretamente em modo streaming e exibe com print()."""
    if not call_llm: # Check if import failed
        print("[Error] LLM interface not available.") # No rich formatting
        return

    logger.info(f"Streaming direct prompt: '{prompt[:50]}...' to URL: {llm_url_override or 'default'}")
    # Replace Panel with simple print
    print("--- Streaming LLM Response ---")
    messages = [{"role": "user", "content": prompt}]
    try:
        # Assume call_llm handles initialization/connection implicitly now
        async for chunk in call_llm(messages, llm_url=llm_url_override, stream=True):
            # Replace console.print with standard print
            print(chunk, end="")
        print() # Final newline
    except Exception as stream_err:
        logger.exception("Error during direct LLM stream:")
        # No rich formatting
        print(f"\n[Error Streaming] {stream_err}")

