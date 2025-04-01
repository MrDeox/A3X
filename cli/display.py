import logging
import asyncio # Added asyncio
from typing import Optional # Added for type hinting compatibility if needed later
import json # Import json for potentially pretty-printing action input

# Assuming project_root setup is handled elsewhere or core modules are in PYTHONPATH
try:
    from core.agent import ReactAgent # Needs ReactAgent type hint
    from core.llm_interface import call_llm # Added call_llm import
except ImportError:
    call_llm = None # Placeholder if import fails

# Logger setup (consider passing logger instance or using a shared config)
logger = logging.getLogger(__name__) # Or get logger configured in interface.py

async def handle_agent_interaction(agent: ReactAgent, command: str, conversation_history: list):
    """Gerencia a interação com o agente, exibindo passos intermediários com print()."""
    logger.info(f"Processing command: '{command}'")
    # Replace Panel with simple print
    print(f"--- User Input ---")
    print(f"> {command}")
    print("-" * 18) # Separator

    final_response = ""
    agent_outcome = None

    # Replace console.print with simple print
    print("A³X está pensando...") # Feedback inicial
    try:
        # --- Iterar sobre os passos do agente (run se torna um generator) ---
        async for step_output in agent.run(objective=command):
            if step_output["type"] == "thought":
                # Replace Panel with simple print
                print("--- 🤔 Thought ---")
                print(step_output['content'])
                print("-" * 18) # Separator
            elif step_output["type"] == "action":
                action_name = step_output['tool_name']
                action_input_raw = step_output['tool_input']
                # Attempt to pretty-print JSON if possible
                try:
                    action_input_formatted = json.dumps(action_input_raw, indent=2, ensure_ascii=False)
                except (TypeError, ValueError):
                    action_input_formatted = str(action_input_raw) # Fallback to string
                # Replace Panel with simple print
                print("--- 🎬 Action ---")
                print(f"Tool: {action_name}")
                print(f"Input:\n{action_input_formatted}")
                print("-" * 18) # Separator
            elif step_output["type"] == "observation":
                obs_data = step_output['content']
                status = obs_data.get("status", "unknown")
                raw_message = obs_data.get("data", {}).get("message", str(obs_data))
                message_to_print = str(raw_message)

                title = "--- 👀 Observation ---"
                if status == "success":
                    title = "--- ✅ Observation (Success) ---"
                elif status == "error":
                     title = "--- ❌ Observation (Error) ---"
                elif status == "no_change":
                     title = "--- ⚠️ Observation (No Change) ---"

                # Replace Panel with simple print
                print(title)
                print(message_to_print)
                print("-" * len(title)) # Separator matching title length

            elif step_output["type"] == "final_answer":
                final_response = step_output['content']
                # Replace Panel with simple print
                print("--- 🏁 Final Answer ---")
                print(final_response)
                print("-" * 22) # Separator
                agent_outcome = {"status": "success", "action": "react_cycle_completed", "data": {"message": final_response}}
                break # Fim do ciclo
            else:
                 # Fallback para tipos desconhecidos
                 print("--- Unknown Step ---")
                 print(str(step_output))
                 print("-" * 18) # Separator

        # Se o loop terminar sem final_answer (ex: max iterations)
        if not final_response:
             final_response = "Agent reached max iterations or finished without a final answer."
             # Replace Panel with simple print
             print("--- 🏁 Agent Finished ---")
             print(final_response)
             print("-" * 24) # Separator
             agent_outcome = agent_outcome or {"status": "finished", "action": "react_cycle_ended", "data": {"message": final_response}}


    except Exception as e:
        logger.exception(f"Fatal Agent Error processing command '{command}':")
        final_response = f"Desculpe, ocorreu um erro interno crítico ao processar seu comando: {e}"
        agent_outcome = {"status": "error", "action": "react_cycle_failed", "data": {"message": str(e)}}
        # Replace Panel with simple print
        print("--- Agent Error ---")
        print(f"Erro do Agente:\n{final_response}")
        print("-" * 17) # Separator

    # Não adicionamos assistant response ao history aqui, agente cuida disso
    # conversation_history.append({
    #     "role": "assistant",
    #     "content": final_response, # Armazena apenas a resposta final no histórico por simplicidade
    #     "agent_outcome": agent_outcome
    # })
    # ... (limitar histórico se necessário) ...

# Added stream_direct_llm function
async def stream_direct_llm(prompt: str):
    """Chama o LLM diretamente em modo streaming e exibe com print()."""
    if not call_llm: # Check if import failed
        print("[Error] LLM interface not available.") # No rich formatting
        return

    logger.info(f"Streaming direct prompt: '{prompt[:50]}...'")
    # Replace Panel with simple print
    print("--- Streaming LLM Response ---")
    messages = [{"role": "user", "content": prompt}]
    try:
        # Assume call_llm handles initialization/connection implicitly now
        async for chunk in call_llm(messages, stream=True):
            # Replace console.print with standard print
            print(chunk, end="")
        print() # Final newline
    except Exception as stream_err:
        logger.exception("Error during direct LLM stream:")
        # No rich formatting
        print(f"\n[Error Streaming] {stream_err}")

