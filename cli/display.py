import logging
import asyncio # Added asyncio
from typing import Optional # Added for type hinting compatibility if needed later

from rich.console import Console
from rich.panel import Panel

# Assuming project_root setup is handled elsewhere or core modules are in PYTHONPATH
try:
    from core.agent import ReactAgent # Needs ReactAgent type hint
    from core.llm_interface import call_llm # Added call_llm import
except ImportError:
    # Handle case where core modules might not be directly importable this way
    # This might require adjusting PYTHONPATH or how ReactAgent is passed
    ReactAgent = object # Placeholder if import fails, adjust as necessary
    call_llm = None # Placeholder if import fails

# Logger setup (consider passing logger instance or using a shared config)
logger = logging.getLogger(__name__) # Or get logger configured in interface.py
console = Console() # Assuming a global console instance is acceptable

async def handle_agent_interaction(agent: ReactAgent, command: str, conversation_history: list):
    """Gerencia a interação com o agente, exibindo passos intermediários com Rich."""
    logger.info(f"Processing command: '{command}'")
    console.print(Panel(f"[bold magenta]>[/bold magenta] {command}", title="User Input", border_style="magenta"))
    # Não adicionamos mais o input raw ao history aqui, o agente cuida disso internamente
    # conversation_history.append({"role": "user", "content": command})

    final_response = ""
    agent_outcome = None

    console.print("[yellow]A³X está pensando...[/]") # Feedback inicial
    try:
        # --- NOVO: Iterar sobre os passos do agente (run se torna um generator) ---
        async for step_output in agent.run(objective=command):
            if step_output["type"] == "thought":
                console.print(Panel(step_output['content'], title="🤔 Thought", border_style="yellow", title_align="left"))
            elif step_output["type"] == "action":
                action_name = step_output['tool_name']
                action_input = step_output['tool_input']
                console.print(Panel(f"""Tool: [bold cyan]{action_name}[/]
Input: {action_input}""", title="🎬 Action", border_style="cyan", title_align="left"))
            elif step_output["type"] == "observation":
                # Adicionar formatação baseada no status do resultado da tool
                obs_data = step_output['content']
                status = obs_data.get("status", "unknown")
                message = obs_data.get("data", {}).get("message", str(obs_data)) # Fallback para string

                border_style = "grey50"
                title = "👀 Observation"
                if status == "success":
                    border_style = "green"
                    title = "✅ Observation (Success)"
                elif status == "error":
                     border_style = "red"
                     title = "❌ Observation (Error)"
                elif status == "no_change":
                     border_style = "yellow"
                     title = "⚠️ Observation (No Change)"

                console.print(Panel(message, title=title, border_style=border_style, title_align="left"))
            elif step_output["type"] == "final_answer":
                final_response = step_output['content']
                console.print(Panel(final_response, title="🏁 Final Answer", border_style="bold green", title_align="left"))
                agent_outcome = {"status": "success", "action": "react_cycle_completed", "data": {"message": final_response}}
                break # Fim do ciclo
            else:
                 # Fallback para tipos desconhecidos
                 console.print(Panel(str(step_output), title="Unknown Step", border_style="grey50"))

        # Se o loop terminar sem final_answer (ex: max iterations)
        if not final_response:
             final_response = "Agent reached max iterations or finished without a final answer."
             console.print(Panel(final_response, title="🏁 Agent Finished", border_style="yellow"))
             agent_outcome = agent_outcome or {"status": "finished", "action": "react_cycle_ended", "data": {"message": final_response}}


    except Exception as e:
        logger.exception(f"Fatal Agent Error processing command '{command}':")
        final_response = f"Desculpe, ocorreu um erro interno crítico ao processar seu comando: {e}"
        agent_outcome = {"status": "error", "action": "react_cycle_failed", "data": {"message": str(e)}}
        console.print(Panel(f"""[bold red]Erro do Agente:[/]
{final_response}""", title="Agent Error", border_style="red"))

    # Não adicionamos assistant response ao history aqui, agente cuida disso
    # conversation_history.append({
    #     "role": "assistant",
    #     "content": final_response, # Armazena apenas a resposta final no histórico por simplicidade
    #     "agent_outcome": agent_outcome
    # })
    # ... (limitar histórico se necessário) ...

# Added stream_direct_llm function
async def stream_direct_llm(prompt: str):
    """Chama o LLM diretamente em modo streaming e exibe na console."""
    if not call_llm: # Check if import failed
        console.print("[bold red][Error][/] LLM interface not available.")
        return

    logger.info(f"Streaming direct prompt: '{prompt[:50]}...'")
    console.print(Panel("--- Streaming LLM Response ---", border_style="blue"))
    messages = [{"role": "user", "content": prompt}]
    try:
        # Assume call_llm handles initialization/connection implicitly now
        async for chunk in call_llm(messages, stream=True):
            console.print(chunk, end="")
        console.print() # Final newline
    except Exception as stream_err:
        logger.exception("Error during direct LLM stream:")
        console.print(f"\n[bold red][Error Streaming][/] {stream_err}")

