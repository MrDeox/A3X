import logging
from typing import Optional  # Added for type hinting compatibility if needed later
import json  # Import json for potentially pretty-printing action input
import time
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.console import Console

# Assuming project_root setup is handled elsewhere or core modules are in PYTHONPATH
try:
    from a3x.core.cerebrumx import CerebrumXAgent  # <<< ADD
    # REMOVED direct import of call_llm
    # ADDED necessary class imports
    from a3x.core.llm_interface import LLMInterface, DEFAULT_LLM_URL
except ImportError as e:
    # Handle potential import errors for core components if needed
    CerebrumXAgent = None
    LLMInterface = None
    DEFAULT_LLM_URL = "http://127.0.0.1:8080/completion" # Provide a fallback default
    print(f"[Warning] Failed to import core modules: {e}")

# Logger setup (consider passing logger instance or using a shared config)
logger = logging.getLogger(__name__)  # Or get logger configured in interface.py


async def handle_agent_interaction(
    agent: 'CerebrumXAgent', command: str, conversation_history: list
):  # <<< UPDATE TYPE HINT
    """Gerencia a interação com o agente CerebrumX, exibindo passos do ciclo cognitivo com print()."""  # <<< UPDATE DOCSTRING
    logger.info(f"Processing command with CerebrumX: '{command}'")  # <<< UPDATE LOG
    # Replace Panel with simple print
    print("--- User Input ---")
    print(f"> {command}")
    print("-" * 18)  # Separator

    # <<< ADDED: Instantiate Console here >>>
    console = Console()
    final_response = ""
    agent_outcome = None

    # Replace console.print with simple print
    print("CerebrumX está iniciando o ciclo cognitivo...")  # Feedback inicial updated
    try:
        # --- Iterar sobre os passos do ciclo CerebrumX ---
        # async for step_output in agent.run_cerebrumx_cycle(
        #     initial_perception=command
        # ):
        # <<< USE agent.run() instead, which returns the final result dictionary >>>
        final_result = await agent.run(objective=command)

        # <<< Process the final result dictionary >>>
        status = final_result.get("status", "unknown")
        message = final_result.get("message", "No message provided.")
        results = final_result.get("results", []) # This might contain detailed step outputs if added

        console.print(f"[bold green]--- Ciclo Concluído (Status: {status}) ---[/]")
        console.print(Panel(message, title="Mensagem Final", border_style="green"))
        if results:
            console.print(Panel(json.dumps(results, indent=2), title="Resultados Detalhados", border_style="blue"))

        # <<< Original loop logic removed as agent.run() is not a generator >>>
        # if isinstance(step_output, dict):
        #     step_type = step_output.get("type", "unknown")
        #     data = step_output.get("data", "")
        #     # ... (rest of the original step handling logic) ...
        # else:
        #     logger.warning(f"Received unexpected step output type: {type(step_output)}")
        #     console.print(Panel(str(step_output), title="Output Inesperado", border_style="yellow"))

        # Keep outcome structure consistent for potential external callers
        agent_outcome = {
            "status": "success",
            "action": "cerebrumx_cycle_completed",
            "data": {"message": message},
        }

    except Exception as e:
        logger.exception(
            f"Fatal CerebrumX Error processing command '{command}':"
        )  # <<< UPDATE LOG
        final_response = (
            f"Desculpe, ocorreu um erro interno crítico durante o ciclo CerebrumX: {e}"
        )
        agent_outcome = {
            "status": "error",
            "action": "cerebrumx_cycle_failed",
            "data": {"message": str(e)},
        }
        # Replace Panel with simple print
        print("--- Agent Error ---")
        print(f"Erro do Agente:\n{final_response}")
        print("-" * 17)  # Separator

    # History is managed by the agent itself (via memory)
    # No need to append here


# Added stream_direct_llm function
async def stream_direct_llm(prompt: str, llm_url_override: Optional[str] = None):
    """Chama o LLM diretamente em modo streaming e exibe com print()."""
    # Check if LLMInterface class import failed
    if not LLMInterface:
        print("[Error] LLM interface class not available due to import error.")
        return

    target_url = llm_url_override or DEFAULT_LLM_URL
    logger.info(
        f"Streaming direct prompt: '{prompt[:50]}...' to URL: {target_url}"
    )
    # Replace Panel with simple print
    print("--- Streaming LLM Response ---")
    messages = [{"role": "user", "content": prompt}]
    try:
        # Create an instance of LLMInterface
        llm_interface = LLMInterface(llm_url=target_url)
        # Call the method on the instance
        async for chunk in llm_interface.call_llm(messages=messages, stream=True):
            # Replace console.print with standard print
            print(chunk, end="")
        print()  # Final newline
    except Exception as stream_err:
        logger.exception("Error during direct LLM stream:")
        # No rich formatting
        print(f"\n[Error Streaming] {stream_err}")


async def _handle_task_argument(agent: 'CerebrumXAgent', task_arg: str):
    """Lida com a execução de uma tarefa única passada via --task."""
    console = Console()
    console.print(Panel(f"Executando Tarefa: [bold cyan]{task_arg}[/]", title="A³X Task Mode", expand=False))
    start_time = time.time()

    # Use a context manager for the progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True, # Remove progress bar on completion
    ) as progress:
        progress_task_id = progress.add_task("[cyan]Processando tarefa...", total=None) # Indeterminate

        try:
            # <<< CORRECTED: Call agent.run instead of agent.run_cerebrumx_cycle >>>
            # Passar a tarefa para o método run do agente
            # Como run não é mais um gerador, esperamos o resultado final
            final_result = await agent.run(task_arg)

            # Processar e exibir o resultado final
            end_time = time.time()
            duration = end_time - start_time
            progress.stop_task(progress_task_id)
            progress.update(progress_task_id, description="[green]Tarefa Concluída[/]")

            status = final_result.get("status", "unknown")
            message = final_result.get("message", "Nenhuma mensagem final.")
            results_list = final_result.get("results", []) # Lista de resultados dos passos

            if status == "completed":
                console.print(Panel(f"[bold green]Sucesso![/] ({duration:.2f}s)\n{message}", title="Resultado Final", border_style="green"))
            elif status == "failed":
                console.print(Panel(f"[bold yellow]Falha.[/] ({duration:.2f}s)\n{message}", title="Resultado Final", border_style="yellow"))
            elif status == "error":
                console.print(Panel(f"[bold red]Erro Crítico![/] ({duration:.2f}s)\n{message}", title="Resultado Final", border_style="red"))
            else:
                 console.print(Panel(f"[bold blue]Status Desconhecido: {status}[/] ({duration:.2f}s)\n{message}", title="Resultado Final", border_style="blue"))

            # Opcional: Exibir resultados detalhados dos passos
            # if results_list:
            #     console.print("\n--- Detalhes da Execução ---")
            #     for i, step_res in enumerate(results_list):
            #         console.print(f"Passo {i+1}: {step_res}")
            #     console.print("---------------------------")

        except Exception as e:
            progress.stop_task(progress_task_id)
            progress.update(progress_task_id, description="[red]Erro[/]")
            logger.exception(f"Erro fatal ao executar tarefa '{task_arg}'")
            console.print(Panel(f"[bold red]Erro Crítico Inesperado:[/] {e}", title="Erro Fatal", border_style="red"))
