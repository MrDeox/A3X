# /home/arthur/Projects/A3X/cli/interface.py
import os
import sys
import asyncio
import argparse
import time
import logging
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel

# Adiciona o diretório raiz ao sys.path para encontrar 'core' e 'skills'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Imports do Core (após adicionar ao path)
try:
    from core.config import MAX_HISTORY_TURNS, LOG_LEVEL, MEMORY_DB_PATH # Assuming DB_FILE and AGENT_STATE_ID are not needed directly here
    from core.db_utils import initialize_database
    from core.agent import ReactAgent
    from core.llm_interface import call_llm # Import para stream_direct
except ImportError as e:
    print(f"[CLI Interface FATAL] Failed to import core modules: {e}. Ensure PYTHONPATH is correct or run from project root.")
    sys.exit(1)

# Configurar logging (poderia vir de uma config centralizada)
logging.basicConfig(level=LOG_LEVEL, format='[%(levelname)s CLI] %(message)s')
logger = logging.getLogger(__name__)

# Instância do Console Rich
console = Console()

# <<< REMOVED DEFAULT_SYSTEM_PROMPT DEFINITION >>>
# DEFAULT_SYSTEM_PROMPT = """..."""

# --- Helper Functions ---

def _load_system_prompt(file_path: str = "prompts/react_system_prompt.md") -> str:
    """Carrega o prompt do sistema de um arquivo."""
    full_path = os.path.join(project_root, file_path)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"System prompt file not found: {full_path}")
        console.print(f"[bold red][Error][/] System prompt file not found at '{full_path}'. Using a minimal fallback.")
        return "You are a helpful assistant."
    except Exception as e:
        logger.exception(f"Error reading system prompt file {full_path}:")
        console.print(f"[bold red][Error][/] Could not read system prompt file: {e}. Using minimal fallback.")
        return "You are a helpful assistant."

def _parse_arguments():
    """Parseia os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description='Assistente CLI A³X (ReAct)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--command', help='Comando único para executar')
    group.add_argument('-i', '--input-file', help='Arquivo para ler comandos sequencialmente (um por linha)')
    # Added interactive mode argument
    group.add_argument('--interactive', action='store_true', help='Inicia o modo interativo')
    parser.add_argument('--task', help='Tarefa única a ser executada pelo agente (prioridade sobre -c, -i, --interactive)')
    parser.add_argument("--stream-direct", help="Prompt direto para o LLM em modo streaming")
    return parser.parse_args()

def _initialize_agent(system_prompt: str) -> Optional[ReactAgent]:
    """Inicializa e retorna uma instância do ReactAgent."""
    logger.info("Initializing ReactAgent...")
    try:
        # Validação da URL do LLM removida (assumindo que não é mais necessária se não usar HTTP)
        # llm_api_url = LLAMA_SERVER_URL
        # if not llm_api_url or not llm_api_url.startswith("http"):
        #     raise ValueError(f"Invalid LLAMA_SERVER_URL found: {llm_api_url}")

        # Assume que ReactAgent não precisa mais de llm_url se for interagir diretamente
        # Ajustar a inicialização do ReactAgent conforme necessário
        agent = ReactAgent(system_prompt=system_prompt)
        logger.info("Agent ready.")
        return agent
    except Exception as agent_init_err:
        logger.exception("Fatal error initializing ReactAgent:")
        console.print(f"[bold red][Fatal Error][/] Could not initialize the agent: {agent_init_err}")
        return None

def change_to_project_root():
    """Garante que o script seja executado a partir do diretório raiz do projeto."""
    try:
        if os.getcwd() != project_root:
            os.chdir(project_root)
        logger.info(f"Running in directory: {os.getcwd()}")
    except FileNotFoundError:
        logger.error(f"Project root directory not found: {project_root}. Exiting.")
        console.print(f"[bold red][Fatal Error][/] Project directory not found: {project_root}")
        sys.exit(1)
    except Exception as cd_err:
        logger.error(f"Error changing directory to {project_root}: {cd_err}. Exiting.")
        console.print(f"[bold red][Fatal Error][/] Could not change to project directory: {cd_err}")
        sys.exit(1)

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

async def stream_direct_llm(prompt: str):
    """Chama o LLM diretamente em modo streaming."""
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

async def _run_interactive_mode(agent: ReactAgent):
    """Executa o loop de interação com o usuário."""
    console.print("[bold green]Modo Interativo A³X.[/] Digite 'sair', 'exit' ou 'quit' para terminar.")
    conversation_history = [] # Reset history for interactive session
    while True:
        try:
            command = console.input("[bold magenta]>[/bold magenta] ")
            if command.lower() in ['sair', 'exit', 'quit']:
                break
            if not command:
                continue
            await handle_agent_interaction(agent, command, conversation_history)
        except KeyboardInterrupt:
            console.print("\nSaindo...")
            break
        except EOFError: # Handle Ctrl+D
            console.print("\nSaindo...")
            break

async def _process_input_file(agent: ReactAgent, file_path: str):
    """Processa comandos de um arquivo de entrada."""
    logger.info(f"Reading commands from: {file_path}")
    conversation_history = [] # Reset history
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                command = line.strip()
                if command and not command.startswith('#'): # Ignora linhas vazias e comentários
                    await handle_agent_interaction(agent, command, conversation_history)
                    console.rule() # Separador entre comandos do arquivo
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        console.print(f"[bold red]Erro:[/bold red] Arquivo de entrada não encontrado: '{file_path}'")
    except Exception as file_err:
        logger.exception(f"Error processing input file {file_path}:")
        console.print(f"[bold red]Erro:[/bold red] Falha ao processar arquivo de entrada: {file_err}")

def run_cli():
    """Função principal que configura e executa a interface de linha de comando."""
    load_dotenv() # Carrega .env
    change_to_project_root() # Garante que estamos no diretório certo

    args = _parse_arguments()

    # conversation_history = [] # History is managed within interaction loops now

    # Inicializa DB
    logger.info("Initializing database...")
    initialize_database()

    # Stream direto não precisa do agente
    if args.stream_direct:
        asyncio.run(stream_direct_llm(args.stream_direct))
        return

    # Carrega prompt do sistema e inicializa o agente para os outros modos
    system_prompt = _load_system_prompt()
    agent = _initialize_agent(system_prompt)

    if not agent:
        logger.error("Agent initialization failed. Exiting.")
        sys.exit(1) # Sai se o agente não puder ser inicializado

    # Lógica de Execução Principal refatorada
    if args.task:
        logger.info(f"Executing single task: {args.task}")
        asyncio.run(handle_agent_interaction(agent, args.task, [])) # Pass empty history
    elif args.command:
        logger.info(f"Executing single command: {args.command}")
        asyncio.run(handle_agent_interaction(agent, args.command, [])) # Pass empty history
    elif args.input_file:
        asyncio.run(_process_input_file(agent, args.input_file))
    elif args.interactive:
        asyncio.run(_run_interactive_mode(agent))
    else:
        # Default to interactive mode if no other mode is specified
        asyncio.run(_run_interactive_mode(agent))

# Ponto de entrada do script
if __name__ == "__main__":
    run_cli()
