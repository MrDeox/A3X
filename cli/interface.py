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
    from core.config import MAX_HISTORY_TURNS, LLAMA_SERVER_URL, LOG_LEVEL, DB_FILE, AGENT_STATE_ID
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

# <<< DEFINIR PROMPT PADRÃO AQUI >>>
# Idealmente, carregar de um arquivo de configuração ou .md
DEFAULT_SYSTEM_PROMPT = """Você é um agente autônomo chamado A³X que segue o framework ReAct para atingir objetivos complexos.

Seu ciclo é: pensar, agir, observar.

⚠️ **FORMATO OBRIGATÓRIO** DE RESPOSTA:

Sempre responda neste exato formato, e somente nele:

Thought: <raciocínio sobre o próximo passo>
Action: <nome_da_ferramenta_disponível>
Action Input: <objeto JSON com os parâmetros da ferramenta>

✅ Exemplo para ler um arquivo:

Thought: Para ler o conteúdo do arquivo solicitado, devo usar a ferramenta 'read_file'.
Action: read_file
Action Input: {"file_name": "caminho/do/arquivo.txt"}

✅ Exemplo para listar arquivos:

Thought: Preciso ver os arquivos no diretório 'src'. Usarei a ferramenta 'list_files'.
Action: list_files
Action Input: {"directory": "src"}

Nunca explique o que está fazendo fora do bloco "Thought:". Nunca adicione justificativas ou mensagens fora do formato.

Se não for possível agir ou a tarefa estiver concluída, retorne uma Action chamada 'final_answer' com a resposta final no campo 'answer'.

Esse formato será interpretado por outro sistema e precisa estar 100% correto.

## REGRAS ABSOLUTAS E FERRAMENTAS DISPONÍVEIS:

1.  **USE APENAS AS SEGUINTES FERRAMENTAS:**
    *   `list_files`: Lista nomes de arquivos/diretórios (parâmetro opcional: `directory`).
    *   `read_file`: Lê o conteúdo de um arquivo de texto (parâmetro: `file_name` ou `file_path`).
    *   `create_file`: Cria/sobrescreve um arquivo (parâmetros: `action='create'`, `file_name`, `content`).
    *   `append_to_file`: Adiciona ao final de um arquivo (parâmetros: `action='append'`, `file_name`, `content`).
    *   `delete_file`: Deleta um arquivo (parâmetros: `file_path`, `confirm=True`).
    *   `execute_code`: Executa código Python (parâmetro: `code`).
    *   `modify_code`: Modifica código existente (parâmetros: `modification`, `code_to_modify`).
    *   `generate_code`: Gera novo código (parâmetros: `description`).
    *   `text_to_speech`: Converte texto em fala (parâmetros: `text`, `voice_model_path`, opcional `output_dir`, `filename`).
    *   `final_answer`: Finaliza e dá a resposta (parâmetro: `answer`).

2.  **NUNCA INVENTE FERRAMENTAS:** Não use `ls`, `cd`, `cat`, `env`, `analyze_config`, `search_web` (desativada) ou qualquer outra ferramenta que não esteja EXPLICITAMENTE listada acima.
3.  **SEJA LITERAL:** Use os nomes exatos das ferramentas e seus parâmetros conforme listado.
4.  **PENSE PASSO A PASSO:** No bloco `Thought:`, explique seu raciocínio para escolher a próxima ferramenta e seus parâmetros.

"""

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
    """Gerencia a interação com o agente, exibindo passos intermediários com Rich (A SER IMPLEMENTADO)."""
    logger.info(f"Processing command: '{command}'")
    console.print(Panel(f"[bold magenta]>[/bold magenta] {command}", title="User Input", border_style="magenta"))
    conversation_history.append({"role": "user", "content": command})

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

    conversation_history.append({
        "role": "assistant",
        "content": final_response, # Armazena apenas a resposta final no histórico por simplicidade
        "agent_outcome": agent_outcome
    })
    # ... (limitar histórico se necessário) ...

async def stream_direct_llm(prompt: str):
    """Chama o LLM diretamente em modo streaming."""
    logger.info(f"Streaming direct prompt: '{prompt[:50]}...'")
    console.print(Panel("--- Streaming LLM Response ---", border_style="blue"))
    messages = [{"role": "user", "content": prompt}]
    try:
        async for chunk in call_llm(messages, stream=True):
            console.print(chunk, end="")
        console.print() # Final newline
    except Exception as stream_err:
        logger.exception("Error during direct LLM stream:")
        console.print(f"\n[bold red][Error Streaming][/] {stream_err}")

def run_cli():
    """Função principal que configura e executa a interface de linha de comando."""
    load_dotenv() # Carrega .env
    change_to_project_root() # Garante que estamos no diretório certo

    parser = argparse.ArgumentParser(description='Assistente CLI A³X (ReAct)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--command', help='Comando único para executar')
    group.add_argument('-i', '--input-file', help='Arquivo para ler comandos sequencialmente (um por linha)')
    parser.add_argument('--task', help='Tarefa única a ser executada pelo agente (prioridade sobre -c e -i)')
    parser.add_argument("--stream-direct", help="Prompt direto para o LLM em modo streaming")

    args = parser.parse_args()

    conversation_history = []

    # Inicializa DB
    logger.info("Initializing database...")
    initialize_database()

    # Instanciação do Agente (apenas se necessário)
    agent = None
    if not args.stream_direct:
        logger.info("Initializing ReactAgent...")
        try:
            # Valida URL do LLM
            llm_api_url = LLAMA_SERVER_URL
            if not llm_api_url or not llm_api_url.startswith("http"):
                raise ValueError(f"Invalid LLAMA_SERVER_URL found: {llm_api_url}")

            # TODO: Carregar o prompt de um arquivo ou config central

            agent = ReactAgent(llm_url=llm_api_url, system_prompt=DEFAULT_SYSTEM_PROMPT)
            logger.info("Agent ready.")
        except Exception as agent_init_err:
             logger.exception("Fatal error initializing ReactAgent:")
             console.print(f"[bold red][Fatal Error][/] Could not initialize the agent: {agent_init_err}")
             exit(1)

    # Lógica de Execução Principal
    if args.stream_direct:
        asyncio.run(stream_direct_llm(args.stream_direct)) # No agent, no LD_PATH needed

    elif args.task:
        logger.info(f"Executing single task: {args.task}")
        if agent:
            asyncio.run(handle_agent_interaction(agent, args.task, conversation_history))
        else:
            logger.error("Agent not initialized, cannot run task.")
            console.print("[bold red]Erro:[/bold red] Agente não inicializado.")

    elif args.command:
        logger.info(f"Executing single command: {args.command}")
        if agent:
            asyncio.run(handle_agent_interaction(agent, args.command, conversation_history))
        else:
             logger.error("Agent not initialized, cannot run task.")
             console.print("[bold red]Erro:[/bold red] Agente não inicializado.")

    elif args.input_file:
        if not agent:
            logger.error("Agent not initialized, cannot process input file.")
            console.print("[bold red]Erro:[/bold red] Agente não inicializado.")
            return # Sai da função run_cli

        try:
            logger.info(f"Reading commands from: {args.input_file}")
            with open(args.input_file, 'r', encoding='utf-8') as f:
                commands = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            logger.info(f"Found {len(commands)} commands to process.")
            console.print(Panel(f"Processing [bold]{len(commands)}[/] commands from '{args.input_file}'", border_style="blue"))

            for line_num, command in enumerate(commands, 1):
                logger.info(f"--- Processing command from line {line_num} ---")
                console.print(f"\n[cyan]--- Command from Line {line_num} ---[/]")
                asyncio.run(handle_agent_interaction(agent, command, conversation_history))
                time.sleep(1) # Pausa opcional
            logger.info("Finished processing input file.")
            console.print("\n[blue][Info][/] End of input file.")
        except FileNotFoundError:
            logger.error(f"Input file not found: {args.input_file}")
            console.print(f"[bold red][Fatal Error][/] Input file not found: {args.input_file}")
        except Exception as file_proc_err:
            logger.exception(f"Error processing input file '{args.input_file}':")
            console.print(f"\n[bold red][Fatal Error][/] An error occurred while processing the file '{args.input_file}': {file_proc_err}")

    else:
        # Modo interativo
        if not agent:
             logger.error("Agent not initialized, cannot start interactive mode.")
             console.print("[bold red]Erro:[/bold red] Agente não inicializado.")
             return # Sai da função run_cli

        console.print(Panel("[bold green]Assistente CLI A³X (ReAct)[/] iniciado. Digite '[cyan]sair[/]' para encerrar.", title="Welcome", border_style="green"))
        while True:
            try:
                command = console.input("\n[bold green]>[/] ").strip()
                if command.lower() in ['sair', 'exit', 'quit']:
                    logger.info("Exit command received. Shutting down.")
                    console.print("[yellow]Encerrando assistente...[/]")
                    break
                if not command:
                    continue
                asyncio.run(handle_agent_interaction(agent, command, conversation_history))
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Shutting down.")
                console.print("\n[yellow]Encerrando assistente...[/]")
                break
            except Exception as loop_err:
                logger.exception("Unexpected error in main interactive loop:")
                console.print(f"\n[bold red][Unexpected Error][/] {loop_err}")

# Note: O ponto de entrada principal (`if __name__ == "__main__":`) ficará em assistant_cli.py
