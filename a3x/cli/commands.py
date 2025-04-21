# a3x/cli/commands.py
import asyncio
import logging
import json
import yaml
import os
from typing import Optional, Tuple, AsyncGenerator, Set, Dict, Any, Callable
from pathlib import Path
from collections import namedtuple

from rich.console import Console

# Use relative imports for modules within the same package (cli)
from .display import handle_agent_interaction, stream_direct_llm
from .parsing import parse_skill_arguments
from .llm_utils import create_skill_execution_context

# Import core components needed by handlers
# Need to figure out absolute/relative path for these
# Assuming execution context allows finding 'a3x' package
try:
    from a3x.core.cerebrumx import CerebrumXAgent
    from a3x.core.skills import get_skill, SKILL_REGISTRY
    from a3x.core.tool_executor import execute_tool
    from a3x.core.llm_interface import LLMInterface
    from a3x.training.trainer import run_qlora_finetuning
    from a3x.core.config import PROJECT_ROOT
except ImportError as e:
    # Fallback or error logging if imports fail
    print(f"[CLI Commands Error] Failed to import core modules: {e}")
    CerebrumXAgent = None # Define as None to avoid runtime errors later
    get_skill = None
    SKILL_REGISTRY = {}
    execute_tool = None
    LLMInterface = None
    run_qlora_finetuning = None
    PROJECT_ROOT = "."

logger = logging.getLogger(__name__)
console = Console()


async def run_interactive_mode(agent: CerebrumXAgent):
    """Runs the agent in interactive mode."""
    if not agent:
        console.print("[bold red]Error:[/bold red] Agent not initialized.")
        return
    console.print(
        "[bold green]Entering interactive mode.[/bold green] Type 'exit' or 'quit' to leave."
    )
    while True:
        try:
            user_input = await asyncio.to_thread(console.input, "[bold cyan]You:[/bold cyan] ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue

            # Pass the agent, user input, and the console print function
            await handle_agent_interaction(agent, user_input, console.print)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Interrupted. Exiting interactive mode.[/bold yellow]")
            break
        except Exception as e:
            logger.exception("Error during interactive loop:")
            console.print(f"[bold red]An error occurred:[/bold red] {e}")

async def run_from_file(agent: CerebrumXAgent, file_path: str):
    """Processes commands sequentially from an input file."""
    if not agent:
        console.print("[bold red]Error:[/bold red] Agent not initialized.")
        return
    console.print(f"[bold green]Processing commands from file:[/bold green] {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                command = line.strip()
                if not command or command.startswith("#"):
                    continue
                console.print(f"[bold cyan]Executing (from file):[/bold cyan] {command}")
                # Pass the agent, command, and the console print function
                await handle_agent_interaction(agent, command, console.print)
                console.print("---") # Separator between commands
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        console.print(f"[bold red]Error:[/bold red] Input file not found: {file_path}")
    except Exception as e:
        logger.exception(f"Error processing input file {file_path}:")
        console.print(f"[bold red]An error occurred processing the file:[/bold red] {e}")

async def run_task(agent: CerebrumXAgent, task_arg: str):
    """Handles the execution of a single task provided via --task argument."""
    if not agent:
        console.print("[bold red]Error:[/bold red] Agent not initialized.")
        return
    console.print(f"[bold green]Executing task:[/bold green] {task_arg}")
    # Pass the agent, task, and the console print function
    await handle_agent_interaction(agent, task_arg, console.print)


async def run_single_command(agent: CerebrumXAgent, command_arg: str):
    """Handles the execution of a single command provided via -c argument."""
    if not agent:
        console.print("[bold red]Error:[/bold red] Agent not initialized.")
        return
    console.print(f"[bold green]Executing command:[/bold green] {command_arg}")
    await handle_agent_interaction(agent, command_arg, console.print)

async def run_stream_direct(prompt: str, llm_url_override: Optional[str] = None):
    """Handles the --stream-direct argument, streaming LLM output."""
    console.print(f"[bold green]Streaming direct response for:[/bold green] {prompt}")
    # Use the streaming wrapper from the display module
    # Need to instantiate LLMInterface here or pass it in
    llm_interface = LLMInterface(base_url=llm_url_override) # Use override if provided
    await stream_direct_llm(llm_interface, prompt, console.print)


async def run_training_cycle():
    """Handles the --train argument."""
    if run_qlora_finetuning:
        console.print("[bold green]Starting QLoRA fine-tuning cycle...[/bold green]")
        try:
            # Consider making run_qlora_finetuning async or running sync in thread
            # await asyncio.to_thread(run_qlora_finetuning) # Example if sync
            # If it's already async:
            # await run_qlora_finetuning()
            # For now, assuming it's synchronous for simplicity
            console.print(
                "[bold yellow]Warning:[/bold yellow] Running synchronous training function. CLI might block."
            )
            run_qlora_finetuning() # Placeholder call
            console.print("[bold green]Fine-tuning cycle completed.[/bold green]")
        except Exception as e:
            logger.exception("Error during fine-tuning cycle:")
            console.print(f"[bold red]Fine-tuning failed:[/bold red] {e}")
    else:
        console.print(
            "[bold red]Error:[/bold red] Training module not available. Cannot run --train."
        )

async def run_skill_directly(args, llm_url_override: Optional[str]):
    """Handles the --run-skill argument for direct skill execution."""
    skill_name = args.run_skill
    console.print(f"[bold green]Attempting to run skill directly:[/bold green] {skill_name}")

    if not get_skill or not SKILL_REGISTRY:
        console.print("[bold red]Error:[/bold red] Skill registry not available.")
        return

    skill_func = get_skill(skill_name)
    if not skill_func:
        console.print(f"[bold red]Error:[/bold red] Skill '{skill_name}' not found in registry.")
        console.print(f"Available skills: {list(SKILL_REGISTRY.keys())}")
        return

    skill_args = parse_skill_arguments(args)
    if skill_args is None:
        console.print("[bold red]Failed to parse skill arguments. Aborting skill execution.[/bold red]")
        return

    console.print(f"Running '{skill_name}' with args: {skill_args}")

    try:
        ctx = create_skill_execution_context(llm_url_override, logger)
        if not ctx:
            console.print("[bold red]Error:[/bold red] Failed to create skill execution context.")
            return

        if asyncio.iscoroutinefunction(skill_func):
            result = await skill_func(ctx=ctx, **skill_args)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: skill_func(ctx=ctx, **skill_args))

        console.print("[bold green]Skill Result:[/bold green]")
        try:
            console.print(json.dumps(result, indent=2, default=str))
        except TypeError:
             console.print(str(result))

    except TypeError as te:
         logger.exception(f"TypeError executing skill '{skill_name}': Likely incorrect arguments provided. {te}")
         console.print(f"[bold red]Error:[/bold red] TypeError executing skill '{skill_name}'. Check arguments provided ({skill_args}). Error: {te}")
         try:
             import inspect
             sig = inspect.signature(skill_func)
             console.print(f"Expected signature: {skill_name}{sig}")
         except (ValueError, ImportError):
              pass
    except Exception as e:
        logger.exception(f"Error running skill '{skill_name}':")
        console.print(f"[bold red]An error occurred:[/bold red] {e}") 