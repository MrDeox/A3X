import logging
from typing import List, Dict, Any, Optional
import json # Added json import for output formatting

# Core A3X Imports (adjust based on actual project structure)
try:
    from a3x.fragments.base import BaseFragment, FragmentContext # CORRECTED PATH
    # from a3x.decorators.fragment_decorator import fragment # INCORRECT PATH
    from a3x.fragments.registry import fragment # CORRECTED PATH
except ImportError as e:
    print(f"[EvolutionExecutorFragment] Warning: Could not import core A3X components ({e}). Using placeholders.")
    # Define placeholders if imports fail
    class FragmentContext:
        memory: Any = None
        tool_registry: Any = None
        llm: Any = None
        workspace_root: str = "."
        async def run_symbolic_command(self, command: str, **kwargs) -> Dict[str, Any]:
            print(f"Placeholder: Running symbolic command {command}")
            return {"status": "placeholder_success", "message": "Command executed symbolically."}

    class BaseFragment:
        def __init__(self, *args, **kwargs): pass
    # Placeholder for the decorator if the real one fails to import
    if 'fragment' not in locals():
        def fragment(*args, **kwargs):
            def decorator(cls): return cls
            return decorator

logger = logging.getLogger(__name__)

@fragment(name="evolution_executor", description="Executes planned evolution steps using appropriate skills.")
class EvolutionExecutorFragment(BaseFragment):
    """
    Executes a sequence of A3L directives proposed by the EvolutionPlannerFragment
    to drive autonomous system evolution.
    """

    async def execute(self, ctx: FragmentContext, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the automatic evolution cycle.

        Args:
            ctx: The fragment execution context.
            args: Optional dictionary (not used).

        Returns:
            A dictionary containing the status and a log of the execution results for each directive.
            e.g., {"status": "success", "execution_log": "✅ cmd1\nOutput...\n\n❌ cmd2\nError..."}
        """
        logger.info("--- Starting Evolution Executor Fragment --- ")

        # 1. Check for necessary context components
        if not hasattr(ctx, 'fragment_registry'):
            logger.error("FragmentRegistry not found in context (ctx.fragment_registry).")
            return {"status": "error", "message": "Internal configuration error: FragmentRegistry missing."}
        if not hasattr(ctx, 'run_symbolic_command'):
             logger.error("run_symbolic_command method not found in context (ctx.run_symbolic_command).")
             return {"status": "error", "message": "Internal configuration error: Context lacks run_symbolic_command method."}


        # 2. Get the evolution plan (A3L directives)
        try:
            logger.info("Requesting evolution plan...")
            planner_result = await ctx.fragment_registry.execute_fragment(
                "evolution_planner", {} # Use name or trigger
            )

            if not isinstance(planner_result, dict) or planner_result.get("status") != "success":
                error_msg = planner_result.get("message", "Planner failed or returned invalid format") if isinstance(planner_result, dict) else "Invalid response type from planner"
                logger.error(f"Failed to get evolution plan: {error_msg}")
                return {"status": "error", "message": f"Failed to get evolution plan: {error_msg}"}

            a3l_directives: List[str] = planner_result.get("a3l_directives", [])
            if not a3l_directives:
                 logger.info("Evolution planner returned no directives to execute.")
                 return {"status": "success", "execution_log": "No evolutionary directives were proposed."}

            logger.info(f"Received {len(a3l_directives)} directives from planner.")
            logger.debug(f"Directives to execute: {a3l_directives}")

        except Exception as e:
            logger.exception("Error executing EvolutionPlannerFragment:")
            return {"status": "error", "message": f"Error getting evolution plan: {e}"}

        # 3. Execute directives one by one, allowing for dynamic additions
        execution_results = []
        logger.info("Executing planned A3L directives (with potential dynamic additions)...")
        i = 0 # Use while loop for dynamic list modification
        while i < len(a3l_directives):
            cmd = a3l_directives[i]
            logger.info(f"Executing directive {i+1}/{len(a3l_directives)}: {cmd}")
            output = None # Initialize output
            try:
                # Assuming run_symbolic_command returns a dict 
                # (or needs modification to do so consistently)
                output = await ctx.run_symbolic_command(cmd)
                
                # Format output nicely (handle dict vs string)
                output_str = str(output)
                if isinstance(output, dict):
                     output_str = json.dumps(output, indent=2, ensure_ascii=False)

                logger.info(f"Directive {i+1} execution successful.")
                logger.debug(f"Output for '{cmd}':\n{output_str}")
                execution_results.append(f"✅ Directive {i+1}: {cmd}\n-- Output --\n{output_str}\n------------")

                # --- Robust Handling of Suggested Commands --- 
                if isinstance(output, dict) and "suggested_a3l_commands" in output:
                    suggested_commands = output["suggested_a3l_commands"]
                    if isinstance(suggested_commands, list) and suggested_commands:
                        logger.info(f"Received {len(suggested_commands)} suggested A3L commands from '{cmd}'. Adding to queue.")
                        # Insert the suggested commands immediately after the current one
                        for j, suggested_cmd in enumerate(reversed(suggested_commands)): # Insert in reverse to maintain order
                             if isinstance(suggested_cmd, str) and suggested_cmd.strip():
                                 a3l_directives.insert(i + 1, suggested_cmd)
                                 logger.debug(f"Inserted suggested command: {suggested_cmd}")
                             else:
                                 logger.warning(f"Skipping invalid suggested command: {suggested_cmd}")
                        logger.info(f"Plan updated. New length: {len(a3l_directives)}")
                # --- End Handling --- 

            except Exception as e:
                logger.exception(f"Error executing directive '{cmd}':")
                execution_results.append(f"❌ Directive {i+1}: {cmd}\n-- Error --\n{type(e).__name__}: {e}\n-----------")
                # Optional: Decide whether to stop execution on error
                # break # Uncomment to stop plan on first error
            
            i += 1 # Move to the next command

        # 4. Format and return the log
        final_log = "\n\n".join(execution_results)
        logger.info("Finished executing all planned directives.")
        logger.debug(f"Final Execution Log:\n{final_log}")

        return {"status": "success", "execution_log": final_log} 