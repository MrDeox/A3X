# Start of new file 
import logging
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import datetime

# Ensure skill decorator and utils are imported correctly
try:
    from a3x.core.skills import skill
    from a3x.core.context import Context
    # If called from within agent, we might need execute_tool
    # from a3x.core.skill_utils import execute_tool 
except ImportError:
    # Fallback for standalone testing
    def skill(**kwargs):
        def decorator(func):
            return func
        return decorator
    Context = Any
    # Dummy execute_tool if needed for testing
    # async def execute_tool(*args, **kwargs):
    #     print("[Dummy] execute_tool called for generalize_heuristics")
    #     return {"status": "success", "data": {"generated_rules_count": 1}}

logger = logging.getLogger(__name__)

# Constants
LEARNING_LOG_DIR = "memory/learning_logs"
HEURISTIC_LOG_FILE = os.path.join(LEARNING_LOG_DIR, "learned_heuristics.jsonl")
GENERALIZED_RULES_FILE = os.path.join(LEARNING_LOG_DIR, "generalized_rules.jsonl")
DEFAULT_GENERALIZATION_THRESHOLD = 10 # Trigger generalization after N new heuristics

def _get_timestamp(iso_timestamp_str: Optional[str]) -> Optional[datetime.datetime]:
    """Safely parses an ISO timestamp string into a datetime object."""
    if not iso_timestamp_str:
        return None
    try:
        # --- START: Modification for Robust Timestamp Parsing ---
        cleaned_timestamp_str = iso_timestamp_str
        # First, remove trailing 'Z' if it exists, regardless of offset
        if cleaned_timestamp_str.endswith('Z'):
            cleaned_timestamp_str = cleaned_timestamp_str[:-1]
        
        # Now, attempt parsing
        # Let fromisoformat handle standard offsets (+HH:MM or Z if we didn't strip it)
        # If it fails, log the warning.
        # --- END: Modification --- 

        dt_aware = datetime.datetime.fromisoformat(cleaned_timestamp_str)
        # If timezone info present, convert to UTC and make naive
        if dt_aware.tzinfo:
            dt_utc = dt_aware.astimezone(datetime.timezone.utc)
            return dt_utc.replace(tzinfo=None)
        else:
             # Assume naive timestamp is already UTC-like
             return dt_aware
    except ValueError:
        logger.warning(f"Could not parse timestamp: {iso_timestamp_str}")
        return None
    except Exception as e: # Catch broader errors like offset parsing
        logger.error(f"Unexpected error parsing timestamp '{iso_timestamp_str}': {e}")
        return None

async def _get_last_generalization_time(rules_file_path: Path) -> Optional[datetime.datetime]:
    """Finds the timestamp of the most recent entry in the generalized rules file."""
    last_time = None
    if not rules_file_path.exists():
        return None
    try:
        with open(rules_file_path, 'r', encoding='utf-8') as f:
            last_line = None
            for line in f: # Efficiently get the last line
                if line.strip(): # Ensure line is not empty
                    last_line = line
            if last_line:
                 try:
                    entry = json.loads(last_line.strip())
                    last_time = _get_timestamp(entry.get("timestamp"))
                 except json.JSONDecodeError:
                     logger.error(f"Failed to parse last line of generalized rules file: {last_line.strip()}")
    except Exception as e:
        logger.exception(f"Error reading last generalization time from {rules_file_path}: {e}")
    return last_time

async def _count_new_heuristics(heuristic_file_path: Path, since_time: Optional[datetime.datetime]) -> int:
    """Counts heuristics logged after a specific time."""
    count = 0
    if not heuristic_file_path.exists():
        return 0
    try:
        with open(heuristic_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                 if not line.strip(): continue # Skip empty lines
                 try:
                    entry = json.loads(line.strip())
                    entry_time = _get_timestamp(entry.get("timestamp"))
                    if entry_time:
                        # If no previous generalization, count all
                        # Compare naive UTC times
                        if since_time is None or entry_time > since_time:
                            # Count only original heuristics
                            if entry.get("type") in ["success", "failure"]:
                                count += 1
                 except json.JSONDecodeError:
                     logger.warning(f"Skipping invalid JSON line in heuristic log: {line.strip()}")
    except Exception as e:
        logger.exception(f"Error counting new heuristics in {heuristic_file_path}: {e}")
    return count

@skill(
    name="auto_generalize_heuristics",
    description="Verifica se novas heurísticas foram aprendidas e aciona a generalização se um limite for atingido.",
    parameters={
        "threshold": {"type": Optional[int], "description": "O limite de novas heurísticas para acionar a generalização (padrão: 10).", "default": DEFAULT_GENERALIZATION_THRESHOLD}
    }
)
async def auto_generalize_heuristics(ctx: Context, threshold: Optional[int] = None) -> Dict[str, Any]:
    """Checks if enough new heuristics exist and triggers the generalize_heuristics skill."""
    log_prefix = "[AutoGeneralize Skill]"
    threshold = threshold or DEFAULT_GENERALIZATION_THRESHOLD
    logger.info(f"{log_prefix} Verificando necessidade de generalização (limite: {threshold}).")

    generalization_triggered = False
    message = ""
    generalization_result_data = {}

    try:
        workspace_root = Path(getattr(ctx, 'workspace_root', '.'))
        rules_file = workspace_root / GENERALIZED_RULES_FILE
        heuristic_file = workspace_root / HEURISTIC_LOG_FILE
        
        # <<< CORREÇÃO: Garantir que o diretório exista ANTES de tentar ler/escrever >>>
        learning_dir = heuristic_file.parent
        try:
            learning_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"{log_prefix} Diretório de log garantido: {learning_dir}")
        except OSError as e:
            logger.error(f"{log_prefix} Falha ao criar diretório de log {learning_dir}: {e}")
            return {"status": "error", "data": {"message": f"Falha ao criar diretório de log: {e}"}}
        # <<< FIM CORREÇÃO >>>

        # 1. Get last generalization time
        last_gen_time = await _get_last_generalization_time(rules_file)
        if last_gen_time:
            logger.info(f"{log_prefix} Última generalização detectada em (naive UTC): {last_gen_time.isoformat()}")
        else:
            logger.info(f"{log_prefix} Nenhuma generalização anterior encontrada. Verificando todas as heurísticas.")

        # 2. Count new heuristics since then
        new_heuristic_count = await _count_new_heuristics(heuristic_file, last_gen_time)
        logger.info(f"{log_prefix} Novas heurísticas encontradas desde a última generalização: {new_heuristic_count}")

        # 3. Check threshold and trigger
        if new_heuristic_count >= threshold:
            logger.info(f"{log_prefix} Limite ({threshold}) atingido! Acionando generalize_heuristics...")
            try:
                # --- How to call generalize_heuristics? ---
                # Option A: Direct call (if running within the same agent process and imported)
                # from .generalize_heuristics import generalize_heuristics # Needs relative import adjustment
                # result = await generalize_heuristics(ctx=ctx)
                
                # Option B: Using execute_tool (Requires access to tools_dict and execute_tool in ctx)
                if hasattr(ctx, 'execute_tool') and hasattr(ctx, 'tools'):
                    logger.debug(f"{log_prefix} Calling generalize_heuristics via ctx.execute_tool")
                    # Pass necessary sub-context details if generalize_heuristics needs them
                    # For now, assuming generalize_heuristics primarily uses ctx for workspace and llm_url
                    result = await ctx.execute_tool(
                        tool_name="generalize_heuristics",
                        action_input={}, # Default top_n should be used by generalize_heuristics
                        tools_dict=ctx.tools,
                        context=ctx # Pass the agent context itself
                    )
                # Option C: Fallback if context lacks execute_tool (e.g., running standalone)
                elif 'generalize_heuristics' in globals(): # Check if imported directly
                    logger.warning(f"{log_prefix} Context lacks execute_tool. Attempting direct call to generalize_heuristics.")
                    # Need to import it if not already done
                    from a3x.skills.core.generalize_heuristics import generalize_heuristics
                    result = await generalize_heuristics(ctx=ctx) # Pass context if available
                else:
                     err_msg = "Cannot call generalize_heuristics: execute_tool/tools not in context and skill not directly imported."
                     logger.error(f"{log_prefix} {err_msg}")
                     raise RuntimeError(err_msg)
                # --- End Call Options ---

                if result.get("status") == "success":
                    generalization_result_data = result.get("data", {})
                    # Correct key based on generalize_heuristics output
                    action = result.get("action", "unknown")
                    if action == "generalization_generated":
                         gen_rule = generalization_result_data.get('general_rule', '(Regra não encontrada)')
                         message = f"Generalização acionada com sucesso. Nova regra: {gen_rule[:80]}..."
                         logger.info(f"{log_prefix} {message}")
                    else:
                         message = generalization_result_data.get("message", "Generalização executada, mas sem novas regras ou status desconhecido.")
                         logger.info(f"{log_prefix} {message} (Action: {action})")
                    generalization_triggered = True # Triggered even if no rule was generated this time
                else:
                    error_msg = result.get("data", {}).get("message", "Erro desconhecido")
                    message = f"Falha ao executar generalize_heuristics: {error_msg}"
                    logger.error(f"{log_prefix} {message}")
            except Exception as exec_e:
                logger.exception(f"{log_prefix} Erro ao chamar generalize_heuristics: {exec_e}")
                message = f"Erro ao executar skill de generalização: {exec_e}"
        else:
            message = f"Limite ({threshold}) não atingido ({new_heuristic_count} novas). Generalização não acionada."
            logger.info(f"{log_prefix} {message}")

        return {
            "status": "success",
            "data": {
                "message": message,
                "triggered_generalization": generalization_triggered,
                "new_heuristics_count": new_heuristic_count,
                "threshold": threshold,
                # Include results only if triggered and successful in generating
                "generalization_results": generalization_result_data if generalization_triggered and generalization_result_data.get('general_rule') else {}
            }
        }

    except Exception as e:
        logger.exception(f"{log_prefix} Erro inesperado durante a auto-generalização:")
        return {"status": "error", "data": {"message": f"Erro inesperado: {e}"}}

# Example Test Block
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # Requires dummy log files created previously

    # Dummy context for testing execute_tool call
    class DummyContext:
        workspace_root = '.'
        llm_url = None # Set if testing requires real LLM calls via generalize
        # Simulate tools dictionary if using execute_tool
        tools = {
             # "generalize_heuristics": generalize_heuristics # Assumes direct import works
        }
        async def execute_tool(self, tool_name, action_input, tools_dict, context):
            print(f"[DummyContext] Mock executing tool: {tool_name} with input {action_input}")
            if tool_name == "generalize_heuristics":
                # Simulate success by calling the actual function if possible
                try:
                     from a3x.skills.core.generalize_heuristics import generalize_heuristics
                     # Pass self as context to generalize_heuristics
                     return await generalize_heuristics(ctx=self, **action_input) 
                except ImportError:
                     return {"status": "error", "data": {"message": "generalize_heuristics not found for direct call in dummy context"}}
                except Exception as e:
                     return {"status": "error", "data": {"message": f"Error calling generalize_heuristics in dummy: {e}"}}
            return {"status": "error", "data": {"message": f"Tool '{tool_name}' not found in dummy context"}}

    async def run_main_test():
        print("\n--- Running Auto-Generalize Heuristics Test --- ")
        # Ensure log files exist for testing
        if not os.path.exists(LEARNING_LOG_DIR):
             os.makedirs(LEARNING_LOG_DIR)
        # Create dummy generalized file if needed (ensure it's empty or old for trigger)
        rule_file = Path(GENERALIZED_RULES_FILE)
        if rule_file.exists(): os.remove(rule_file) # Start fresh for test
        # with open(GENERALIZED_RULES_FILE, 'w', encoding='utf-8') as f: 
        #      # Write an old entry
        #      old_time = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat()
        #      f.write(json.dumps({"timestamp": old_time , "rule": "Dummy old rule"}) + '\n')
             
        # Add enough new heuristics to trigger generalization
        heuristic_file = Path(HEURISTIC_LOG_FILE)
        if heuristic_file.exists(): os.remove(heuristic_file) # Start fresh
        from a3x.core.learning_logs import log_heuristic_with_traceability
        base_time = datetime.datetime.utcnow()
        for i in range(DEFAULT_GENERALIZATION_THRESHOLD + 2):
            ts = (base_time + datetime.timedelta(seconds=i+1)).isoformat() + 'Z'
            heuristic = {
                "timestamp": ts,
                "type": "failure",
                "heuristic": f"New heuristic {i+1}",
                "context_snapshot": {}
            }
            # Gera plan_id/execution_id fictícios para teste
            plan_id = f"test-plan-{i+1}"
            execution_id = f"test-exec-{i+1}"
            log_heuristic_with_traceability(heuristic, plan_id, execution_id, validation_status="pending")

        dummy_ctx = DummyContext()
        result = await auto_generalize_heuristics(ctx=dummy_ctx)
        print("\n--- Auto-Generalization Result (Triggered) ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Test case: Threshold not met
        print("\n--- Running Auto-Generalize Heuristics Test (Threshold Not Met) --- ")
        # Use a high threshold
        result_not_met = await auto_generalize_heuristics(threshold=50, ctx=dummy_ctx)
        print("\n--- Auto-Generalization Result (Not Met) ---")
        print(json.dumps(result_not_met, indent=2, ensure_ascii=False))

    import asyncio
    asyncio.run(run_main_test()) 