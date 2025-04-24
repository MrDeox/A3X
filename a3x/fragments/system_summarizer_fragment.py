import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path
# from a3x.core.fragment import fragment, Fragment # Old import
# from a3x.core.context import Context # Old import
from a3x.fragments.base import BaseFragment, FragmentContext # Corrected import
from a3x.fragments.registry import fragment # CORRECTED PATH
import logging

logger = logging.getLogger(__name__)

# Core A3X Imports (adjust based on actual project structure)
try:
    # from a3x.core.fragment import BaseFragment, FragmentContext # OLD INCORRECT PATH
    from a3x.fragments.base import BaseFragment, FragmentContext # CORRECTED PATH
    from a3x.core.memory.memory_manager import MemoryManager
    # Assuming decorator is needed and path is correct:
    # from a3x.decorators.fragment_decorator import fragment # INCORRECT PATH
    from a3x.fragments.registry import fragment # CORRECTED PATH
except ImportError as e:
    print(f"[SystemSummarizerFragment] Warning: Could not import core A3X components ({e}). Using placeholders.")
    # Define placeholders if imports fail
    class FragmentContext:
        memory: Optional['MemoryManager'] = None
        workspace_root: Optional[str] = None
        llm: Any = None # Added placeholder for LLM if needed
    class BaseFragment:
        def __init__(self, *args, **kwargs): pass
    class MemoryManager:
        async def get_recent_episodes(self, limit: int) -> List[Dict[str, Any]]: return []
    # Placeholder for the decorator if the real one fails to import
    if 'fragment' not in locals():
        def fragment(*args, **kwargs):
            def decorator(cls): return cls
            return decorator

# Ensure only valid arguments (name, description, category, skills, managed_skills) are used
@fragment(name="system_summarizer", description="Summarizes the system state and suggests evolution steps.")
class SystemSummarizerFragment(BaseFragment): # Changed Fragment to BaseFragment
    """
    Generates a symbolic summary of the A³X system's state, combines it with an
    evolutionary goal, prompts an LLM for suggestions, and logs the interaction.
    """

    def __init__(self, context: FragmentContext): # Changed Context to FragmentContext
        super().__init__(context)
        # Assuming context now directly provides a3x_home, adjust if needed
        self.data_dir = os.path.join(self.ctx.a3x_home, "a3net", "data")
        self.suggestions_file = os.path.join(self.data_dir, "evolution_suggestions.jsonl")
        self.model_file = os.path.join(self.data_dir, "fragment_success_predictor.pkl")
        self.jsonl_files = [
            "symbolic_experience.jsonl",
            "evaluation_summary.jsonl",
            "mutation_history.jsonl",
            "meta_insights.jsonl",
        ]

    def _count_lines(self, filepath: str) -> int | None:
        """Counts lines in a file, returns None if file not found."""
        try:
            with open(filepath, 'r') as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error counting lines in {filepath}: {e}")
            return None

    def _generate_summary(self) -> str:
        """Generates the symbolic summary of the system state."""
        summary_parts = []

        # 1. Registered Fragments
        try:
            # Replace self.ctx.fragment_registry with correct access if context structure changed
            # registered_fragments = self.ctx.fragment_registry.list_all()
            # Assuming FragmentContext provides fragment_registry directly or via ctx
            registered_fragments = self.ctx.fragment_registry.list_all() # Try direct access from context
            if registered_fragments:
                summary_parts.append("# Fragmentos Registrados:")
                for frag_name in sorted(registered_fragments):
                    summary_parts.append(f"- {frag_name}")
            else:
                summary_parts.append("# Fragmentos Registrados:\n- Nenhum fragmento registrado.")
        except Exception as e:
            logger.error(f"Error listing registered fragments: {e}")
            summary_parts.append("# Fragmentos Registrados:\n- Erro ao listar fragmentos.")

        summary_parts.append("") # Add a blank line

        # 2. Data Files
        summary_parts.append("# Dados Existentes:")
        data_found = False
        os.makedirs(self.data_dir, exist_ok=True) # Ensure data dir exists

        for filename in self.jsonl_files:
            filepath = os.path.join(self.data_dir, filename)
            line_count = self._count_lines(filepath)
            if line_count is not None:
                summary_parts.append(f"- {filename} ({line_count} linhas)")
                data_found = True
            else:
                summary_parts.append(f"- {filename} (ausente)")

        if not data_found and not os.path.exists(self.model_file):
             summary_parts.append("- Nenhum arquivo de dados encontrado.")

        summary_parts.append("") # Add a blank line

        # 3. Model File
        summary_parts.append("# Modelo:")
        model_exists = os.path.exists(self.model_file)
        status = 'presente' if model_exists else 'ausente'
        summary_parts.append(f"- fragment_success_predictor.pkl ({status})")

        return "\n".join(summary_parts)

    async def execute(self, ctx: FragmentContext, args: dict | None = None) -> str | dict: # Changed execute signature
        """Executes the system summarization and suggestion generation process."""
        logger.info("Generating system summary and evolution suggestions...")

        # Ensure using the correct context variable (ctx)
        summary = self._generate_summary() # _generate_summary might need ctx passed if it uses it
        logger.debug(f"""Generated Summary:
{summary}""")

        # 2. Define Fixed Objective
        evolution_objective = (
            "Objetivo: Transformar o A³X em um sistema autônomo que evolui sua cognição, "
            "estrutura simbólica e rede neural continuamente, criando novos componentes, "
            "otimizando os antigos e adaptando sua linguagem própria."
        )

        # 3. Construct Prompt
        prompt = f"""Você é um especialista em evolução de sistemas cognitivos autônomos.

Seu objetivo é ajudar um sistema chamado A³X a evoluir sozinho.

Ele já possui:
{summary}

{evolution_objective}

Com base nisso, sugira:
- Um novo fragmento a ser criado (descreva sua função)
- Um possível comando A3L para gerar esse fragmento (use a sintaxe 'create fragment <nome> description "<desc>"')
- Melhorias para os módulos ou fragmentos existentes
"""
        logger.debug(f"""Generated Prompt:
{prompt}""")

        # 4. Execute ProfessorLLMFragment
        try:
            logger.info("Sending prompt to ProfessorLLMFragment...")
            # Use ctx.fragment_registry
            llm_response = await ctx.fragment_registry.execute_fragment(
                "professor_llm", {"objective": prompt}
            )
            logger.info("Received response from ProfessorLLMFragment.")
            if isinstance(llm_response, dict) and "error" in llm_response:
                 logger.error(f"Error from ProfessorLLMFragment: {llm_response['error']}")
                 return {"error": f"ProfessorLLMFragment failed: {llm_response['error']}"}
            elif not isinstance(llm_response, str):
                 logger.warning(f"Unexpected response type from ProfessorLLMFragment: {type(llm_response)}")
                 llm_response = str(llm_response) # Attempt to convert

        except Exception as e:
            logger.exception("Failed to execute ProfessorLLMFragment")
            # Pass context details if helpful
            return {"error": f"Failed to execute ProfessorLLMFragment: {e}"}

        # 5. Log Result
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "summary_sent": summary,
                "llm_response": llm_response,
                "interpreted_suggestion": None # Placeholder for future interpretation
            }
            # Use ctx.a3x_home if available
            # os.makedirs(os.path.dirname(self.suggestions_file), exist_ok=True)
            with open(self.suggestions_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            logger.info(f"Evolution suggestion logged to {self.suggestions_file}")

        except Exception as e:
            logger.exception(f"Failed to log evolution suggestion to {self.suggestions_file}")
            # Continue even if logging fails, but report the error

        # 6. (Optional) Interpret and return A3L commands - Skipping for now
        # TODO: Implement interpretation logic if needed

        return {"summary": summary, "llm_suggestion": llm_response} 