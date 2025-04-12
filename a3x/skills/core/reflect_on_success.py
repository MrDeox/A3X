import logging
import json
from typing import Dict, Any, Optional, List
import inspect

from a3x.core.skills import skill
from a3x.core.llm_interface import call_llm
# Try importing Context, handle potential ImportError if structure changes
try:
    from a3x.core.context import Context
except ImportError:
    Context = Any # Fallback type

# Initialize logger for this skill
reflect_logger = logging.getLogger(f"a3x.skills.reflect_on_success")

# --- System Prompt for Generating Positive Heuristics ---
# TODO: Refine this prompt based on testing
SUCCESS_HEURISTIC_SYSTEM_PROMPT = """
Você é um analista de processos especializado em identificar padrões de sucesso em logs de execução de agentes autônomos.
Sua tarefa é analisar um trecho de uma execução BEM-SUCEDIDA de um agente e gerar UMA ÚNICA heurística POSITIVA e ACIONÁVEL.

**Formato da Análise Fornecida:**
- **Objetivo:** O objetivo geral que o agente tentava alcançar.
- **Passos Relevantes:** A sequência de ações (skills usadas com parâmetros) que levaram ao sucesso neste trecho.
- **Resultado Final:** A confirmação do sucesso.

**Sua Resposta:**
- Gere APENAS um JSON contendo a chave "heuristic".
- A heurística deve ser uma regra curta e direta, focando no QUE funcionou e PORQUÊ (se óbvio).
- Exemplo: "Para [tipo de objetivo], usar a skill [nome_skill] com [parâmetro chave] mostrou-se eficaz para [resultado específico]."
- Exemplo: "A sequência [skill_A] -> [skill_B] é uma boa abordagem para tarefas de [categoria de tarefa]."
- EVITE generalizações excessivas se os dados forem específicos.

**Exemplo de Entrada:**
```json
{
  "objective": "Listar arquivos no diretório 'src'",
  "relevant_steps": [
    {"step": 1, "action": "list_dir", "action_input": {"relative_workspace_path": "src"}, "observation": "[lista de arquivos...]"}
  ],
  "final_outcome": "Successfully listed files in src."
}
```

**Exemplo de Saída Esperada (APENAS JSON):**
```json
{
  "heuristic": "Para listar arquivos, a skill 'list_dir' com o parâmetro 'relative_workspace_path' direcionado ao diretório desejado é eficaz."
}
```

Analise a execução fornecida e gere a heurística positiva em formato JSON."""

@skill(
    name="reflect_on_success",
    description="Analisa uma execução bem-sucedida e gera uma heurística positiva (uma regra sobre o que funcionou).",
    parameters={
        "objective": (str, ...),
        "relevant_steps": (List[Dict[str, Any]], ...),
        "final_outcome": (str, ...),
        "ctx": (Context, None)
    }
)
async def reflect_on_success(
    objective: str,
    relevant_steps: List[Dict[str, Any]],
    final_outcome: str,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Uses an LLM to analyze a successful execution sequence and generate a positive heuristic.
    """
    reflect_logger.info(f"Reflecting on success for objective: {objective[:100]}...")

    # Prepare input for the LLM
    llm_input_data = {
        "objective": objective,
        "relevant_steps": relevant_steps,
        "final_outcome": final_outcome
    }

    # Build prompt messages
    prompt_messages = [
        {"role": "system", "content": SUCCESS_HEURISTIC_SYSTEM_PROMPT},\
        {"role": "user", "content": f"Analise a seguinte execução bem-sucedida:\n```json\n{json.dumps(llm_input_data, indent=2)}\n```\nGere a heurística positiva em formato JSON."}\
    ]

    heuristic_json_str = ""
    llm_url = getattr(ctx, 'llm_url', None) if ctx else None
    try:
        reflect_logger.debug("Calling LLM to generate positive heuristic...")
        async for chunk in call_llm(prompt_messages, llm_url=llm_url, stream=False, temperature=0.2): # Lower temp for focused output
             heuristic_json_str += chunk

        reflect_logger.debug(f"LLM raw response for heuristic: {heuristic_json_str}")
        parsed_output = parse_llm_json_output(heuristic_json_str, ["heuristic"], reflect_logger)

        if parsed_output and "heuristic" in parsed_output:
            heuristic_text = parsed_output["heuristic"]
            reflect_logger.info(f"Generated positive heuristic: {heuristic_text}")
            # Prepare context snapshot (simple version for now)
            context_snapshot = {
                "objective_summary": objective[:150] + ("..." if len(objective) > 150 else ""),
                "successful_actions": [step.get("action") for step in relevant_steps],\
                 # Add more context later: tools available, relevant state, etc.
            }
            return {
                "status": "success",
                "data": {
                    "heuristic": heuristic_text,
                    "context_snapshot": context_snapshot
                }
            }
        else:
             reflect_logger.error("Failed to parse valid heuristic from LLM response.")
             return {"status": "error", "data": {"message": "LLM did not return a valid heuristic JSON."}}

    except Exception as e:
        reflect_logger.exception(f"Error during positive heuristic generation: {e}")
        return {"status": "error", "data": {"message": f"Exception during LLM call: {e}"}}

# Example usage (for testing)
if __name__ == '__main__':
    import asyncio
    # import json # Already imported above
    logging.basicConfig(level=logging.DEBUG)

    async def run_test():
        test_data = {
            "objective": "Criar um novo arquivo chamado 'test.txt' e escrever 'hello world' nele.",
            "relevant_steps": [
                {"step": 1, "action": "create_file", "action_input": {"target_file": "test.txt"}, "observation": "File created."},
                {"step": 2, "action": "edit_file", "action_input": {"target_file": "test.txt", "code_edit": "hello world", "instructions": "Write hello world"}, "observation": "File updated."}
            ],
            "final_outcome": "Successfully created and wrote to test.txt.",
            "ctx": None # No context needed for this basic test
        }
        # Corrected import for json
        # import json # Already imported above
        result = await reflect_on_success(**test_data)
        print("\n--- Reflection Result ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(run_test()) 