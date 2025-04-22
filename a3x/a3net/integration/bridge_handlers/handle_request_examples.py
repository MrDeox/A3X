import logging
import json
import re
import asyncio
from typing import Dict, Optional, Any

# Assuming these are available in the environment
from a3x.a3net.core.professor_llm_fragment import ProfessorLLMFragment
from a3x.a3net.utils.data_logger import registrar_exemplo_de_aprendizado as log_example

logger = logging.getLogger(__name__)

async def handle_request_examples(
    directive: Dict[str, Any],
    fragment_instances: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Handles the 'solicitar_exemplos' directive logic."""

    task_name = directive.get("task_name")
    professor_id = directive.get("professor_id", "prof_geral") # Default professor
    num_examples = directive.get("num_examples", 10) # Default number
    example_format = directive.get("example_format") # Optional format

    if not task_name:
        logger.error("[A3X Bridge Handler - RequestExamples] 'task_name' missing.")
        return { "status": "error", "message": "'task_name' missing for solicitar_exemplos" }
    
    if not fragment_instances:
         logger.error("[A3X Bridge Handler - RequestExamples] fragment_instances dictionary not provided.")
         return { "status": "error", "message": "Fragment instances not available to find Professor." }

    # --- Find Professor Fragment ---
    professor_fragment = fragment_instances.get(professor_id)
    if not isinstance(professor_fragment, ProfessorLLMFragment):
         logger.error(f"[A3X Bridge Handler - RequestExamples] Professor fragment '{professor_id}' not found or is not a ProfessorLLMFragment.")
         return { "status": "error", "message": f"Professor fragment '{professor_id}' not available." }
    
    logger.info(f"[A3X Bridge Handler - RequestExamples] Requesting {num_examples} examples for task '{task_name}' from Professor '{professor_id}'.")
    
    # --- Generate Prompt (Adapts based on example_format) ---
    prompt = ""
    if example_format and isinstance(example_format, dict):
        try:
            format_str = json.dumps(example_format, ensure_ascii=False, indent=2) # Pretty print for prompt
            logger.info(f"[A3X Bridge Handler - RequestExamples] Using planned example format: {format_str}")
            prompt = f'''
Por favor, gere {num_examples} exemplos de dados para a tarefa '{task_name}'.
Use EXATAMENTE a seguinte estrutura JSON para cada exemplo:
{format_str}

Retorne a resposta como uma lista JSON contendo os {num_examples} exemplos.
Exemplo de Resposta:
[
  {format_str}, 
  {{... outro exemplo ...}}
]
Retorne APENAS a lista JSON, sem nenhum texto adicional antes ou depois.
'''
        except Exception as json_err:
            logger.error(f"[A3X Bridge Handler - RequestExamples] Error serializing example_format: {json_err}. Falling back to default prompt.", exc_info=True)
            example_format = None # Force fallback to default prompt
    
    if not example_format: # If format wasn't provided or serialization failed
        logger.info("[A3X Bridge Handler - RequestExamples] Using default example format prompt (input/label).")
        prompt = f'''
Por favor, gere {num_examples} exemplos de dados de entrada e rótulos correspondentes para a tarefa '{task_name}'.
Os exemplos devem ser representativos da tarefa.
Retorne a resposta como uma lista JSON de objetos, onde cada objeto tem as chaves "input" e "label".
Exemplo de formato:
[
  {{"input": "exemplo de entrada 1", "label": "rótulo A"}},
  {{"input": "exemplo de entrada 2", "label": "rótulo B"}}
]
Retorne APENAS a lista JSON, sem nenhum texto adicional antes ou depois.
'''
    
    try:
        # --- Call Professor ---
        response = await professor_fragment.ask_llm(prompt)
        if not response:
             logger.error(f"[A3X Bridge Handler - RequestExamples] Professor '{professor_id}' returned an empty response.")
             return { "status": "error", "message": "Professor returned empty/invalid response" }
             
        logger.info(f"[A3X Bridge Handler - RequestExamples] Received response from Professor: {response[:200]}...")

        # --- Parse Response --- 
        parsed_examples = []
        try:
            json_match = re.search(r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\})*\s*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_examples = json.loads(json_str)
                if not isinstance(parsed_examples, list):
                    raise ValueError("Parsed JSON is not a list.")
                logger.info(f"[A3X Bridge Handler - RequestExamples] Successfully parsed {len(parsed_examples)} examples from JSON.")
            else:
                logger.error("[A3X Bridge Handler - RequestExamples] No valid JSON list found in Professor's response.")
                return { "status": "error", "message": "Could not find valid JSON list in Professor response." }

        except (json.JSONDecodeError, ValueError) as json_err:
            logger.error(f"[A3X Bridge Handler - RequestExamples] Failed to parse JSON response from Professor: {json_err}. Response: {response}")
            return { "status": "error", "message": f"Failed to parse Professor response as JSON list: {json_err}" }

        # --- Register Examples Asynchronously --- 
        registration_tasks = []
        registered_count = 0
        failed_count = 0
        for example in parsed_examples:
            if isinstance(example, dict) and "input" in example and "label" in example:
                input_data = example["input"]
                label_data = example["label"]
                # Simple type check for basic input/label case
                if isinstance(input_data, str) and isinstance(label_data, (str, int, float, bool, list, dict)):
                    registration_tasks.append(
                        log_example(task_name=task_name, input_data=input_data, label=label_data)
                    )
                else:
                    logger.warning(f"[A3X Bridge Handler - RequestExamples] Skipping example with potentially unsupported input/label types: {type(input_data)} / {type(label_data)} in example: {example}")
                    failed_count += 1
            else:
                logger.warning(f"[A3X Bridge Handler - RequestExamples] Skipping invalid example format (missing input/label): {example}")
                failed_count += 1
        
        if registration_tasks:
            results = await asyncio.gather(*registration_tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    logger.error(f"[A3X Bridge Handler - RequestExamples] Error registering example: {res}", exc_info=res)
                    failed_count += 1
                elif res is True:
                    registered_count += 1
                else: 
                    logger.warning(f"[A3X Bridge Handler - RequestExamples] data_logger reported failure for an example.")
                    failed_count += 1

        logger.info(f"[A3X Bridge Handler - RequestExamples] Registration complete for task '{task_name}'. Registered: {registered_count}, Failed/Skipped: {failed_count}")
        return {
            "status": "success",
            "message": f"Requested examples for '{task_name}'. Registered: {registered_count}, Failed/Skipped: {failed_count}",
            "registered_count": registered_count,
            "failed_count": failed_count
        }

    except Exception as e:
        logger.error(f"[A3X Bridge Handler - RequestExamples] Unexpected error: {e}", exc_info=True)
        return { "status": "error", "message": f"Unexpected error during solicitar_exemplos: {e}" } 