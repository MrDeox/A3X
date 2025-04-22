import logging
import json
import time
import asyncio
import functools
from pathlib import Path
from typing import Dict, Optional, Any

# Assuming these are available in the environment
from a3x.a3net.core.memory_bank import MemoryBank
from a3x.a3net.core.context_store import ContextStore
from a3x.a3net.core.fragment_cell import FragmentCell
from a3x.a3net.core.professor_llm_fragment import ProfessorLLMFragment
from a3x.a3net.trainer.dataset_builder import build_dataset_from_context
from ...modules.utils import append_to_log

logger = logging.getLogger(__name__)

async def handle_verify_knowledge(
    directive: Dict[str, Any],
    memory_bank: MemoryBank,
    fragment_instances: Optional[Dict[str, Any]],
    context_store: Optional[ContextStore]
) -> Optional[Dict[str, Any]]:
    """Handles the 'verificar_conhecimento' directive logic."""

    target_fragment_id = directive.get("fragment_id")
    professor_id = directive.get("professor_id", "prof_geral") # Default professor
    origin = directive.get("_origin", "Unknown Verify Origin")

    if not target_fragment_id:
        logger.error("[A3X Bridge Handler - Verify] 'fragment_id' missing.")
        return {"status": "error", "message": "'fragment_id' missing for verificar_conhecimento"}

    if not fragment_instances:
        logger.error("[A3X Bridge Handler - Verify] fragment_instances not available.")
        return {"status": "error", "message": "fragment_instances unavailable"}
    
    if not context_store:
         logger.error("[A3X Bridge Handler - Verify] ContextStore not available.")
         return {"status": "error", "message": "ContextStore unavailable for verification"}

    logger.info(f"[A3X Bridge Handler - Verify] Verifying knowledge of fragment '{target_fragment_id}' using professor '{professor_id}' (Origin: {origin})...")

    # --- Load Target Fragment --- 
    target_fragment = memory_bank.load(target_fragment_id)
    if not isinstance(target_fragment, FragmentCell): # Check base class or specific trainable types
        logger.error(f"[A3X Bridge Handler - Verify] Target fragment '{target_fragment_id}' not found or invalid type.")
        return {"status": "error", "message": f"Target fragment '{target_fragment_id}' not found or invalid."}

    # --- Load Professor Fragment --- 
    professor_fragment = fragment_instances.get(professor_id)
    if not isinstance(professor_fragment, ProfessorLLMFragment):
         logger.error(f"[A3X Bridge Handler - Verify] Professor fragment '{professor_id}' not found or is not a ProfessorLLMFragment.")
         return { "status": "error", "message": f"Professor fragment '{professor_id}' not available." }
    
    # --- Get Task Name and Sample Data --- 
    task_name = getattr(target_fragment, 'associated_task_name', None)
    sample_data_str = ""
    if not task_name:
         logger.warning(f"[A3X Bridge Handler - Verify] Fragment '{target_fragment_id}' has no associated task name. Cannot fetch sample data.")
         sample_data_str = "Nenhum dado de exemplo disponível (sem nome de tarefa associado)."
    else:
         try:
             logger.info(f"[A3X Bridge Handler - Verify] Fetching sample data for task '{task_name}'...")
             # Load full dataset and take a small sample
             loop = asyncio.get_running_loop()
             build_fn = functools.partial(build_dataset_from_context, task_name, getattr(target_fragment, 'num_classes', 3))
             full_dataset = await loop.run_in_executor(None, build_fn)
             
             if not full_dataset:
                  sample_data_str = f"Nenhum dado de exemplo encontrado para a tarefa '{task_name}'."
             else:
                 sample_size = min(len(full_dataset), 5)
                 dataset_file = Path(f"data/datasets/a3net/{task_name}.jsonl")
                 if dataset_file.is_file():
                     raw_samples = []
                     try:
                         with open(dataset_file, 'r', encoding='utf-8') as f:
                             for i, line in enumerate(f):
                                 if i >= sample_size: break
                                 try: 
                                     sample = json.loads(line.strip())
                                     if isinstance(sample, dict) and 'input' in sample and 'label' in sample:
                                         raw_samples.append(sample)
                                 except json.JSONDecodeError: pass
                         if raw_samples:
                            sample_data_str = "Amostra de Dados da Tarefa:\n"
                            for sample in raw_samples:
                                 sample_data_str += f"  - Input: {sample.get('input')}\n    Label: {sample.get('label')}\n"
                         else:
                            sample_data_str = f"Nenhum exemplo JSON válido encontrado no arquivo {dataset_file}."
                     except Exception as read_err:
                          sample_data_str = f"Erro ao ler arquivo de dataset {dataset_file}: {read_err}"
                 else:
                    sample_data_str = f"Arquivo de dataset {dataset_file} não encontrado para obter exemplos brutos."
                     
         except Exception as data_err:
             logger.error(f"[A3X Bridge Handler - Verify] Error fetching/processing sample data for task '{task_name}': {data_err}", exc_info=True)
             sample_data_str = f"Erro ao obter dados de exemplo para a tarefa '{task_name}': {data_err}"

    # --- Get Fragment Description --- 
    fragment_desc = "Descrição não disponível."
    if hasattr(target_fragment, 'generate_reflection_a3l') and callable(getattr(target_fragment, 'generate_reflection_a3l')):
        try:
            fragment_desc = target_fragment.generate_reflection_a3l()
        except Exception as reflect_err:
            logger.warning(f"[A3X Bridge Handler - Verify] Error generating reflection for '{target_fragment_id}': {reflect_err}")
            fragment_desc = f"Erro ao gerar reflexão: {reflect_err}"
    elif hasattr(target_fragment, 'description'):
         fragment_desc = getattr(target_fragment, 'description', fragment_desc)

    # --- Craft Prompt for Professor --- 
    prompt = f"""
Avalie o fragmento de rede neural com ID '{target_fragment_id}'.

Descrição do Fragmento:
{fragment_desc}

{sample_data_str}

Com base na descrição e nos exemplos (se disponíveis), este fragmento parece ter aprendido a tarefa '{task_name or 'desconhecida'}' de forma satisfatória e generalizada?

Responda APENAS com "sim", "não", ou "incerto". Se "não" ou "incerto", pode opcionalmente adicionar uma sugestão MUITO breve após a palavra (ex: "não, precisa mais exemplos", "incerto, avaliar com dados reais").
"""
    logger.debug(f"[A3X Bridge Handler - Verify] Prompt para Professor '{professor_id}':\n{prompt}")

    # --- Ask Professor --- 
    try:
        response = await professor_fragment.ask_llm(prompt)
        logger.info(f"[A3X Bridge Handler - Verify] Resposta do Professor '{professor_id}': {response}")
        
        response_lower = response.strip().lower()
        approved = False
        status_message = f"Verificação de conhecimento para '{target_fragment_id}' concluída."
        
        if response_lower.startswith("sim"):
            approved = True
            status_message += " Fragmento aprovado pelo Professor." 
            logger.info(f"[A3X Bridge Handler - Verify] Fragmento '{target_fragment_id}' APROVADO pelo Professor.")
            try:
                 status_key = f"fragment_status:{target_fragment_id}"
                 status_data = {"knowledge_verified": True, "verified_by": professor_id, "timestamp": time.time()}
                 await context_store.set(status_key, status_data)
                 logger.info(f"[A3X Bridge Handler - Verify] Status 'knowledge_verified=True' salvo no ContextStore para '{target_fragment_id}'.")
                 append_to_log(f"# [VERIFICACAO CONHECIMENTO] Fragmento '{target_fragment_id}' APROVADO por '{professor_id}'")
            except Exception as cs_err:
                 logger.error(f"[A3X Bridge Handler - Verify] Falha ao salvar status de aprovação no ContextStore para '{target_fragment_id}': {cs_err}", exc_info=True)
                 status_message += " (Erro ao salvar status no ContextStore)"
        elif response_lower.startswith("não") or response_lower.startswith("incerto"):
             approved = False
             status_message += f" Fragmento NÃO aprovado pelo Professor. Resposta: {response}"
             logger.warning(f"[A3X Bridge Handler - Verify] Fragmento '{target_fragment_id}' NÃO APROVADO pelo Professor. Resposta: {response}")
             append_to_log(f"# [VERIFICACAO CONHECIMENTO] Fragmento '{target_fragment_id}' REPROVADO/INCERTO por '{professor_id}'. Resposta: {response}")
        else:
             approved = False
             status_message += f" Resposta do Professor não reconhecida: {response}"
             logger.warning(f"[A3X Bridge Handler - Verify] Resposta do Professor para '{target_fragment_id}' não reconhecida: {response}")
             append_to_log(f"# [VERIFICACAO CONHECIMENTO] Resposta não reconhecida de '{professor_id}' para '{target_fragment_id}'. Resposta: {response}")

        return {"status": "success", "message": status_message, "fragment_id": target_fragment_id, "professor_id": professor_id, "approved": approved, "professor_response": response}

    except Exception as e:
        logger.error(f"[A3X Bridge Handler - Verify] Erro ao verificar conhecimento de '{target_fragment_id}' com Professor '{professor_id}': {e}", exc_info=True)
        append_to_log(f"# [FALHA VERIFICACAO] Erro ao consultar professor '{professor_id}' sobre '{target_fragment_id}': {e}")
        return {"status": "error", "message": f"Erro ao consultar Professor: {e}", "fragment_id": target_fragment_id, "professor_id": professor_id} 