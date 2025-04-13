import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import subprocess
from a3x.core.config import LEARNING_LOGS_DIR, HEURISTIC_LOG_FILE # Import constants

logger = logging.getLogger(__name__)

def collect_training_examples(
    logs_dir: Path = LEARNING_LOGS_DIR, # Use constant for default logs_dir
    max_examples: int = 100,
    prioritize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Coleta e prioriza exemplos de aprendizado (sucessos, falhas, heurísticas) dos logs.
    """
    examples = []
    # Use the imported constant for the specific file
    log_file = HEURISTIC_LOG_FILE
    if not log_file.exists():
        logger.warning(f"[FinetunePipeline] Log file {log_file} not found.")
        return examples
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                # Prioriza heurísticas de sucesso/falha e exemplos recentes
                if entry.get("type") in ("success", "failure", "recovery_success", "parsing_fallback"):
                    examples.append(entry)
            except Exception as e:
                logger.warning(f"[FinetunePipeline] Failed to parse log line: {e}")
    # Ordena por timestamp (mais recentes primeiro)
    examples.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    if prioritize:
        # Pode-se adicionar lógica de priorização por impacto, diversidade, etc.
        pass
    return examples[:max_examples]

def save_examples_for_finetuning(examples: List[Dict[str, Any]], output_path: Path):
    """
    Salva exemplos em formato JSONL para fine-tuning.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"[FinetunePipeline] Saved {len(examples)} examples to {output_path}")

def run_qlora_finetune(
    data_path: Path,
    model_path: Path,
    output_dir: Path,
    batch_size: int = 1,
    epochs: int = 1,
    use_cpu: bool = True,
):
    """
    Executa fine-tuning QLoRA/LoRA em modo CPU, usando script externo (HuggingFace, bitsandbytes, etc).
    """
    logger.info("[FinetunePipeline] Starting QLoRA fine-tuning (CPU mode)...")
    # Exemplo de comando (ajuste conforme seu script/infraestrutura)
    command = [
        "python3", "scripts/finetune_qlora.py", # Assuming script exists at this relative path
        "--model_name_or_path", str(model_path),
        "--train_file", str(data_path),
        "--output_dir", str(output_dir),
        "--per_device_train_batch_size", str(batch_size),
        "--num_train_epochs", str(epochs),
        "--quantization", "4bit",
        "--use_cpu", "True" if use_cpu else "False",
    ]
    logger.info(f"[FinetunePipeline] Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    logger.info(f"[FinetunePipeline] Fine-tune stdout:\n{result.stdout}")
    logger.info(f"[FinetunePipeline] Fine-tune stderr:\n{result.stderr}")
    if result.returncode != 0:
        logger.error(f"[FinetunePipeline] Fine-tuning failed with exit code {result.returncode}")
    else:
        logger.info("[FinetunePipeline] Fine-tuning completed successfully.")

def validate_and_rollback(model_path: Path, validation_fn, backup_path: Path):
    """
    Valida o modelo após fine-tuning e faz rollback se necessário.
    """
    logger.info("[FinetunePipeline] Validating fine-tuned model...")
    is_valid = validation_fn(model_path)
    if not is_valid:
        logger.warning("[FinetunePipeline] Validation failed. Rolling back to previous model.")
        if backup_path.exists():
            model_path.write_bytes(backup_path.read_bytes())
            logger.info("[FinetunePipeline] Rollback completed.")
        else:
            logger.error("[FinetunePipeline] No backup found for rollback.")
    else:
        logger.info("[FinetunePipeline] Model validated and accepted.")

def run_finetune_pipeline(
    logs_dir: Path = LEARNING_LOGS_DIR, # Use constant for default logs_dir
    model_path: Path = Path("models/base_model.gguf"), # Keep model paths relative or make them config constants too
    output_dir: Path = Path("models/finetuned"), # Keep model paths relative or make them config constants too
    max_examples: int = 100,
    batch_size: int = 1,
    epochs: int = 1,
):
    """
    Pipeline completo: coleta exemplos, salva dataset, executa fine-tuning e valida.
    """
    logger.info("[FinetunePipeline] Starting full fine-tune pipeline...")
    examples = collect_training_examples(logs_dir, max_examples)
    # Use logs_dir (which now defaults to the constant) to create the dataset path
    data_path = logs_dir / "finetune_dataset.jsonl"
    save_examples_for_finetuning(examples, data_path)
    # Backup do modelo original
    backup_path = model_path.with_suffix(".bak")
    if model_path.exists():
        backup_path.write_bytes(model_path.read_bytes())
    run_qlora_finetune(data_path, model_path, output_dir, batch_size, epochs, use_cpu=True)
    # Validação simples: placeholder (substitua por benchmark real)
    def dummy_validation(model_path): return True
    validate_and_rollback(model_path, dummy_validation, backup_path)
    logger.info("[FinetunePipeline] Pipeline completed.")
