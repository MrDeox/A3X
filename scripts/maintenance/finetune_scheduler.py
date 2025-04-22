"""
Agendador e trigger inteligente para o pipeline de fine-tuning do A³X.
- Pode ser executado periodicamente (cron, systemd, etc) ou manualmente.
- Aciona o fine-tuning se houver exemplos suficientes, após ciclos cognitivos ou sob demanda.
- Garante logs, versionamento e rollback seguro.
"""

import time
import logging
from pathlib import Path
from a3x.core.finetune_pipeline import run_finetune_pipeline, collect_training_examples

# Configurações principais
LOGS_DIR = Path("memory/learning_logs")
MODEL_PATH = Path("models/base_model.gguf")
OUTPUT_DIR = Path("models/finetuned")
MIN_EXAMPLES = 50  # Quantidade mínima de exemplos para acionar fine-tuning
CHECK_INTERVAL = 60 * 60 * 6  # 6 horas (pode ser ajustado)
MAX_EXAMPLES = 100
BATCH_SIZE = 1
EPOCHS = 1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finetune_scheduler")

def should_trigger_finetune(logs_dir: Path, min_examples: int) -> bool:
    """
    Verifica se há exemplos suficientes para acionar o fine-tuning.
    """
    examples = collect_training_examples(logs_dir, max_examples=1000)
    logger.info(f"[FinetuneScheduler] Exemplos disponíveis para fine-tuning: {len(examples)}")
    return len(examples) >= min_examples

def main_loop():
    """
    Loop principal do agendador: verifica periodicamente e aciona o pipeline se necessário.
    """
    logger.info("[FinetuneScheduler] Iniciando agendador de fine-tuning do A³X.")
    while True:
        if should_trigger_finetune(LOGS_DIR, MIN_EXAMPLES):
            logger.info("[FinetuneScheduler] Critério atingido. Acionando pipeline de fine-tuning.")
            run_finetune_pipeline(
                logs_dir=LOGS_DIR,
                model_path=MODEL_PATH,
                output_dir=OUTPUT_DIR,
                max_examples=MAX_EXAMPLES,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
            )
        else:
            logger.info("[FinetuneScheduler] Critério não atingido. Aguardando próximo ciclo.")
        logger.info(f"[FinetuneScheduler] Dormindo por {CHECK_INTERVAL/3600:.1f} horas...")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main_loop()