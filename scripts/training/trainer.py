import torch
import logging
import os # Importar os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import accelerate # Implicitamente usado pelo Trainer
import bitsandbytes # Para o otimizador 8bit
from datasets import Dataset
import json

from a3x.core.db_utils import sample_experiences
from a3x.core.config import (
    BASE_MODEL_NAME, # Ex: "google/gemma-2b"
    QLORA_R,
    QLORA_ALPHA,
    QLORA_DROPOUT,
    TRAINING_OUTPUT_DIR,
    TRAINING_BATCH_SIZE,
    TRAINING_GRAD_ACCUMULATION,
    TRAINING_EPOCHS,
    TRAINING_LEARNING_RATE,
    # Adicionar mais configs se necessário (quant_type, compute_dtype, etc.)
)

logger = logging.getLogger(__name__)

def prepare_dataset(tokenizer, batch_size=100):
    """Busca experiências do buffer, as formata e cria um Dataset."""
    logger.info(f"Sampling experiences for training dataset (batch size: {batch_size})")
    # TODO: Aumentar batch_size aqui pode ser útil para ter mais variedade inicial
    experiences = sample_experiences(batch_size=batch_size)

    if not experiences:
        logger.warning("No experiences sampled from buffer. Cannot create dataset.")
        return None

    formatted_texts = []
    for exp in experiences:
        context = exp['context']
        action = exp['action']
        outcome = exp['outcome']
        metadata_str = exp['metadata']
        metadata = json.loads(metadata_str) if metadata_str else {}

        # Formato: Contexto -> Ação -> Resultado
        text = f"### Context:\n{context}\n\n### Action Taken:\n{action}\n\n### Observed Outcome:\n{outcome} {json.dumps(metadata)}\n{tokenizer.eos_token}"
        formatted_texts.append(text)

    logger.info(f"Formatted {len(formatted_texts)} experiences for training.")

    # Tokenizar os textos formatados
    logger.info("Tokenizing formatted texts...")
    # Usar padding=False aqui, o DataCollator cuidará disso
    # Manter truncation para evitar exemplos excessivamente longos
    # Verificar max_length com base no CONTEXT_SIZE do config?
    tokenized_inputs = tokenizer(formatted_texts, truncation=True, max_length=512, padding=False)

    # Criar um objeto Dataset
    try:
        dataset = Dataset.from_dict(tokenized_inputs)
        logger.info("Dataset object created successfully.")
        return dataset
    except Exception as e:
        logger.exception(f"Failed to create Dataset object:")
        return None

def run_qlora_finetuning():
    """Executa um ciclo de fine-tuning QLoRA."""
    logger.info("Starting QLoRA fine-tuning cycle...")

    # 1. Configurar Quantização (BitsAndBytes)
    # TODO: Permitir configuração via config.py (BNB_COMPUTE_DTYPE, etc.)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", # Normal Float 4
        bnb_4bit_compute_dtype=torch.bfloat16 # Ou float16 se bfloat16 não for suportado
    )
    logger.info(f"BitsAndBytesConfig: {bnb_config}")

    # 2. Carregar Modelo e Tokenizer Base
    logger.info(f"Loading base model: {BASE_MODEL_NAME}")
    # TODO: Adicionar `device_map='auto'` para accelerate gerenciar dispositivos?
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto", # Deixa accelerate gerenciar
        # trust_remote_code=True, # Se necessário para alguns modelos
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    # Definir padding token se não existir
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    logger.info("Base model and tokenizer loaded.")

    # 3. Preparar Modelo para Treinamento K-bit (QLoRA)
    model = prepare_model_for_kbit_training(model)
    logger.info("Model prepared for K-bit training.")

    # 4. Configurar LoRA
    # TODO: Ler target_modules do config? Precisa parsear a string.
    # Usando um valor comum para Gemma como padrão por enquanto.
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    logger.warning(f"Using default target_modules for Gemma: {lora_target_modules}. Adjust in config.py if needed.")

    lora_config = LoraConfig(
        r=QLORA_R,
        lora_alpha=QLORA_ALPHA,
        lora_dropout=QLORA_DROPOUT,
        target_modules=lora_target_modules, # <<< Usar a variável
        bias="none",
        task_type="CAUSAL_LM",
    )
    logger.info(f"LoRA Config: {lora_config}")

    # 5. Aplicar PEFT (LoRA) ao Modelo
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("PEFT model created.")

    # 6. Preparar Dataset
    # Ajustar o batch_size aqui se necessário para pegar variedade
    train_dataset = prepare_dataset(tokenizer, batch_size=TRAINING_BATCH_SIZE * 10)
    if train_dataset is None:
        logger.error("Training dataset could not be prepared. Aborting fine-tuning cycle.")
        return

    # 7. Configurar Argumentos de Treinamento
    training_args = TrainingArguments(
        output_dir=TRAINING_OUTPUT_DIR,
        per_device_train_batch_size=TRAINING_BATCH_SIZE,
        gradient_accumulation_steps=TRAINING_GRAD_ACCUMULATION,
        num_train_epochs=TRAINING_EPOCHS,
        learning_rate=TRAINING_LEARNING_RATE,
        optim="paged_adamw_8bit", # Otimizador QLoRA
        fp16=False, # Usar bfloat16 se disponível via compute_dtype do BnB
        bf16=True if bnb_config.bnb_4bit_compute_dtype == torch.bfloat16 else False,
        logging_steps=10, # Logar progresso
        save_strategy="epoch", # Salvar adaptador a cada época
        # gradient_checkpointing=True, # Considerar habilitar
        # TODO: Adicionar mais argumentos? Warmup, weight decay, etc.
    )
    logger.info(f"TrainingArguments: {training_args}")

    # 8. Inicializar Data Collator
    # mlm=False significa que não estamos fazendo Masked Language Modeling, mas sim Causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    logger.info("DataCollatorForLanguageModeling initialized.")

    # 9. Inicializar Trainer (passando o data_collator)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 10. Iniciar Treinamento
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training finished successfully.")

        # 11. Salvar Adaptador LoRA Final
        final_adapter_path = os.path.join(TRAINING_OUTPUT_DIR, "final_adapter")
        model.save_pretrained(final_adapter_path)
        logger.info(f"Final LoRA adapter saved to: {final_adapter_path}")

    except Exception as e:
        logger.exception("An error occurred during training:")

if __name__ == '__main__':
    # Apenas para teste rápido, idealmente seria chamado por outro módulo
    logging.basicConfig(level=logging.INFO)
    # Não precisa mais definir configs aqui, elas são importadas de a3x.core.config
    # from a3x.core import config
    # config.BASE_MODEL_NAME = "google/gemma-2b" # Exemplo!
    # config.QLORA_R = 8
    # ... (remover as definições manuais)
    # A criação do diretório já está no config.py
    # import os
    # os.makedirs(config.TRAINING_OUTPUT_DIR, exist_ok=True)

    run_qlora_finetuning() 