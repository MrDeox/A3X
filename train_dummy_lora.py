import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# --- Configuration ---
model_name = "google/gemma-1.1-2b-it" # Use 2B for initial test to save resources? Or stick with 4B? Let's try 2B first.
# model_name = "google/gemma-3-4b-it" # If 2B works, try 4B
dataset_path = "workspace/dummy_train.txt" # Path to dummy text file
output_dir = "./lora_adapters/dummy_rank4_gemma2b" # Directory to save the adapter
lora_rank = 4
lora_alpha = 16 # Standard practice: alpha = 2 * rank
lora_dropout = 0.05
# Training Args (minimal for just generating adapter)
num_train_epochs = 1
per_device_train_batch_size = 1
gradient_accumulation_steps = 1
learning_rate = 2e-4
# --- End Configuration ---

# --- Setup ---
os.makedirs(output_dir, exist_ok=True)

# Quantization Config (4-bit)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Or float16 if bfloat16 not supported
    bnb_4bit_use_double_quant=False,
)

# Load Model (Quantized)
print(f"Loading base model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto", # Automatically use GPU if available, otherwise CPU/RAM
    trust_remote_code=True # Gemma requires this
)
model.config.use_cache = False # Recommended for fine-tuning
model.config.pretraining_tp = 1 # Not sure if needed for Gemma, but often recommended

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Set pad token

# Prepare model for K-bit training (important for QLoRA)
print("Preparing model for K-bit training...")
model = prepare_model_for_kbit_training(model)

# LoRA Config
peft_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    # Target modules might vary per model - check Gemma's architecture
    # Common targets: 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
    # Let's try targeting common ones, might need adjustment
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # Gemma might use different names or structures
        # "gate_proj",
        # "up_proj",
        # "down_proj",
    ],
)

# Apply LoRA adapter
print("Applying PEFT LoRA adapter...")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Load Dataset (simple text file)
print(f"Loading dataset from: {dataset_path}")
dataset = load_dataset("text", data_files={"train": dataset_path})

# Basic function to tokenize dataset (adjust as needed)
def tokenize_function(examples):
    # Simple tokenization, might need max_length etc. for real tasks
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training Arguments (Minimal)
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    logging_steps=10, # Log frequently for short run
    save_strategy="epoch", # Save at the end of the (only) epoch
    report_to="none", # Disable external reporting like wandb
    fp16=False, # Use bf16 if available and specified in quant_config
    bf16=True if torch.cuda.is_bf16_supported() else False,
    # Use CPU offloading if VRAM is extremely tight (will be very slow)
    # gradient_checkpointing=True, # Helps save VRAM but slows down
    # optim="paged_adamw_8bit", # Paged optimizer saves VRAM
)

# Trainer (using SFTTrainer might be simpler for text datasets)
# Using basic Trainer for now
from transformers import Trainer, DataCollatorForLanguageModeling

print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Train (Fine-tune LoRA weights)
print("Starting dummy training...")
trainer.train()

# Save the LoRA adapter
print(f"Saving LoRA adapter to: {output_dir}")
trainer.save_model(output_dir) # Saves the adapter config and weights

# Explicitly delete model and clear cache to free VRAM if needed
# del model
# del trainer
# torch.cuda.empty_cache()

print("Dummy LoRA training complete. Adapter saved.")
print(f"Next step: Convert the adapter in '{output_dir}' to GGUF using llama.cpp/convert_lora_to_gguf.py") 