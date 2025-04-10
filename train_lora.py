import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import os

# --- Configuration ---
MODEL_ID = "google/gemma-2b"
DATA_PATH = "data.jsonl"
OUTPUT_DIR = "./gemma-2b-lora-finetuned"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"] # Common for Gemma/Llama family

# Training Args
PER_DEVICE_BATCH_SIZE = 1 # Start very small for low VRAM
GRAD_ACCUMULATION_STEPS = 4
OPTIMIZER = "adamw_torch"
LEARNING_RATE = 2e-5
FP16_ENABLED = torch.cuda.is_available() # Enable FP16 only if CUDA is detected
GRADIENT_CHECKPOINTING_ENABLED = True
MAX_STEPS = 10 # Set low for a quick test run
LOGGING_STEPS = 1
SAVE_STEPS = 5

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    # Optional: Check bitsandbytes compatibility for potential future use
    try:
        import bitsandbytes
        print("bitsandbytes is available.")
    except ImportError:
        print("bitsandbytes not found or incompatible.")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")
    FP16_ENABLED = False # FP16 requires GPU support

# --- Load Model and Tokenizer ---
print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Set pad token if it doesn't exist (common for models like Gemma)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    # device_map="auto", # Use accelerate or manual .to(device)
    # Consider torch_dtype=torch.float16 if CUDA is available and VRAM is sufficient, even without bnb
    torch_dtype=torch.float16 if FP16_ENABLED else torch.float32,
)

# Enable gradient checkpointing BEFORE applying LoRA
if GRADIENT_CHECKPOINTING_ENABLED:
    model.gradient_checkpointing_enable()
    print("Gradient Checkpointing Enabled.")


# --- LoRA Configuration ---
print("Applying LoRA configuration...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Move model to device *after* PEFT application if not using device_map='auto'
model.to(device)
print(f"Model moved to device: {model.device}")


# --- Load and Prepare Dataset ---
print(f"Loading dataset from: {DATA_PATH}")
dataset = load_dataset("json", data_files=DATA_PATH)["train"]

def tokenize_function(examples):
    # Simple tokenization, adjust max_length as needed
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

print("Tokenizing dataset...")
# Use remove_columns to keep only model inputs
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
print(f"Tokenized dataset features: {tokenized_dataset.features}")


# --- Configure Training ---
print("Setting up Training Arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    optim=OPTIMIZER,
    learning_rate=LEARNING_RATE,
    fp16=FP16_ENABLED,
    gradient_checkpointing=GRADIENT_CHECKPOINTING_ENABLED, # Already enabled on model, but good practice
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    max_steps=MAX_STEPS,
    report_to="none", # Disable wandb/tensorboard reporting for this example
    save_total_limit=1, # Only keep the latest checkpoint
    remove_unused_columns=False # PEFT adds columns
)

# --- Initialize Trainer ---
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer, # Pass tokenizer for proper saving
)

# --- Start Training ---
print("Starting training...")
trainer.train()

# --- Save Final Model ---
final_save_path = os.path.join(OUTPUT_DIR, "final_model")
print(f"Saving final LoRA adapter model to: {final_save_path}")
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)

print("Training finished.")

# --- Memory Cleanup (Optional) ---
if torch.cuda.is_available():
    print("Cleaning CUDA cache...")
    torch.cuda.empty_cache() 