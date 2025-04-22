import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a language model using LoRA.")

    # Model arguments
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID (e.g., google/gemma-2b)")
    parser.add_argument("--trust_remote_code", action="store_true", help="Set trust_remote_code=True for AutoModel/Tokenizer")
    parser.add_argument("--quantization", type=str, default="none", choices=["none", "4bit"], help="Quantization mode (none or 4bit)")
    parser.add_argument("--compute_dtype", type=str, default="bfloat16", help="Compute dtype for 4-bit quantization (e.g., bfloat16, float16)")

    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training dataset file (text or jsonl).")
    parser.add_argument("--dataset_format", type=str, default="text", choices=["text", "json"], help="Format of the dataset.")
    parser.add_argument("--text_field", type=str, default="text", help="Name of the text field in the JSON dataset.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenization.")

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank for LoRA.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha for LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout for LoRA.")
    parser.add_argument("--target_modules", nargs='+', default=["q_proj", "v_proj"], help="List of modules to target with LoRA.")

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the LoRA adapter and training outputs.")
    parser.add_argument("--num_train_epochs", type=float, default=None, help="Number of training epochs.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps (overrides epochs).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps for gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--optimizer", type=str, default="adamw_torch", help="Optimizer to use.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 training (requires CUDA).")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 training (requires CUDA with Ampere or newer).")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch", "no"], help="When to save checkpoints.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps (if save_strategy='steps').")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Limit the total number of checkpoints.")
    parser.add_argument("--report_to", type=str, default="none", help="Report results to (e.g., 'wandb', 'tensorboard', 'none').")
    parser.add_argument("--use_cpu", action="store_true", help="Force training on CPU.")


    return parser.parse_args()

def main():
    args = parse_arguments()

    # --- Device Setup ---
    if args.use_cpu:
        device = torch.device("cpu")
        print("Using CPU.")
        args.fp16 = False # FP16 requires CUDA
        args.bf16 = False # BF16 requires CUDA
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        if args.quantization == "4bit" and not args.fp16 and not args.bf16:
             # Check compute_dtype compatibility
             if args.compute_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                 print("Warning: BF16 not supported on this device, falling back to FP16 for compute_dtype.")
                 args.compute_dtype = "float16"
             elif args.compute_dtype == "float16":
                 pass # FP16 is generally supported
             else:
                 print(f"Warning: Unknown compute_dtype '{args.compute_dtype}'. Defaulting to float16.")
                 args.compute_dtype = "float16"
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
        args.fp16 = False
        args.bf16 = False

    # --- Load Tokenizer ---
    print(f"Loading tokenizer for model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code
    )
    # Set pad token if it doesn't exist
    if tokenizer.pad_token is None:
        print("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token


    # --- Load Model ---
    print(f"Loading base model: {args.model_id}")
    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
    }

    quantization_config = None
    if args.quantization == "4bit":
        print("Setting up 4-bit quantization...")
        if args.compute_dtype == "bfloat16":
            compute_dtype = torch.bfloat16
        else: # Default or fallback
            compute_dtype = torch.float16

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto" # Recommended for 4-bit
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16
    elif args.bf16:
         model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32


    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        **model_kwargs
    )

    if tokenizer.pad_token_id is None:
        print("Warning: pad_token_id not set. Setting to eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.use_cache = False # Recommended for fine-tuning

    if args.quantization == "4bit":
        print("Preparing model for K-bit training...")
        # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing) # Pass GC here if needed
        model = prepare_model_for_kbit_training(model) # Don't enable GC here, Trainer args handle it
    elif args.gradient_checkpointing:
        # Enable gradient checkpointing BEFORE applying LoRA for non-quantized models
        model.gradient_checkpointing_enable()
        print("Gradient Checkpointing Enabled.")


    # --- LoRA Configuration ---
    print("Applying LoRA configuration...")
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Move model to device if not using device_map="auto" (e.g., for non-quantized)
    if args.quantization != "4bit" and device.type != 'cpu':
         model.to(device)
         print(f"Model moved to device: {model.device}")


    # --- Load and Prepare Dataset ---
    print(f"Loading dataset from: {args.dataset_path} (Format: {args.dataset_format})")
    data_files = {"train": args.dataset_path}
    dataset = load_dataset(args.dataset_format, data_files=data_files)["train"]

    def tokenize_function(examples):
        return tokenizer(
            examples[args.text_field],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length" # Pad to max_length
        )

    print("Tokenizing dataset...")
    # Use remove_columns only for JSON datasets to avoid issues with 'text' format
    remove_cols = dataset.column_names if args.dataset_format == "json" else None
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_cols
    )
    print(f"Tokenized dataset features: {tokenized_dataset.features}")


    # --- Configure Training ---
    print("Setting up Training Arguments...")
    os.makedirs(args.output_dir, exist_ok=True)

    training_args_dict = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optim": args.optimizer,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": args.report_to,
        "gradient_checkpointing": args.gradient_checkpointing,
        # Only set max_steps if provided, otherwise use num_train_epochs
        "max_steps": args.max_steps if args.max_steps > 0 else -1,
        "num_train_epochs": args.num_train_epochs if args.max_steps <= 0 and args.num_train_epochs else 0,
        "remove_unused_columns": False, # Important for PEFT
        "fp16": args.fp16,
        "bf16": args.bf16,
    }

    # Filter out num_train_epochs or max_steps if not used
    if training_args_dict["max_steps"] == -1:
        del training_args_dict["max_steps"]
    else:
        del training_args_dict["num_train_epochs"]

    training_arguments = TrainingArguments(**training_args_dict)


    # --- Initialize Trainer ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), # Use LM collator
    )

    # --- Start Training ---
    print("Starting training...")
    trainer.train()

    # --- Save Final Model ---
    final_save_path = os.path.join(args.output_dir, "final_adapter")
    print(f"Saving final LoRA adapter model to: {final_save_path}")
    model.save_pretrained(final_save_path) # Saves only the adapter
    # Save tokenizer for convenience
    tokenizer.save_pretrained(final_save_path)

    print("Training finished.")

    # --- Memory Cleanup (Optional) ---
    if device.type == 'cuda':
        print("Cleaning CUDA cache...")
        # del model
        # del trainer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 