#!/usr/bin/env python3
"""
Gera um adaptador LoRA rank‑2 com pesos zerados para google/gemma-2b‑it
e converte para GGUF via script do llama.cpp.
"""
import os, torch, tempfile, subprocess, json
from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2b-it"
OUT_DIR  = Path("lora_zero_r2_gemma2b")
GGUF_OUT = Path("lora_zero_r2_gemma2b.gguf")
CONVERT  = Path("llama.cpp/convert_lora_to_gguf.py")

OUT_DIR.mkdir(exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu")
lora_cfg = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj","v_proj"])
model = get_peft_model(model, lora_cfg)
model.save_pretrained(OUT_DIR)
# # cria weights zerados - Removido
# for p in OUT_DIR.glob("adapter_model.bin"):
#     with open(p, "wb") as f: f.write(b"\0")

# converte para GGUF
subprocess.run(
    ["python", str(CONVERT), str(OUT_DIR), "--outfile", str(GGUF_OUT)],
    check=True
)
print("LoRA GGUF salvo em", GGUF_OUT.resolve()) 