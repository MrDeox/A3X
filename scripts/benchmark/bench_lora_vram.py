#!/usr/bin/env python3
"""Benchmark how many LoRA adaptadores GGUF cabem na GPU RX‑6400
sem ROCm, usando o executável `main` do llama.cpp.

Uso:
    python scripts/bench_lora_vram.py \
        --model ggml-gemma-2b-q4_k_s.gguf \
        --rank 2 \
        --max-loras 4 

Pré‑requisitos:
1. Ter o executável `llama.cpp/main` acessível no PATH ou informar via --llama-main.
2. Ter o script `make_dummy_lora.py` (abaixo) ou as LoRAs‑dummy já geradas.
3. Rodar como usuário com permissão de leitura em /sys/class/drm/card0 (para VRAM).

O script gera (se não existirem) LoRAs vazios GGUF, carrega 1..N simultaneamente
via parâmetro `--lora` do llama.cpp e mede:
* latência de geração de 8 tokens
* VRAM em MB logo após o carregamento.

Saída: CSV na tela e salvo em ./bench_results_lora_vram.csv
"""
import argparse
import subprocess
import time
import os
import sys
import json
import shutil
from pathlib import Path
import csv

VRAM_PATH = Path("/sys/class/drm/card0/device/mem_info_vram_used")


def read_vram_mb() -> float | None:
    """Lê uso de VRAM (MB) em GPUs AMD via sysfs. Retorna None se indisponível."""
    try:
        with VRAM_PATH.open() as f:
            return int(f.read().strip()) / 1024 / 1024
    except FileNotFoundError:
        return None


def bench(model: Path, rank: int, max_loras: int, llama_main: Path, prompt: str):
    results = []
    # Usa sempre o mesmo LoRA GGUF gerado
    lora_path = Path("lora_zero_r2_gemma2b.gguf")
    if not lora_path.exists():
        print(f"Erro: Arquivo LoRA {lora_path} não encontrado!", file=sys.stderr)
        print("Execute 'python scripts/make_zero_lora.py' primeiro.", file=sys.stderr)
        sys.exit(1)

    for n in range(1, max_loras + 1):
        # cria lista com n cópias do path do LoRA
        lora_paths = [lora_path] * n
        cmd = [
            str(llama_main),
            "-m", str(model),
        ]
        for lp in lora_paths:
            cmd += ["--lora", str(lp)]
        cmd += ["-p", prompt, "-n", "8", "-ngl", "20"]

        vram_before = read_vram_mb()
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Erro executando {' '.join(cmd)}: {e}", file=sys.stderr)
            break
        latency = time.time() - t0
        vram_after = read_vram_mb()
        results.append({
            "num_loras": n,
            "rank": rank,
            "latency_s": round(latency, 3),
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after,
        })
        print(f"{n} LoRAs ▶ latency {latency:.2f}s ▶ VRAM {vram_after} MB")
    # salvar CSV
    csv_path = Path("bench_results_lora_vram.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Resultados salvos em {csv_path.resolve()}")


def cli():
    p = argparse.ArgumentParser(description="Benchmark VRAM vs Nº de LoRAs para llama.cpp")
    p.add_argument("--model", required=True, type=Path, help="Caminho do modelo GGUF base")
    p.add_argument("--rank", type=int, default=2, help="Rank das LoRAs‑dummy")
    p.add_argument("--max-loras", type=int, default=4, help="Máximo de LoRAs empilhadas no teste")
    p.add_argument("--llama-main", type=Path, default=Path("./llama.cpp/main"), help="Executável main do llama.cpp")
    p.add_argument("--prompt", default="Olá", help="Prompt curto para inferência durante o teste")
    args = p.parse_args()

    if not args.model.exists():
        print("Modelo GGUF não encontrado", file=sys.stderr)
        sys.exit(1)
    if not args.llama_main.exists():
        print("Executável llama.cpp/main não encontrado", file=sys.stderr)
        sys.exit(1)

    bench(args.model, args.rank, args.max_loras, args.llama_main, args.prompt)


if __name__ == "__main__":
    cli() 