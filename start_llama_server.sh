#!/bin/bash
# Script to start llama-server with the optimal configuration based on benchmark results

# Path to the model
MODEL_PATH="/home/arthur/projects/A3X/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

# Optimal ngl value based on benchmark (CPU only for best performance)
NGL_VALUE=0

# Start the llama-server with the specified model and ngl value
echo "Starting llama-server with model $MODEL_PATH and ngl=$NGL_VALUE..."
/home/arthur/projects/A3X/llama.cpp/build/bin/llama-server -m $MODEL_PATH -ngl $NGL_VALUE

echo "llama-server started on http://127.0.0.1:8080" 