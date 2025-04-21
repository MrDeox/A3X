#!/usr/bin/env bash

# Script to start the llama.cpp server compiled with OpenCL support

# --- Configuration --- #
# Path to the compiled server binary relative to the project root
LLAMA_BIN="./llama.cpp/build/bin/server"
# Default model path (relative to project root). Update if your model is elsewhere.
MODEL_PATH="./models/gemma-2b-q4_k.gguf" # TODO: Ensure this model exists
# Number of layers to offload to GPU. Adjust based on your GPU VRAM.
# 20 layers is an example for a ~4GB VRAM GPU like RX 6400 with a 2B Q4_K model.
GPU_LAYERS=20
# Context size for the model.
CONTEXT=4096
# Port for the server
PORT=8000
# Host for the server
HOST="127.0.0.1"
# --- End Configuration --- #

# Check if binary exists
if [ ! -f "$LLAMA_BIN" ]; then
    echo "Error: Server binary not found at $LLAMA_BIN" >&2
    echo "Please run the build script first: scripts/build_llama_cpp.sh" >&2
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH" >&2
    echo "Please ensure the model exists or update the MODEL_PATH variable in this script." >&2
    exit 1
fi

echo "Starting llama.cpp server..."
echo "Model: $MODEL_PATH"
echo "GPU Layers: $GPU_LAYERS"
echo "Context Size: $CONTEXT"
echo "Host: $HOST"
echo "Port: $PORT"

# Execute the server
"$LLAMA_BIN" \
  --model "$MODEL_PATH" \
  --port "$PORT" \
  --host "$HOST" \
  --ctx-size "$CONTEXT" \
  --mlock \
  --n-gpu-layers "$GPU_LAYERS"

echo "Server exited." 