#!/usr/bin/env bash
set -e

# Remove existing directory if it exists to ensure a clean clone
rm -rf llama.cpp

# Clone the repository
git clone https://github.com/ggerganov/llama.cpp.git --depth 1 -b master
cd llama.cpp

# Install CLBlast dependency (Debian/Ubuntu based)
# Use non-interactive to avoid prompts during installation
echo "Installing clblast-dev dependency..."
sudo apt-get update && sudo apt-get install -y clblast-dev

# Configure CMake build with CLBlast ON
echo "Configuring CMake..."
cmake -B build -DLLAMA_CLBLAST=ON -DLLAMA_OPENBLAS=OFF -DLLAMA_CUBLAS=OFF -DLLAMA_HIPBLAS=OFF

# Build the project using all available processor cores
echo "Building llama.cpp server..."
cmake --build build --config Release -j $(nproc)

echo "Build complete. Server binary should be in llama.cpp/build/bin/server" 