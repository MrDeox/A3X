# /home/arthur/Projects/A3X/docs/SELECTED_MODELS.md
# Modelos de IA Open Source Selecionados para A³X

Esta lista documenta os modelos e ferramentas open source escolhidos para as diferentes funcionalidades do A³X, considerando o hardware alvo (i5 11th gen, RX 6400 4GB ROCm, 16GB RAM) e a necessidade de suporte a Português/Inglês.

## 1. Geração de Texto (LLM Principal / Agente ReAct)
*   **Modelo:** **Mistral 7B Instruct** (quantizado em GGUF, ex: Q4_K_M)
*   **Justificativa:** Ótimo equilíbrio desempenho/tamanho, licença permissiva, bom suporte multilíngue, viável em hardware modesto via `llama.cpp`.
*   **Repositório/Link:** [HF (MistralAI)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1), [HF (TheBloke GGUF)](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)

## 2. Embeddings (Busca Semântica / Memória)
*   **Modelo:** **`intfloat/multilingual-e5-base`**
*   **Justificativa:** Pequeno (~110M params), rápido, nativamente multilíngue (PT/EN), forte em recuperação, compatível com `sentence-transformers` (CPU/ROCm). Dimensão 768 gerenciável.
*   **Repositório/Link:** [HF (intfloat)](https://huggingface.co/intfloat/multilingual-e5-base)

## 3. Classificação de Texto / Análise de Sentimento
*   **Modelo:** **`nlptown/bert-base-multilingual-uncased-sentiment`**
*   **Justificativa:** Modelo BERT-base já fine-tunado para análise de sentimento (5 classes) em várias línguas (incl. PT/EN). Roda bem em CPU. Alternativa: `DistilBERT` para fine-tuning customizado.
*   **Repositório/Link:** [HuggingFace](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

## 4. Agentes Autônomos (Framework)
*   **Ferramenta:** **LangChain**
*   **Justificativa:** Framework modular e flexível para criar agentes com LLMs locais e ferramentas customizadas.
*   **Repositório/Link:** [GitHub](https://github.com/hwchase17/langchain)

## 5. Geração de Imagem (Texto -> Imagem)
*   **Modelo:** **Stable Diffusion 1.5**
*   **Justificativa:** Maduro, grande comunidade, executável em 4GB VRAM (com otimizações), compatível com ROCm via `diffusers` ou SHARK.
*   **Repositório/Link:** [HF (CompVis)](https://huggingface.co/runwayml/stable-diffusion-v1-5) (Nota: v1.4 é similar, v1.5 distribuída via runwayml)

## 6. Geração de Voz (TTS)
*   **Ferramenta/Modelo:** **Piper TTS** (com modelo ONNX pt-BR)
*   **Justificativa:** Otimizado para CPU local, rápido, boa qualidade, suporte a pt-BR, fácil integração via ONNX Runtime.
*   **Repositório/Link:** [GitHub (Piper)](https://github.com/rhasspy/piper), [HF (Modelos Piper)](https://huggingface.co/rhasspy/piper-voices/tree/main)

## 7. Tradução Automática (Offline PT/EN)
*   **Ferramenta:** **Argos Translate** (biblioteca Python)
*   **Justificativa:** Solução offline completa, fácil de usar (`pip install`), otimizada para CPU, usa modelos eficientes (OPUS-MT).
*   **Repositório/Link:** [Site Oficial](https://www.argosopentech.com/), [GitHub](https://github.com/argosopentech/argos-translate)

## 8. OCR (Reconhecimento de Texto em Imagem)
*   **Ferramenta:** **Tesseract OCR** (via `pytesseract`)
*   **Justificativa:** Leve, rápido em CPU, excelente suporte a idiomas (PT/EN), bom para documentos.
*   **Repositório/Link:** [GitHub (Tesseract)](https://github.com/tesseract-ocr/tesseract), [PyPI (pytesseract)](https://pypi.org/project/pytesseract/)

## 9. Visão Computacional (Detecção de Objetos)
*   **Modelo:** **YOLOv8n** (Nano)
*   **Justificativa:** Leve (~3M params), rápido em CPU, bom desempenho para detecção em tempo real em hardware modesto.
*   **Repositório/Link:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## Llama.cpp with OpenCL/CLBlast (for AMD GPUs)

For users with AMD GPUs or those preferring OpenCL, `llama.cpp` can be compiled with CLBlast support.

### Requirements

- A C/C++ compiler (GCC/Clang)
- CMake
- `clblast-dev` library. On Debian/Ubuntu based systems, install with:
  ```bash
  sudo apt-get update && sudo apt-get install -y clblast-dev
  ```

### Building

A convenience script is provided to handle the build process:

```bash
./scripts/build_llama_cpp.sh
```

This script will:
1. Clone the `llama.cpp` repository.
2. Install `clblast-dev`.
3. Configure the build using CMake with `LLAMA_CLBLAST=ON`.
4. Compile the server binary to `llama.cpp/build/bin/server`.

### Running the Server

Use the `tools/start_llama_server.sh` script:

```bash
./tools/start_llama_server.sh
```

Before running, you might need to adjust parameters within the script:
- `MODEL_PATH`: Ensure this points to your downloaded GGUF model file.
- `GPU_LAYERS`: Adjust the number of layers offloaded to the GPU based on your VRAM.

The script starts the server on `http://127.0.0.1:8000` by default.
