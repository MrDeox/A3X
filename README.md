# A³X - Sistema de Inteligência Artificial Local

Sistema modular de inteligência artificial local com suporte a GPU AMD via ROCm.

## 🚀 Características

- Execução local de modelos GGUF
- Suporte a GPU AMD via ROCm
- Interface Python simples e modular
- Baseado no llama.cpp

## 🛠️ Requisitos

- Python 3.8+
- GPU AMD com suporte a ROCm
- llama.cpp compilado com suporte a ROCm

## 📦 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/MrDeox/A3X.git
cd A3X
```

2. Compile o llama.cpp com suporte a ROCm:
```bash
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_HIPBLAS=ON
cmake --build . --config Release
cd ../..
```

3. Baixe o modelo (opcional, se não tiver):
```bash
# O modelo será baixado automaticamente na primeira execução
```

## 💡 Uso

```python
from llm.inference import run_llm

# Exemplo simples
resposta = run_llm("Qual é a sua missão?")
print(resposta)

# Com número máximo de tokens personalizado
resposta = run_llm("Explique o que é inteligência artificial", max_tokens=256)
print(resposta)
```

## 🏗️ Estrutura do Projeto

```
A3X/
├── llm/                    # Módulo Python para inferência
│   ├── __init__.py
│   └── inference.py
├── llama.cpp/             # Submódulo llama.cpp
├── models/                # Diretório para modelos GGUF
└── README.md
```

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, leia o [CONTRIBUTING.md](CONTRIBUTING.md) para detalhes sobre nosso código de conduta e o processo para enviar pull requests. 