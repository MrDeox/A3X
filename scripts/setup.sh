#!/bin/bash

# Script de Configuração do A³X
# ----------------------------
# Este script configura o ambiente necessário para rodar o A³X,
# incluindo dependências, modelos e estrutura de diretórios.

set -e  # Parar se houver erro

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para imprimir mensagens
print_msg() {
    echo -e "${GREEN}[A³X Setup]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[A³X Warning]${NC} $1"
}

print_error() {
    echo -e "${RED}[A³X Error]${NC} $1"
}

# Verificar Python
print_msg "Verificando instalação do Python..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 não encontrado. Por favor, instale o Python 3.8 ou superior."
    exit 1
fi

# Verificar versão do Python
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$PYTHON_VERSION < 3.8" | bc -l) )); then
    print_error "Python 3.8 ou superior é necessário. Versão atual: $PYTHON_VERSION"
    exit 1
fi

# Criar ambiente virtual
print_msg "Criando ambiente virtual..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
else
    print_warning "Ambiente virtual já existe. Ativando..."
    source .venv/bin/activate
fi

# Instalar dependências
print_msg "Instalando dependências..."
pip install --upgrade pip
pip install -r requirements.txt

# Criar estrutura de diretórios
print_msg "Criando estrutura de diretórios..."
mkdir -p models
mkdir -p a3x/memory/indexes/semantic_memory
mkdir -p logs
mkdir -p a3x_training_output/qlora_adapters

# Configurar arquivo de ambiente
print_msg "Configurando arquivo de ambiente..."
if [ ! -f ".env" ]; then
    cp config/env.example .env
    print_msg "Arquivo .env criado. Por favor, edite-o com suas configurações."
else
    print_warning "Arquivo .env já existe. Pulando..."
fi

# Verificar llama.cpp
print_msg "Verificando llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    print_msg "Clonando llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    print_msg "Compilando llama.cpp..."
    make
    cd ..
else
    print_warning "Diretório llama.cpp já existe. Pulando..."
fi

# Verificar modelo
print_msg "Verificando modelo..."
if [ ! -f "models/google_gemma-3-4b-it-Q4_K_S.gguf" ]; then
    print_warning "Modelo não encontrado em models/google_gemma-3-4b-it-Q4_K_S.gguf"
    print_msg "Por favor, baixe o modelo manualmente e coloque-o no diretório models/"
    print_msg "Link sugerido: https://huggingface.co/google/gemma-3b-it/tree/main"
fi

# Inicializar banco de dados
print_msg "Inicializando banco de dados..."
python3 -c "
from a3x.core.db_utils import initialize_database
initialize_database()
"

# Verificar instalação do firejail (opcional)
if command -v firejail &> /dev/null; then
    print_msg "Firejail encontrado. Sandbox disponível."
else
    print_warning "Firejail não encontrado. Sandbox não estará disponível."
    print_msg "Para instalar (Ubuntu/Debian): sudo apt-get install firejail"
    print_msg "Para instalar (Arch): sudo pacman -S firejail"
fi

print_msg "Configuração concluída!"
print_msg "Para começar, use: python scripts/quickstart.py --help" 