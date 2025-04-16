#!/bin/bash

# Navega para o diretório onde o script está localizado (raiz do projeto)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR" || exit 1

# Define o caminho para o diretório do ambiente virtual
VENV_DIR="venv"

# Verifica se o diretório venv existe
if [ ! -d "$VENV_DIR" ]; then
    echo "Erro: Diretório do ambiente virtual '$VENV_DIR' não encontrado em $SCRIPT_DIR."
    echo "Por favor, crie o venv ou ajuste a variável VENV_DIR no script."
    exit 1
fi

# Ativa o ambiente virtual
# O comando 'source' funciona na maioria dos shells (bash, zsh)
source "$VENV_DIR/bin/activate"

# Verifica se a ativação funcionou (opcional, checando se python vem do venv)
WHICH_PYTHON=$(which python)
if [[ "$WHICH_PYTHON" != "$SCRIPT_DIR/$VENV_DIR/bin/python"* ]]; then
    echo "Aviso: Parece que o venv não foi ativado corretamente."
    echo "Python encontrado em: $WHICH_PYTHON"
fi

echo "Ambiente virtual ativado."

# Executa o script Python do assistente, passando todos os argumentos recebidos ($@)
# echo "Executando: python a3x/assistant_cli.py $@"
# python a3x/assistant_cli.py "$@"

# CORRIGIDO: Executar o módulo interface.py que foi corrigido
echo "Executando: python -m a3x.cli.interface $@"
python -m a3x.cli.interface "$@"

# Desativa o ambiente virtual ao final (opcional, mas boa prática)
# O ambiente será desativado automaticamente quando o script terminar,
# mas 'deactivate' pode ser útil se o script for 'sourced'.
# deactivate

echo "Execução concluída." 