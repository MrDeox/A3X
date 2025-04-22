# sandbox_hello.py
import requests
import os
from pathlib import Path
import sys

print("Olá do sandbox!")
print(f"Executando dentro do Python: {sys.executable}")
print(f"Diretório atual: {os.getcwd()}")

# 1. Attempt network access (expected to fail)
try:
    print("Tentando acessar a rede (https://httpbin.org/get)...")
    # Use um timeout curto
    response = requests.get("https://httpbin.org/get", timeout=5)
    # Se chegar aqui, é inesperado
    print(f"!!! ALERTA: Acesso à rede SUCEDEU (inesperado!). Status: {response.status_code}")
    sys.exit(1) # Falha o script se a rede funcionar
except requests.exceptions.RequestException as e:
    # Esta é a exceção esperada devido ao --net=none
    print(f"--- Acesso à rede FALHOU (esperado): {type(e).__name__}: {e}")
except Exception as e:
    # Captura outros erros inesperados
    print(f"!!! ALERTA: Erro inesperado ao acessar a rede: {type(e).__name__}: {e}")
    sys.exit(1)

# 2. Attempt file write (expected to succeed inside the sandbox mapped dir)
# O diretório atual deve ser /sandbox, que é o root do projeto mapeado
file_path = Path("teste_sandbox.txt")
try:
    abs_path = file_path.resolve()
    print(f"Tentando escrever em {abs_path}...")
    with open(file_path, "w") as f:
        f.write("Escrito de dentro do sandbox!")
    print(f"--- Escrita em {abs_path} SUCEDEU (esperado).")
    # Verifica se o arquivo existe após escrever
    if file_path.exists():
        print(f"--- Verificação: Arquivo {abs_path} existe após escrita (esperado).")
    else:
        print(f"!!! ALERTA: Arquivo {abs_path} NÃO existe após escrita (inesperado).")
        sys.exit(1)
except Exception as e:
    print(f"!!! ALERTA: Escrita em {abs_path} FALHOU (inesperado): {type(e).__name__}: {e}")
    sys.exit(1)

print("--- Fim do script sandbox_hello.py (Sucesso) ---")
# Saída com código 0 indica sucesso
sys.exit(0) 