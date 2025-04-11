import requests
import sys

# --- Configuração ---
IMAGE_PATH = "dog.jpg"  # Ou dog.jpg, ou o caminho completo
PROMPT_TEXT = "USER: <image>\nDescribe this image.\nASSISTANT:" # LLaVA 1.5 Prompt Format
SERVER_URL = "http://localhost:8080/completion"
# --- Fim Configuração ---

# Verifica se o arquivo de imagem existe
try:
    with open(IMAGE_PATH, "rb") as f:
        image_content = f.read()
except FileNotFoundError:
    print(f"Erro: Arquivo de imagem não encontrado em '{IMAGE_PATH}'")
    sys.exit(1)
except Exception as e:
    print(f"Erro ao ler o arquivo de imagem: {e}")
    sys.exit(1)

# Prepara os dados multipart
# Note que 'files' é usado para o arquivo, e 'data' para outros campos
files = {
    'file0': (IMAGE_PATH, image_content, 'image/jpeg') # Envia como image/jpeg, ajuste se necessário
}
data = {
    'prompt': PROMPT_TEXT,
    'n_predict': 128  # Limitar tokens de saída
}

print(f"Enviando requisição para {SERVER_URL} com a imagem {IMAGE_PATH} e n_predict=128...")

try:
    response = requests.post(SERVER_URL, files=files, data=data)
    response.raise_for_status()  # Levanta exceção para erros HTTP (4xx ou 5xx)

    # Imprime a resposta
    print("\n--- Resposta do Servidor ---")
    try:
        # Tenta imprimir como JSON se possível
        print(response.json())
    except requests.exceptions.JSONDecodeError:
        # Imprime como texto se não for JSON
        print(response.text)
    print("---------------------------\n")

except requests.exceptions.ConnectionError as e:
    print(f"\nErro de conexão: Não foi possível conectar ao servidor em {SERVER_URL}")
    print(f"Verifique se o servidor llama-server está rodando.")
    print(f"Detalhes: {e}")
except requests.exceptions.Timeout as e:
    print(f"\nErro: Tempo limite da requisição excedido para {SERVER_URL}")
    print(f"Detalhes: {e}")
except requests.exceptions.HTTPError as e:
    print(f"\nErro HTTP {response.status_code} recebido do servidor:")
    try:
        print(response.json()) # Tenta mostrar erro JSON do servidor
    except:
        print(response.text)
    print(f"Detalhes: {e}")
except requests.exceptions.RequestException as e:
    print(f"\nOcorreu um erro durante a requisição:")
    print(f"Detalhes: {e}")
except Exception as e:
    print(f"\nOcorreu um erro inesperado:")
    print(f"Detalhes: {e}") 