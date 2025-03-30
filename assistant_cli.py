import sys
import os

# Adiciona o diretório raiz ao sys.path para encontrar 'cli'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
     sys.path.insert(0, project_root)

try:
    # Importa a função principal da interface
    from cli.interface import run_cli
except ImportError:
    print("Erro: Não foi possível importar 'run_cli' de 'cli.interface'. Verifique a estrutura do projeto e o PYTHONPATH.")
    sys.exit(1)

if __name__ == "__main__":
    # Chama a função principal da interface CLI
    run_cli() 