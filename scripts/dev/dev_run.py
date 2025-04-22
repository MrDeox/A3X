import sys
import os
import asyncio # Import asyncio

# Adiciona a raiz do projeto ao PYTHONPATH
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Agora que o sys.path está correto, podemos importar
try:
    from a3x.a3net.run import main_loop # Importar a função main_loop
except ImportError as e:
    print(f"Erro ao importar main_loop: {e}")
    print(f"Verifique se o script está na raiz do projeto e se a estrutura a3x/a3net existe.")
    sys.exit(1)

async def run_main(): # Criar uma função async para usar await
    # O caminho para o script A3L deve ser relativo à raiz onde dev_run.py está
    a3l_script_path = "a3x/a3net/examples/teste_selfstarter.a3l"
    
    # Verificar se o arquivo A3L existe
    if not os.path.isfile(a3l_script_path):
        print(f"Erro: Arquivo A3L não encontrado em '{a3l_script_path}'")
        print(f"Certifique-se de que o arquivo existe ou ajuste o caminho.")
        sys.exit(1)
        
    print(f"--- Iniciando A³Net via dev_run.py --- ")
    print(f"Executando script: {a3l_script_path}")
    
    # Chamar a função main_loop que agora é assíncrona
    await main_loop(a3l_script_path)
    print("--- A³Net (main_loop) concluído ou interrompido ---")

if __name__ == "__main__":
    # Usar asyncio.run() para executar a função async
    try:
        asyncio.run(run_main())
    except KeyboardInterrupt:
        print("\nInterrupção pelo usuário (Ctrl+C) detectada em dev_run.py.")
    except Exception as e:
        print(f"\nErro inesperado na execução via dev_run.py: {e}", file=sys.stderr)
        # Considerar logar o traceback completo aqui se necessário
    finally:
        print("--- dev_run.py finalizado ---") 