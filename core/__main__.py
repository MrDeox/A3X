"""
Módulo principal do A³X.
Permite a execução direta do sistema através do comando python -m core.
"""

import logging
import sys
from typing import Optional
from .executor import Executor

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configura o logging do sistema.
    
    Args:
        level: Nível de logging
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main() -> None:
    """
    Função principal que executa o sistema em modo interativo.
    """
    # Configura logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Inicializa o executor
    executor = Executor(logger)
    logger.info("A³X iniciado. Digite 'sair' para encerrar.")
    
    # Loop principal
    while True:
        try:
            # Lê comando do usuário
            command = input("\nA³X> ").strip()
            
            # Verifica se deve sair
            if command.lower() in ('sair', 'exit', 'quit'):
                logger.info("Encerrando A³X...")
                break
                
            # Processa o comando
            result = executor.process_command(command)
            
            # Exibe resultado
            if result['status'] == 'success':
                print(result['response'])
            else:
                print(f"Erro: {result['error']}")
                
        except KeyboardInterrupt:
            logger.info("\nEncerrando A³X...")
            break
        except Exception as e:
            logger.error(f"Erro inesperado: {str(e)}")
            print(f"Erro: {str(e)}")

if __name__ == "__main__":
    main() 