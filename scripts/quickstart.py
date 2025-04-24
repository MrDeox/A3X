#!/usr/bin/env python3
"""
Script de Inicialização Rápida do A³X
------------------------------------

Este script facilita o início rápido do sistema A³X, configurando
automaticamente o ambiente e iniciando os componentes necessários.

Uso:
    python quickstart.py [--task TASK] [--autonomous] [--workspace PATH]
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Adicionar diretório raiz ao PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from a3x.core.a3x_unified import create_a3x_system
from a3x.core.config import (
    PROJECT_ROOT,
    LLAMA_SERVER_URL,
    LLAMA_SERVER_BINARY,
    LLAMA_SERVER_ARGS,
    DATABASE_PATH,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('a3x_quickstart.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Processa argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Script de inicialização rápida do A³X"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Tarefa inicial para executar"
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Inicia o ciclo autônomo após a tarefa inicial"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="Caminho para o workspace (opcional)"
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Não inicia o servidor LLM automaticamente"
    )
    return parser.parse_args()

async def check_environment():
    """Verifica e configura o ambiente necessário."""
    logger.info("Verificando ambiente...")
    
    # Verificar diretório de modelos
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists():
        logger.info("Criando diretório de modelos...")
        models_dir.mkdir(parents=True)
    
    # Verificar diretório de memória
    memory_dir = Path(DATABASE_PATH).parent
    if not memory_dir.exists():
        logger.info("Criando diretório de memória...")
        memory_dir.mkdir(parents=True)
    
    # Verificar servidor LLM
    if not Path(LLAMA_SERVER_BINARY).exists():
        logger.warning(
            f"Servidor LLM não encontrado em {LLAMA_SERVER_BINARY}. "
            "Por favor, instale o servidor llama.cpp manualmente."
        )
        return False
    
    return True

async def main():
    """Função principal do script de quickstart."""
    args = parse_args()
    
    try:
        # Verificar ambiente
        if not await check_environment():
            logger.error("Ambiente não está configurado corretamente.")
            sys.exit(1)
        
        # Definir workspace
        workspace_root = Path(args.workspace) if args.workspace else PROJECT_ROOT
        
        # Criar e inicializar sistema
        logger.info("Inicializando sistema A³X...")
        system = await create_a3x_system(
            workspace_root=workspace_root
        )
        
        try:
            # Executar tarefa inicial se especificada
            if args.task:
                logger.info(f"Executando tarefa: {args.task}")
                result = await system.execute_task(args.task)
                print("\nResultado da tarefa:")
                print("=" * 40)
                print(result)
                print("=" * 40)
            
            # Iniciar ciclo autônomo se solicitado
            if args.autonomous:
                logger.info("Iniciando ciclo autônomo...")
                await system.start_autonomous_cycle(
                    initial_goal="Aprender e otimizar o sistema continuamente"
                )
            
            # Se nenhuma opção foi especificada, mostrar ajuda
            if not args.task and not args.autonomous:
                print(__doc__)
                print("\nUse --help para ver todas as opções disponíveis.")
        
        finally:
            # Garantir limpeza adequada
            await system.cleanup()
    
    except KeyboardInterrupt:
        logger.info("\nOperação interrompida pelo usuário.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Erro durante execução:")
        sys.exit(1)

if __name__ == "__main__":
    # Rodar o quickstart
    asyncio.run(main()) 