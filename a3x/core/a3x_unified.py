"""
A³X Unified System
-----------------
Este módulo unifica todos os componentes do A³X em um sistema coeso,
integrando a parte simbólica (A3L) com a neural (A3Net).
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import signal
import sys

# Componentes Core
from a3x.core.config import (
    PROJECT_ROOT,
    LLAMA_SERVER_URL,
    LLAMA_SERVER_BINARY,
    LLAMA_SERVER_ARGS,
    DATABASE_PATH,
    MEMORY_DIR,
)
from a3x.core.orchestrator import TaskOrchestrator
from a3x.core.llm_interface import LLMInterface
from a3x.core.server_manager import start_all_servers, stop_all_servers
from a3x.core.tool_registry import ToolRegistry
from a3x.core.skills import discover_skills, SKILL_REGISTRY
from a3x.core.db_utils import initialize_database, close_db_connection

# Componentes A3Net
from a3x.a3net.core.knowledge_interpreter_fragment import KnowledgeInterpreterFragment
from a3x.a3net.core.professor_llm_fragment import ProfessorLLMFragment
from a3x.a3net.fragments.autonomous_self_starter import AutonomousSelfStarterFragment

# Componentes de Memória e Contexto
from a3x.core.memory.memory_manager import MemoryManager
from a3x.core.context import SharedTaskContext
from a3x.fragments.registry import FragmentRegistry

# Logging
logger = logging.getLogger(__name__)

class A3XUnified:
    """
    Sistema A³X Unificado que integra todos os componentes em uma interface coesa.
    """
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or PROJECT_ROOT
        self.logger = logging.getLogger("A3XUnified")
        
        # Componentes principais (inicializados em setup())
        self.llm_interface: Optional[LLMInterface] = None
        self.tool_registry: Optional[ToolRegistry] = None
        self.fragment_registry: Optional[FragmentRegistry] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.orchestrator: Optional[TaskOrchestrator] = None
        
        # Fragments essenciais
        self.professor_fragment: Optional[ProfessorLLMFragment] = None
        self.knowledge_interpreter: Optional[KnowledgeInterpreterFragment] = None
        self.autonomous_starter: Optional[AutonomousSelfStarterFragment] = None
        
        # Estado do sistema
        self.is_initialized = False
        self.servers_started = False

    async def setup(self):
        """Inicializa todos os componentes do sistema."""
        if self.is_initialized:
            return
            
        try:
            # 1. Inicializar banco de dados
            self.logger.info("Inicializando banco de dados...")
            initialize_database()
            
            # 2. Iniciar servidores necessários
            self.logger.info("Iniciando servidores...")
            self.servers_started = await start_all_servers()
            if self.servers_started:
                # Registrar cleanup
                signal.signal(signal.SIGTERM, lambda sig, frame: self.cleanup())
                signal.signal(signal.SIGINT, lambda sig, frame: self.cleanup())
            
            # 3. Inicializar interface LLM
            self.logger.info("Inicializando interface LLM...")
            self.llm_interface = LLMInterface(
                llm_url=LLAMA_SERVER_URL,
                model_name="gemma-3-4b-it",
                context_size=4096,
                timeout=600
            )
            
            # 4. Descobrir e registrar skills
            self.logger.info("Carregando skills...")
            discover_skills()
            self.tool_registry = ToolRegistry()
            for skill_name, skill_info in SKILL_REGISTRY.items():
                self.tool_registry.register_tool(
                    name=skill_name,
                    instance=None,
                    tool=skill_info["function"],
                    schema=skill_info.get("schema", {})
                )
            
            # 5. Inicializar gerenciador de memória
            self.logger.info("Inicializando gerenciador de memória...")
            self.memory_manager = MemoryManager(
                db_path=DATABASE_PATH,
                memory_dir=MEMORY_DIR
            )
            
            # 6. Inicializar registro de fragments
            self.logger.info("Inicializando registro de fragments...")
            self.fragment_registry = FragmentRegistry(skill_registry=self.tool_registry)
            self.fragment_registry.discover_and_register_fragments()
            
            # 7. Inicializar fragments essenciais
            self.logger.info("Inicializando fragments essenciais...")
            self.professor_fragment = ProfessorLLMFragment(
                fragment_id="professor_main",
                description="Professor LLM principal do sistema",
                llm_url=LLAMA_SERVER_URL
            )
            self.knowledge_interpreter = KnowledgeInterpreterFragment(
                fragment_id="knowledge_interpreter_main",
                description="Interpretador principal de conhecimento",
                professor_fragment=self.professor_fragment
            )
            self.autonomous_starter = AutonomousSelfStarterFragment(
                fragment_id="autonomous_starter_main",
                description="Iniciador de ciclos autônomos"
            )
            
            # 8. Inicializar orquestrador
            self.logger.info("Inicializando orquestrador...")
            self.orchestrator = TaskOrchestrator(
                fragment_registry=self.fragment_registry,
                tool_registry=self.tool_registry,
                memory_manager=self.memory_manager,
                llm_interface=self.llm_interface,
                workspace_root=self.workspace_root,
                agent_logger=self.logger
            )
            
            self.is_initialized = True
            self.logger.info("Sistema A³X inicializado com sucesso!")
            
        except Exception as e:
            self.logger.exception("Erro durante inicialização do A³X:")
            await self.cleanup()
            raise RuntimeError(f"Falha na inicialização do A³X: {e}")

    async def execute_task(self, objective: str, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Executa uma tarefa usando o orquestrador."""
        if not self.is_initialized:
            raise RuntimeError("Sistema não inicializado. Chame setup() primeiro.")
            
        self.logger.info(f"Executando tarefa: {objective}")
        try:
            result = await self.orchestrator.orchestrate(
                objective=objective,
                max_steps=max_steps
            )
            return result
        except Exception as e:
            self.logger.exception("Erro durante execução da tarefa:")
            return {
                "status": "error",
                "reason": "execution_error",
                "message": f"Erro durante execução: {e}"
            }

    async def start_autonomous_cycle(self, initial_goal: Optional[str] = None) -> None:
        """Inicia um ciclo autônomo de aprendizado e execução."""
        if not self.is_initialized:
            raise RuntimeError("Sistema não inicializado. Chame setup() primeiro.")
            
        self.logger.info("Iniciando ciclo autônomo...")
        try:
            # Criar contexto compartilhado para o ciclo
            shared_context = SharedTaskContext(
                task_id="autonomous_cycle",
                objective=initial_goal or "Aprender e evoluir autonomamente",
                memory_manager=self.memory_manager
            )
            
            # Iniciar ciclo através do AutonomousSelfStarter
            await self.autonomous_starter.execute(
                ctx=shared_context,
                args={"initial_goal": initial_goal} if initial_goal else {}
            )
        except Exception as e:
            self.logger.exception("Erro durante ciclo autônomo:")
            raise

    async def cleanup(self):
        """Limpa recursos e finaliza componentes."""
        self.logger.info("Iniciando limpeza do sistema...")
        
        # Parar orquestrador
        if self.orchestrator:
            try:
                await self.orchestrator.shutdown()
            except Exception as e:
                self.logger.error(f"Erro ao desligar orquestrador: {e}")
        
        # Fechar conexão com banco de dados
        try:
            close_db_connection()
        except Exception as e:
            self.logger.error(f"Erro ao fechar conexão com banco de dados: {e}")
        
        # Parar servidores se foram iniciados por nós
        if self.servers_started:
            try:
                stop_all_servers()
            except Exception as e:
                self.logger.error(f"Erro ao parar servidores: {e}")
        
        self.is_initialized = False
        self.logger.info("Limpeza do sistema concluída.")

# Função helper para uso do sistema
async def create_a3x_system(workspace_root: Optional[Path] = None) -> A3XUnified:
    """
    Cria e inicializa uma instância do sistema A³X unificado.
    
    Args:
        workspace_root: Diretório raiz do workspace (opcional)
        
    Returns:
        Sistema A³X inicializado e pronto para uso
    """
    system = A3XUnified(workspace_root=workspace_root)
    await system.setup()
    return system

# Exemplo de uso:
if __name__ == "__main__":
    async def main():
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
        )
        
        try:
            # Criar e inicializar sistema
            system = await create_a3x_system()
            
            # Exemplo: executar uma tarefa
            result = await system.execute_task(
                "Analisar a estrutura atual do projeto e sugerir melhorias."
            )
            print(f"Resultado da tarefa: {result}")
            
            # Exemplo: iniciar ciclo autônomo
            await system.start_autonomous_cycle()
            
        except Exception as e:
            print(f"Erro durante execução: {e}")
            raise
        finally:
            # Garantir limpeza adequada
            if 'system' in locals():
                await system.cleanup()

    # Rodar o exemplo
    asyncio.run(main()) 