import torch.nn as nn
import logging
import asyncio
import random
from typing import Optional, List, Dict

# Supondo que a memória/registro esteja acessível via bridge ou contexto
from a3x.a3net.integration.a3x_bridge import MEMORY_BANK

logger = logging.getLogger(__name__)

# Intervalo para verificações periódicas (em segundos)
PERIODIC_CHECK_INTERVAL = 300 # 5 minutos (ajustar conforme necessário)
REFLECTION_PROBABILITY = 0.1 # Probabilidade de refletir sobre um fragmento aleatório a cada ciclo

class SupervisorFragment(nn.Module):
    """Supervisiona o sistema, dispara verificações periódicas e reflexões."""
    
    def __init__(self, fragment_id: str, description: str):
        super().__init__()
        self.fragment_id = fragment_id
        self.description = description
        self._is_running = False
        self._task: Optional[asyncio.Task] = None
        # Necessário para postar mensagens - será injetado
        self.post_chat_message = None 
        
        logger.info(f"SupervisorFragment '{self.fragment_id}' initialized.")

    def set_message_handler(self, handler):
        """Injeta a função para postar mensagens."""
        self.post_chat_message = handler
        logger.info(f"Supervisor '{self.fragment_id}' message handler set.")

    async def start(self):
        """Inicia o loop de supervisão periódica."""
        if not self.post_chat_message:
            logger.error(f"Supervisor '{self.fragment_id}' cannot start: message handler not set.")
            return
            
        if not self._is_running:
            self._is_running = True
            self._task = asyncio.create_task(self._periodic_check_loop())
            logger.info(f"Supervisor '{self.fragment_id}' started periodic checks (Interval: {PERIODIC_CHECK_INTERVAL}s).")

    async def stop(self):
        """Para o loop de supervisão."""
        if self._is_running and self._task:
            self._is_running = False
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.info(f"Supervisor '{self.fragment_id}' periodic check loop cancelled.")
            self._task = None
            logger.info(f"Supervisor '{self.fragment_id}' stopped.")

    async def _periodic_check_loop(self):
        """Loop que executa as verificações periodicamente."""
        while self._is_running:
            try:
                await asyncio.sleep(PERIODIC_CHECK_INTERVAL)
                if not self._is_running: break # Checa novamente após o sleep
                
                logger.info(f"Supervisor '{self.fragment_id}' performing periodic check...")
                
                # --- Lógica de Reflexão Periódica Aleatória ---
                if random.random() < REFLECTION_PROBABILITY:
                    all_fragments = MEMORY_BANK.get_all_fragment_ids() # Precisa existir esse método
                    if all_fragments:
                        chosen_fragment_id = random.choice(all_fragments)
                        logger.info(f"Supervisor '{self.fragment_id}' triggering random reflection for '{chosen_fragment_id}'.")
                        
                        reflection_command = f"refletir fragmento '{chosen_fragment_id}' como a3l com objetivo de encontrar alternativa mais eficiente"
                        try:
                            # Envia o comando para ser executado pelo Executor
                            await self.post_chat_message(
                                message_type="a3l_command", 
                                content={"command": reflection_command},
                                target_fragment="Executor" # Ou quem processa comandos A3L
                            )
                            logger.info(f"# [Supervisor] Iniciada reflexão periódica para '{chosen_fragment_id}'. Comando: {reflection_command}")
                        except Exception as post_err:
                            logger.error(f"Supervisor '{self.fragment_id}' failed to post reflection command: {post_err}", exc_info=True)
                
                # --- Adicionar outras verificações periódicas aqui --- 
                # Ex: Verificar integridade, saúde do sistema, etc.
                
            except asyncio.CancelledError:
                raise # Propaga o cancelamento
            except Exception as e:
                logger.error(f"Supervisor '{self.fragment_id}' encountered error in periodic loop: {e}", exc_info=True)
                # Espera um pouco antes de tentar novamente para evitar spam em caso de erro persistente
                await asyncio.sleep(60) 