import torch.nn as nn
import logging
import asyncio
import random
from typing import Optional, List, Dict, Callable, Any, Awaitable

# Assuming MemoryBank is available for import
from .memory_bank import MemoryBank

logger = logging.getLogger(__name__)

# Intervalo para verificações periódicas (em segundos)
PERIODIC_CHECK_INTERVAL = 300 # 5 minutos (ajustar conforme necessário)
REFLECTION_PROBABILITY = 0.1 # Probabilidade de refletir sobre um fragmento aleatório a cada ciclo

# Type alias for message handler
MessageHandler = Callable[[str, Dict[str, Any], str], Awaitable[None]]

class SupervisorFragment(nn.Module):
    """Supervisiona o sistema, dispara verificações periódicas e reflexões."""
    
    def __init__(self, fragment_id: str, description: str, memory_bank: Optional[MemoryBank] = None, message_handler: Optional[MessageHandler] = None):
        super().__init__()
        self.fragment_id = fragment_id
        self.description = description
        self._is_running = False
        self._task: Optional[asyncio.Task] = None
        self.post_chat_message = message_handler
        self.memory_bank = memory_bank
        
        logger.info(f"SupervisorFragment '{self.fragment_id}' initialized.")
        if not self.memory_bank: logger.warning(f"Supervisor '{self.fragment_id}' initialized without MemoryBank.")
        if not self.post_chat_message: logger.warning(f"Supervisor '{self.fragment_id}' initialized without message_handler.")

    async def start(self, interval_seconds: int = 60):
        """Inicia a tarefa de verificação periódica em background."""
        if self._task is None or self._task.done():
            logger.info(f"Starting Supervisor '{self.fragment_id}' task...")
            self._task = asyncio.create_task(self._periodic_check_loop(interval_seconds))
            # Opcional: Adicionar callback para logar quando a task terminar (normal ou erro)
            # self._task.add_done_callback(lambda t: logger.info(f"Supervisor task finished: {t}"))
        else:
            logger.warning(f"Supervisor '{self.fragment_id}' task already running.")

    async def stop(self):
        """Para a tarefa de verificação periódica."""
        if self._task and not self._task.done():
            logger.info(f"Stopping Supervisor '{self.fragment_id}' task...")
            self._task.cancel()
            try:
                await self._task # Esperar o cancelamento concluir
            except asyncio.CancelledError:
                logger.info(f"Supervisor '{self.fragment_id}' task successfully cancelled.")
            self._task = None
        else:
            logger.info(f"Supervisor '{self.fragment_id}' task not running or already stopped.")

    async def _periodic_check_loop(self, interval_seconds: int):
        """Loop principal de verificação periódica."""
        logger.info(f"Supervisor '{self.fragment_id}' starting periodic check loop (Interval: {interval_seconds}s)...")
        while True:
            await asyncio.sleep(interval_seconds)
            logger.debug(f"Supervisor '{self.fragment_id}' performing periodic check...")
            try:
                # Exemplo: Escolher um fragmento aleatório para refletir
                if not self.memory_bank:
                     logger.warning(f"Supervisor '{self.fragment_id}': MemoryBank not available, cannot select fragment for reflection.")
                     continue # Skip this cycle if no memory bank
                     
                all_fragments = self.memory_bank.list_fragments()
                if all_fragments:
                    target_fragment_id = random.choice(list(all_fragments.keys()))
                    logger.info(f"Supervisor '{self.fragment_id}' initiating reflection for random fragment: '{target_fragment_id}'")
                    
                    # Criar comando A3L
                    a3l_command = f"refletir sobre fragmento '{target_fragment_id}'"
                    directive_content = {"a3l_command": a3l_command, "source": self.fragment_id}
                    
                    # Enviar comando via handler (se disponível)
                    if self.post_chat_message:
                         await self.post_chat_message(
                             message_type="suggestion", # Ou "command" dependendo do fluxo
                             content=directive_content,
                             target_fragment="ExecutorSupervisorFragment" # Ou outro alvo apropriado
                         )
                    else:
                        logger.warning(f"Supervisor '{self.fragment_id}': No message handler set, cannot post reflection suggestion.")
                else:
                    logger.info(f"Supervisor '{self.fragment_id}': No fragments found in MemoryBank to reflect upon.")
            except Exception as e:
                logger.error(f"Error during Supervisor periodic check: {e}", exc_info=True)

    async def handle_message(self, message_type: str, content: Any, **kwargs):
         """Handles incoming messages (if supervisor needs to react)."""
         logger.debug(f"Supervisor '{self.fragment_id}' received message: Type={message_type}, Content={str(content)[:100]}")
         # Placeholder: Supervisor might react to system events, errors, etc.
         pass 