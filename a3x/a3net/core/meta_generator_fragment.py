import torch.nn as nn
import logging
import asyncio
import random
from typing import Optional, List

logger = logging.getLogger(__name__)

# Intervalo para gerar novos objetivos (em segundos)
META_GENERATION_INTERVAL = 600 # 10 minutos

# Pool de possíveis objetivos/perguntas internas
INTERNAL_GOAL_POOL = [
    "aprender com 'Como posso melhorar a eficiência geral do planejamento A3Net?'",
    "aprender com 'Quais são as melhores práticas para refatorar fragmentos redundantes?'",
    "refletir sobre todos os fragmentos tipo 'PlannerFragment' para comparar estratégias",
    "refletir sobre log de execução recente para identificar gargalos",
    "aprender com 'Como detectar e corrigir loops infinitos em scripts A3L?'",
    "refletir sobre o processo de auto-critica para identificar vieses"
]

class MetaGeneratorFragment(nn.Module):
    """Gera periodicamente novos objetivos internos ou questões de reflexão para o sistema."""

    def __init__(self, fragment_id: str, description: str):
        super().__init__()
        self.fragment_id = fragment_id
        self.description = description
        self.post_chat_message = None # Injetado
        self._is_running = False
        self._task: Optional[asyncio.Task] = None
        
        logger.info(f"MetaGeneratorFragment '{self.fragment_id}' initialized.")
        
    def set_message_handler(self, handler):
        """Injeta o handler de mensagens."""
        self.post_chat_message = handler
        logger.info(f"MetaGenerator '{self.fragment_id}' message handler set.")
        
    async def start(self):
        """Inicia o loop de geração de meta-objetivos."""
        if not self.post_chat_message:
            logger.error(f"MetaGenerator '{self.fragment_id}' cannot start: message handler not set.")
            return
            
        if not self._is_running:
            self._is_running = True
            self._task = asyncio.create_task(self._generation_loop())
            logger.info(f"MetaGenerator '{self.fragment_id}' started internal goal generation (Interval: {META_GENERATION_INTERVAL}s).")

    async def stop(self):
        """Para o loop de geração."""
        if self._is_running and self._task:
            self._is_running = False
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.info(f"MetaGenerator '{self.fragment_id}' generation loop cancelled.")
            self._task = None
            logger.info(f"MetaGenerator '{self.fragment_id}' stopped.")
            
    async def _generation_loop(self):
        """Loop que gera e posta os objetivos/perguntas."""
        while self._is_running:
            await asyncio.sleep(META_GENERATION_INTERVAL)
            if not self._is_running: break
            
            try:
                # Escolher um objetivo aleatório do pool
                chosen_goal_command = random.choice(INTERNAL_GOAL_POOL)
                
                logger.info(f"MetaGenerator '{self.fragment_id}' generating internal goal: {chosen_goal_command}")
                
                # Postar como um comando A3L para o Executor
                await self.post_chat_message(
                     message_type="a3l_command", 
                     content={"command": chosen_goal_command},
                     target_fragment="Executor" # Ou quem processa comandos A3L
                 )
                 # TODO: Adicionar log simbólico se necessário
                 
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"MetaGenerator '{self.fragment_id}' error in generation loop: {e}", exc_info=True)
                await asyncio.sleep(60) 