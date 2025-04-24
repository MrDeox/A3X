import time
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable
from a3x.core.context import SharedTaskContext

class FragmentChatManager:
    """
    Gerencia a comunicação interna entre fragments via SharedTaskContext.internal_chat_queue,
    incluindo postagem e leitura de mensagens, além de construção de prompts para LLM.
    """
    def __init__(self, logger: logging.Logger, fragment_name: str):
        self.logger = logger
        self.fragment_name = fragment_name

    async def post_chat_message(
        self,
        shared_task_context: SharedTaskContext,
        message_type: str,
        content: Dict,
        target_fragment: Optional[str] = None
    ):
        """Envia mensagem para a fila interna de chat compartilhado."""
        if not shared_task_context:
            raise RuntimeError("SharedTaskContext não disponível para chat.")
        if not shared_task_context.internal_chat_queue:
            raise RuntimeError("internal_chat_queue não definida no SharedTaskContext.")
        message = {
            "type": message_type,
            "sender": self.fragment_name,
            "content": content,
            "timestamp": time.time(),
            "target_fragment": target_fragment
        }
        try:
            await shared_task_context.internal_chat_queue.put(message)
            target_info = f"to {target_fragment}" if target_fragment else "(broadcast)"
            self.logger.info(f"[{self.fragment_name}] Posted chat message type '{message_type.upper()}' {target_info}. Subject: {content.get('subject', 'N/A')}")
        except Exception as e:
            self.logger.error(f"Error posting message to internal queue: {e}", exc_info=True)

    async def read_chat_messages(
        self,
        shared_task_context: SharedTaskContext,
        last_index: int = 0
    ) -> List[Dict]:
        """Lê mensagens da fila interna de chat a partir de um índice."""
        if not shared_task_context or not shared_task_context.internal_chat_queue:
            return []
        messages = []
        # Supondo que a fila é uma asyncio.Queue
        # Para leitura baseada em índice, seria necessário um buffer/cópia
        # Aqui apenas um placeholder para interface
        self.logger.debug("read_chat_messages: método não implementado para fila asyncio.Queue.")
        return messages

    @staticmethod
    def build_worker_messages(
        objective: str,
        history: List[Dict],
        allowed_tools: List[Callable],
        shared_task_context: Optional[SharedTaskContext] = None
    ) -> List[Dict]:
        """Monta a lista de mensagens para prompt do LLM, incluindo contexto compartilhado."""
        messages = [
            {"role": "system", "content": f"Objective: {objective}"},
            {"role": "system", "content": f"Allowed tools: {[t.__name__ for t in allowed_tools]}"}
        ]
        if shared_task_context and hasattr(shared_task_context, 'chat_context'):
            messages.append({"role": "system", "content": f"Chat context: {shared_task_context.chat_context}"})
        messages.extend(history)
        return messages
