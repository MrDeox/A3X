import torch.nn as nn
import logging
logger = logging.getLogger(__name__)
import asyncio
import re
from typing import Optional, Dict, Any, List

# Assumir acesso ao ContextStore e funções de log/postagem
try:
    from a3x.core.context.context_store import ContextStore
except ImportError:
    ContextStore = None

# Padrão para extrair sugestões A3L de logs (simplificado)
# Ex: # [Sugestão Planner] criar fragmento 'X' ... Aguardando comando A3L.
# Ex: # [Sugestão KI via Professor Y] Extraído(s): ['comando1', 'comando2']. Aguardando comando A3L.
SUGGESTION_LOG_PATTERN = re.compile(r"^#\s*\[Sugest[ãa]o\s*([^\]]+)\]\s*(?:Extraído\(s\):)?\s*([^#\n]+?)(?:\.\s*Aguardando|\s*$)")

# Intervalo para verificar sugestões (em segundos)
SUGGESTION_CHECK_INTERVAL = 60 # 1 minuto

class ExecutorSupervisorFragment(nn.Module):
    """Monitora sugestões simbólicas logadas e decide sobre sua execução."""

    def __init__(self, fragment_id: str, description: str):
        super().__init__()
        self.fragment_id = fragment_id
        self.description = description
        self.context_store: Optional[ContextStore] = None
        self.post_chat_message = None # Injetado
        self._is_running = False
        self._task: Optional[asyncio.Task] = None
        self._processed_suggestions = set() # Evitar re-execução

        logger.info(f"ExecutorSupervisorFragment '{self.fragment_id}' initialized.")

    def set_context_and_handler(self, context_store: Optional[ContextStore], message_handler):
        """Injeta dependências."""
        self.context_store = context_store
        self.post_chat_message = message_handler
        logger.info(f"ExecutorSupervisor '{self.fragment_id}' context and handler set.")

    async def start(self):
        """Inicia o loop de verificação de sugestões."""
        if not self.post_chat_message: # ContextStore pode ser opcional por enquanto
            logger.error(f"ExecutorSupervisor '{self.fragment_id}' cannot start: message handler not set.")
            return
            
        if not self._is_running:
            self._is_running = True
            self._task = asyncio.create_task(self._suggestion_check_loop())
            logger.info(f"ExecutorSupervisor '{self.fragment_id}' started suggestion checks (Interval: {SUGGESTION_CHECK_INTERVAL}s).")

    async def stop(self):
        """Para o loop de supervisão."""
        if self._is_running and self._task:
            self._is_running = False
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.info(f"ExecutorSupervisor '{self.fragment_id}' suggestion check loop cancelled.")
            self._task = None
            logger.info(f"ExecutorSupervisor '{self.fragment_id}' stopped.")
            
    async def _read_suggestions_from_store(self) -> List[Dict[str, Any]]:
        """Lê e remove sugestões pendentes da ContextStore."""
        if not self.context_store:
            logger.warning(f"ExecutorSupervisor '{self.fragment_id}': ContextStore not available.")
            return []

        try:
            # Ler todos os itens da lista (ou um batch)
            # O método exato depende da implementação da ContextStore (get, pop_all, etc.)
            # Vamos assumir um método get que retorna a lista inteira e depois a removemos.
            pending = await self.context_store.get("pending_suggestions")
            if pending and isinstance(pending, list):
                # Limpar a lista na store para não processar novamente
                await self.context_store.delete("pending_suggestions") 
                logger.info(f"ExecutorSupervisor '{self.fragment_id}' read {len(pending)} suggestions from store.")
                return pending
            else:
                # Lista não existe ou está vazia
                return []
        except Exception as e:
            logger.error(f"ExecutorSupervisor '{self.fragment_id}' failed to read/delete pending_suggestions from store: {e}", exc_info=True)
            return []

    async def _suggestion_check_loop(self):
        """Loop que verifica e potencialmente executa sugestões."""
        while self._is_running:
            await asyncio.sleep(SUGGESTION_CHECK_INTERVAL)
            if not self._is_running: break

            logger.info(f"ExecutorSupervisor '{self.fragment_id}' checking for pending suggestions from store...")
            
            try:
                pending_suggestions_list = await self._read_suggestions_from_store() 
                
                executed_in_cycle = 0
                for suggestion_data in pending_suggestions_list:
                    # O formato esperado é {"command": "...", "reason": "...", "source": "..."}
                    suggestion_cmd = suggestion_data.get("command")
                    suggestion_reason = suggestion_data.get("reason", "N/A")
                    suggestion_source = suggestion_data.get("source", "N/A")
                    
                    if not suggestion_cmd:
                        logger.warning(f"ExecutorSupervisor found suggestion data without command: {suggestion_data}")
                        continue

                    # Usar o comando como chave para evitar reprocessamento
                    suggestion_key = suggestion_cmd # Ou gerar um hash/ID único se necessário
                    if suggestion_key in self._processed_suggestions:
                        logger.debug(f"ExecutorSupervisor skipping already processed suggestion: {suggestion_cmd}")
                        continue 

                    # --- Lógica de Decisão Simples ---
                    # TODO: Implementar lógica mais sofisticada
                    should_execute = True 

                    if should_execute:
                        logger.info(f"ExecutorSupervisor '{self.fragment_id}' deciding to execute suggestion from {suggestion_source} (Reason: {suggestion_reason}): {suggestion_cmd}")
                        try:
                            # Envia o comando A3L
                            await self.post_chat_message(
                                message_type="a3l_command", 
                                content={"command": suggestion_cmd, "origin_suggestion": suggestion_data}, # Passa a sugestão original como contexto?
                                target_fragment="Executor" 
                            )
                            self._processed_suggestions.add(suggestion_key) 
                            executed_in_cycle += 1
                            await asyncio.sleep(1) 
                        except Exception as post_err:
                             logger.error(f"ExecutorSupervisor '{self.fragment_id}' failed to post command '{suggestion_cmd}': {post_err}", exc_info=True)
                    else:
                        # Se não executar agora, a sugestão será perdida pois foi removida da store.
                        # TODO: Implementar re-agendamento ou outra lista para sugestões adiadas.
                        logger.debug(f"ExecutorSupervisor decided NOT to execute suggestion: {suggestion_cmd}")

                if executed_in_cycle > 0:
                     logger.info(f"ExecutorSupervisor '{self.fragment_id}' executed {executed_in_cycle} suggestions in this cycle.")
                else:
                     logger.info(f"ExecutorSupervisor '{self.fragment_id}' found no new suggestions to execute.")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"ExecutorSupervisor '{self.fragment_id}' error in suggestion check loop: {e}", exc_info=True)
                await asyncio.sleep(60) # Delay maior em caso de erro 