import torch.nn as nn
import logging
import asyncio
from typing import Optional, Dict, Any

# Supondo acesso ao ContextStore e MemoryBank/Registry via contexto injetado
try:
    from a3x.core.context.context_store import ContextStore
except ImportError:
    ContextStore = None
    
try:
    from a3x.a3net.integration.a3x_bridge import MEMORY_BANK # Exemplo de acesso
except ImportError:
    MEMORY_BANK = None

logger = logging.getLogger(__name__)

class SelfCriticFragment(nn.Module):
    """Analisa o desempenho histórico, detecta redundâncias e sugere otimizações."""

    def __init__(self, fragment_id: str, description: str):
        super().__init__()
        self.fragment_id = fragment_id
        self.description = description
        self.context_store: Optional[ContextStore] = None
        # Necessário para postar mensagens - será injetado
        self.post_chat_message = None
        
        logger.info(f"SelfCriticFragment '{self.fragment_id}' initialized.")

    def set_context_and_handler(self, context_store: Optional[ContextStore], message_handler):
        """Injeta dependências necessárias."""
        self.context_store = context_store
        self.post_chat_message = message_handler
        logger.info(f"SelfCritic '{self.fragment_id}' context and handler set.")

    async def perform_analysis(self):
        """Executa a análise crítica do sistema."""
        if not self.context_store or not self.post_chat_message or not MEMORY_BANK:
            logger.error(f"SelfCritic '{self.fragment_id}' missing dependencies (store, handler, or memory_bank). Analysis skipped.")
            return

        logger.info(f"SelfCritic '{self.fragment_id}' performing analysis...")

        # --- 1. Analisar Histórico de Planos (Exemplo) ---
        try:
            # Ler chaves de histórico (precisa de um método list ou scan no ContextStore)
            history_keys = []
            if self.context_store and hasattr(self.context_store, 'scan'): # Checa se o método existe
                 history_keys = await self.context_store.scan("a3l_execution_history:*")
            
            if not history_keys:
                 logger.info("SelfCritic: Nenhum histórico de execução encontrado na ContextStore para análise.")
            else:
                logger.info(f"SelfCritic: Analisando {len(history_keys)} registros de histórico...")
                # TODO: Iterar sobre history_keys, carregar cada registro
                #       Analisar os resultados dos comandos ("status", "duration_ms", "error")
                #       Agrupar por tipo de comando, fragmento alvo, etc.
                #       Identificar padrões: comandos que falham frequentemente, fragmentos lentos, etc.
                pass # Placeholder para a lógica de análise
            
        except Exception as e:
            logger.error(f"SelfCritic '{self.fragment_id}' error analyzing execution history: {e}", exc_info=True)

        # --- 2. Analisar Fragmentos (Exemplo) ---
        try:
            all_fragment_ids = MEMORY_BANK.get_all_fragment_ids()
            logger.debug(f"SelfCritic: Analisando {len(all_fragment_ids)} fragmentos...")
            # TODO: Implementar análise de redundância, uso, complexidade, etc.
            #       Exemplo: Detectar fragmentos muito similares, fragmentos não usados.
            #       Propor fusão: criar fragmento C com base em A e B
            #       Propor remoção: remover fragmento D (obsoleto)
            
            # Exemplo de sugestão (requer lógica real):
            if "frag_alpha" in all_fragment_ids and "frag_beta" in all_fragment_ids: # Condição de exemplo
                suggestion_command = "criar fragmento 'frag_alpha_beta_merged' com base em ['frag_alpha', 'frag_beta'] com objetivo de consolidar funcionalidade"
                logger.info(f"SelfCritic '{self.fragment_id}' proposing merge suggestion: {suggestion_command}")
                # Log da sugestão (sem postar comando executável)
                logger.info(f"# [SelfCritic Sugestão] {suggestion_command}. Aguardando comando A3L.")
                # Salvar sugestão na store
                if self.context_store:
                     await self.context_store.push("pending_suggestions", {"command": suggestion_command, "reason": "Análise SelfCritic - Fusão Proposta", "source": self.get_name()})

        except Exception as e:
            logger.error(f"SelfCritic '{self.fragment_id}' error analyzing fragments: {e}", exc_info=True)

        # --- 3. Lógica de Avaliação Comparativa (Placeholder) ---
        try:
            logger.debug("SelfCritic: Iniciando avaliação comparativa de fragmentos...")
            # TODO: Implementar lógica de avaliação:
            # 1. Identificar fragmentos com funcionalidade similar (ex: múltiplos file managers?)
            #    - Pode ser baseado em tipo, descrição, ou análise de código (complexo)
            # 2. Consultar histórico de execução (analisado na Etapa 1) para esses fragmentos.
            # 3. Comparar métricas (taxa de sucesso, tempo médio de execução, etc.).
            # 4. Gerar Sugestão A3L se uma otimização clara for encontrada:
            #    Ex: suggestion = "refletir sobre ['frag_A', 'frag_B'] para possível fusão devido a sobreposição"
            #    Ex: suggestion = "substituir uso de 'frag_slow' por 'frag_fast' no plano X"
            #    Logar a sugestão: logger.info(f"# [SelfCritic Sugestão] {suggestion}. Aguardando comando A3L.")
            pass # Placeholder
            
        except Exception as e:
             logger.error(f"SelfCritic '{self.fragment_id}' error during comparative evaluation: {e}", exc_info=True)

        logger.info(f"SelfCritic '{self.fragment_id}' analysis complete.")

    # --- Métodos Adicionais (Ex: ser chamado via A3L) ---
    async def handle_directive(self, directive: Dict[str, Any]):
        """Processa uma diretiva A3L direcionada a este fragmento."""
        action = directive.get("action")
        if action == "perform_analysis":
            await self.perform_analysis()
            return {"status": "success", "message": "Self-criticism analysis performed."}
        else:
            logger.warning(f"SelfCritic '{self.fragment_id}' received unknown directive action: {action}")
            return {"status": "error", "message": f"Unknown action: {action}"} 