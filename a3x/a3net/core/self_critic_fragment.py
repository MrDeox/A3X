import torch.nn as nn
import logging
import asyncio
from typing import Optional, Dict, Any, Callable, Awaitable

# Supondo acesso ao ContextStore e MemoryBank/Registry via contexto injetado
try:
    from a3x.core.context.context_store import ContextStore
except ImportError:
    # Allow running without full A3X core if needed for standalone testing
    ContextStore = None 
    logging.warning("Could not import ContextStore from a3x.core.context.context_store")
    
# try:
#     from a3x.a3net.integration.a3x_bridge import MEMORY_BANK # Exemplo de acesso
# except ImportError:
#     MEMORY_BANK = None

logger = logging.getLogger(__name__)

# Type alias for message handler
MessageHandler = Callable[[str, Dict[str, Any], str], Awaitable[None]]

class SelfCriticFragment(nn.Module):
    """Analisa o desempenho histórico, detecta redundâncias e sugere otimizações."""

    def __init__(self, fragment_id: str, 
                 description: str,
                 professor_fragment: Optional[Any] = None, # Optional Professor for deeper analysis
                 context_store: Optional[ContextStore] = None,
                 post_chat_message_callback: Optional[MessageHandler] = None):
        super().__init__()
        self.fragment_id = fragment_id
        self.description = description
        self.context_store = context_store
        self.post_chat_message = post_chat_message_callback
        self._professor = professor_fragment # Store professor reference
        
        logger.info(f"SelfCriticFragment '{self.fragment_id}' initialized.")
        if not self.context_store: logger.warning(f"SelfCritic '{self.fragment_id}' initialized without ContextStore.")
        if not self.post_chat_message: logger.warning(f"SelfCritic '{self.fragment_id}' initialized without post_chat_message callback.")
        if not self._professor: logger.warning(f"SelfCritic '{self.fragment_id}' initialized without ProfessorFragment.")

    async def analyze_performance_history(self, task_name: Optional[str] = None, fragment_id: Optional[str] = None):
        """Analisa o histórico de avaliações no ContextStore."""
        if not self.context_store:
            logger.warning(f"SelfCritic '{self.fragment_id}': Cannot analyze performance, ContextStore not available.")
            return
        
        logger.info(f"SelfCritic '{self.fragment_id}' analyzing performance history (Task: {task_name}, Fragment: {fragment_id})...")
        
        # Example: Find evaluation results
        tags_to_find = ["evaluation_result"]
        if task_name: tags_to_find.append(f"task:{task_name}")
        if fragment_id: tags_to_find.append(f"fragment:{fragment_id}")
            
        try:
            eval_keys = await self.context_store.find_keys_by_tags(tags_to_find, match_all=True)
            
            if not eval_keys:
                logger.info(f"No evaluation history found for criteria: {tags_to_find}")
                return

            logger.info(f"Found {len(eval_keys)} evaluation records. Analyzing trends...")
            # TODO: Implement more sophisticated analysis (trends, stagnation, etc.)
            # For now, just log the number of records found.
            
            # Example: Check for stagnation (e.g., accuracy not improving)
            # This requires parsing the evaluation data (assuming it has accuracy and timestamp)
            
        except Exception as e:
            logger.error(f"Error analyzing performance history: {e}", exc_info=True)

    async def suggest_optimizations(self):
        """Gera sugestões de otimização com base na análise."""
        # Placeholder for optimization suggestion logic
        logger.info(f"SelfCritic '{self.fragment_id}' generating optimization suggestions...")
        
        # Example: If stagnation detected, suggest consulting professor or trying different hyperparameters
        suggestion_a3l = None 
        if False: # Replace with actual condition based on analysis
             suggestion_a3l = "# Suggestion: Training stagnated. Consider consulting Professor.\n# aprender com 'professor_orientador' question \"Como melhorar o treino do fragmento X?\""

        if suggestion_a3l and self.post_chat_message:
            logger.info(f"Posting optimization suggestion: {suggestion_a3l}")
            await self.post_chat_message(
                 message_type="suggestion", # Or a specific type like "optimization_suggestion"
                 content={"a3l_command": suggestion_a3l, "source": self.fragment_id},
                 target_fragment="ExecutorSupervisorFragment" # Or appropriate target
            )
        else:
             logger.info("No specific optimizations suggested at this time.")

    async def run_periodic_check(self, interval_seconds: int = 300):
         """Run analysis and suggestion periodically."""
         logger.info(f"SelfCritic '{self.fragment_id}' starting periodic check loop (Interval: {interval_seconds}s)...")
         while True:
             try:
                 await self.analyze_performance_history()
                 await self.suggest_optimizations()
                 await asyncio.sleep(interval_seconds)
             except asyncio.CancelledError:
                 logger.info(f"SelfCritic '{self.fragment_id}' periodic check cancelled.")
                 break
             except Exception as e:
                  logger.error(f"Error in SelfCritic periodic check: {e}", exc_info=True)
                  # Avoid tight loop on error
                  await asyncio.sleep(interval_seconds * 2)

    async def handle_message(self, message_type: str, content: Any, **kwargs):
         """Handles incoming messages, e.g., requests for analysis."""
         logger.debug(f"SelfCritic '{self.fragment_id}' received message: Type={message_type}, Content={str(content)[:100]}")
         if message_type == "request_analysis":
             task = content.get("task_name")
             frag = content.get("fragment_id")
             await self.analyze_performance_history(task_name=task, fragment_id=frag)
             await self.suggest_optimizations()
         else:
             logger.warning(f"Unhandled message type for SelfCritic: {message_type}")

    async def perform_analysis(self):
        """Executa a análise crítica do sistema."""
        if not self.context_store or not self.post_chat_message:
            logger.error(f"SelfCritic '{self.fragment_id}' missing dependencies (store or handler). Analysis skipped.")
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