import logging
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
try:
    import torch
except ImportError:  # Provide lightweight fallback when torch isn't available
    torch = None
import zipfile
import tempfile
import os
from a3x.core.config import LEARNING_LOGS_DIR

class MemoryManager:
    FRAGMENT_STATE_DIR = "fragment_states"
    FRAGMENT_EXPORT_DIR = "fragment_exports"

    """
    Abstrai o acesso à memória semântica (FAISS) e episódica (SQLite),
    e gerencia metadados de fragmentos.
    Permite futura substituição de backends sem alterar o código do agente.
    """

    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        # Diretórios para estados e exportações
        self.state_dir = Path(config.get("MEMORY_STATE_DIR", self.FRAGMENT_STATE_DIR))
        self.export_dir = Path(config.get("MEMORY_EXPORT_DIR", self.FRAGMENT_EXPORT_DIR))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.config = config
        # Importações locais para facilitar substituição futura
        from a3x.core.embeddings import get_embedding
        from a3x.core.semantic_memory_backend import search_index
        from a3x.core.db_utils import (
            add_episodic_record,
            retrieve_recent_episodes,
        )
        self.get_embedding = get_embedding
        self.search_index = search_index
        self.add_episodic_record = add_episodic_record
        self.retrieve_recent_episodes = retrieve_recent_episodes

        self.semantic_index_path = config.get("SEMANTIC_INDEX_PATH")
        self.semantic_top_k = config.get("SEMANTIC_SEARCH_TOP_K", 5)
        self.episodic_limit = config.get("EPISODIC_RETRIEVAL_LIMIT", 5)

        # Added: Fragment Metadata Store
        self.fragment_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_fragment_metadata() # Attempt to load initial metadata

    # --- FRAGMENT STATE MANAGEMENT (NEURAL/INTERNAL) ---
    def save_fragment_state(self, fragment_id: str, state_data: Dict[str, Any]):
        """
        Salva o estado neural (.pt) de um fragmento.
        """
        state_file = self.state_dir / f"{fragment_id}.pt"
        if torch:
            torch.save(state_data, state_file)
        else:
            with open(state_file, "w") as f:
                json.dump(state_data, f)
        self.logger.info(f"[MemoryManager] Saved fragment state for '{fragment_id}' at {state_file}")

    def load_fragment_state(self, fragment_id: str) -> Optional[Dict[str, Any]]:
        """
        Carrega o estado neural (.pt) de um fragmento.
        """
        state_file = self.state_dir / f"{fragment_id}.pt"
        if not state_file.exists():
            self.logger.warning(f"[MemoryManager] State file not found for fragment '{fragment_id}': {state_file}")
            return None
        if torch:
            state_data = torch.load(state_file)
        else:
            with open(state_file, "r") as f:
                state_data = json.load(f)
        self.logger.info(f"[MemoryManager] Loaded state for fragment '{fragment_id}' from {state_file}")
        return state_data

    def delete_fragment_state(self, fragment_id: str) -> bool:
        """
        Remove o arquivo de estado (.pt) associado ao fragmento.
        """
        state_file = self.state_dir / f"{fragment_id}.pt"
        try:
            if state_file.exists():
                os.remove(state_file)
                self.logger.info(f"[MemoryManager] Deleted state file for fragment '{fragment_id}' at {state_file}")
            return True
        except Exception as e:
            self.logger.error(f"[MemoryManager] Error deleting state file for '{fragment_id}': {e}")
            return False

    def export_fragment(self, fragment_id: str, export_filename: Optional[Union[str, Path]] = None) -> bool:
        """
        Exporta o fragmento (estado neural + metadados) para um pacote .a3xfrag.
        """
        state_data = self.load_fragment_state(fragment_id)
        if not state_data:
            self.logger.error(f"[MemoryManager] Cannot export: state for '{fragment_id}' not found.")
            return False
        metadata = self.fragment_metadata.get(fragment_id, {})
        if not metadata:
            self.logger.warning(f"[MemoryManager] Export: no metadata found for '{fragment_id}'. Exporting with minimal metadata.")
            metadata = {"fragment_id": fragment_id}
        if export_filename is None:
            export_path = self.export_dir / f"{fragment_id}.a3xfrag"
        else:
            export_path = Path(export_filename)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            if export_path.suffix != ".a3xfrag":
                export_path = export_path.with_suffix(".a3xfrag")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                state_path = tmpdir_path / "fragment.pt"
                if torch:
                    torch.save(state_data.get("state_dict", state_data), state_path)
                else:
                    with open(state_path, "w") as f:
                        json.dump(state_data.get("state_dict", state_data), f)
                meta_path = tmpdir_path / "metadata.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=4)
                with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(state_path, arcname="fragment.pt")
                    zipf.write(meta_path, arcname="metadata.json")
            self.logger.info(f"[MemoryManager] Exported fragment '{fragment_id}' to {export_path}")
            return True
        except Exception as e:
            self.logger.error(f"[MemoryManager] Error exporting fragment '{fragment_id}': {e}")
            if export_path.exists():
                try:
                    os.remove(export_path)
                except OSError:
                    pass
            return False

    def import_fragment(self, a3xfrag_path: Union[str, Path]) -> bool:
        """
        Importa um fragmento de um pacote .a3xfrag (zip).
        """
        a3xfrag_file = Path(a3xfrag_path)
        if not a3xfrag_file.exists() or not a3xfrag_file.is_file():
            self.logger.error(f"[MemoryManager] Import failed: File not found - {a3xfrag_file}")
            return False
        if a3xfrag_file.suffix != ".a3xfrag":
            self.logger.error(f"[MemoryManager] Import failed: Invalid extension - {a3xfrag_file}")
            return False
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                with zipfile.ZipFile(a3xfrag_file, 'r') as zipf:
                    zipf.extractall(tmpdir_path)
                state_path = tmpdir_path / "fragment.pt"
                meta_path = tmpdir_path / "metadata.json"
                if not state_path.exists() or not meta_path.exists():
                    self.logger.error(f"[MemoryManager] Import failed: missing files in archive {a3xfrag_file}")
                    return False
                if torch:
                    state_dict = torch.load(state_path)
                else:
                    with open(state_path, 'r') as f:
                        state_dict = json.load(f)
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                fragment_id = metadata.get("fragment_id")
                if not fragment_id:
                    self.logger.error(f"[MemoryManager] Import failed: fragment_id missing in metadata.")
                    return False
                # Save state
                save_data = {
                    "class_name": metadata.get("class_name", "Unknown"),
                    "module": metadata.get("module", "Unknown"),
                    "state_dict": state_dict,
                    "init_args": metadata
                }
                self.save_fragment_state(fragment_id, save_data)
                # Atualiza metadados
                self.register_fragment(fragment_id, metadata)
                self.logger.info(f"[MemoryManager] Imported fragment '{fragment_id}' from {a3xfrag_file}")
                return True
        except Exception as e:
            self.logger.error(f"[MemoryManager] Error importing fragment from {a3xfrag_file}: {e}")
            return False
    # --- FRAGMENT METADATA MANAGEMENT (NEW SECTION) ---

    def _load_fragment_metadata(self):
        """
        (Optional) Load fragment metadata from a persistent store on initialization.
        For now, this is a placeholder.
        """
        # TODO: Implement loading from a file (e.g., JSON at self.config.get('FRAGMENT_METADATA_PATH')) or database if needed
        self.logger.info("Placeholder: Fragment metadata initialized in-memory.")
        # Example of adding some initial dummy data if not loading from persistent store:
        # self.fragment_metadata = {
        #     "PlaceholderFragment": {"status": "active", "created_at": "...", "path": "..."},
        #     "AnotherBaseFragment": {"status": "active", "created_at": "...", "path": "..."}
        # }

    def _save_fragment_metadata(self):
        """
        (Optional) Save the current fragment metadata to a persistent store.
        """
        # TODO: Implement saving to a file (e.g., JSON at self.config.get('FRAGMENT_METADATA_PATH')) or database
        self.logger.info("Placeholder: Would save fragment metadata if persistence was implemented.")

    def register_fragment(self, fragment_name: str, initial_metadata: Optional[Dict[str, Any]] = None):
        """
        Registers a new fragment or updates existing metadata.
        Ensures a default 'active' status if none is provided.
        """
        if initial_metadata is None:
            initial_metadata = {}

        if fragment_name not in self.fragment_metadata:
            self.fragment_metadata[fragment_name] = initial_metadata
            # Ensure a default status if not provided or is None
            if self.fragment_metadata[fragment_name].get('status') is None:
                self.fragment_metadata[fragment_name]['status'] = 'active' # Default status
            self.logger.info(f"Registered new fragment: '{fragment_name}' with status: {self.fragment_metadata[fragment_name]['status']}")
            # self._save_fragment_metadata() # Optional: Save after registration
        else:
            # Update existing metadata if new metadata is provided
            self.fragment_metadata[fragment_name].update(initial_metadata)
            # Ensure status exists even after update if it wasn't in initial_metadata
            if self.fragment_metadata[fragment_name].get('status') is None:
                 self.fragment_metadata[fragment_name]['status'] = 'active'
            self.logger.info(f"Updated metadata for existing fragment: '{fragment_name}'. Current status: {self.fragment_metadata[fragment_name]['status']}")
            # self._save_fragment_metadata() # Optional: Save after update

    async def update_fragment_status(self, fragment_name: str, status: str) -> bool:
        """
        Updates the status of a registered fragment.

        Args:
            fragment_name: The name of the fragment.
            status: The new status (e.g., 'promoted', 'archived', 'active').

        Returns:
            True if the status was updated successfully, False otherwise (e.g., fragment not found).
        """
        # Ensure fragment exists before trying to update
        if fragment_name not in self.fragment_metadata:
             # Attempt to register it with default status if completely unknown
             self.logger.warning(f"Fragment '{fragment_name}' not found during status update attempt. Registering with default 'active' status before proceeding.")
             self.register_fragment(fragment_name) # Register with defaults

        # Now update the status
        old_status = self.fragment_metadata[fragment_name].get('status', 'unknown')
        if old_status == status:
             self.logger.debug(f"Fragment '{fragment_name}' already has status '{status}'. No update needed.")
             return True # Indicate success as the state is already correct

        self.fragment_metadata[fragment_name]['status'] = status
        self.logger.info(f"Updated status for fragment '{fragment_name}' from '{old_status}' to '{status}'.")
        # self._save_fragment_metadata() # Optional: Save after status update
        return True
        # Note: Previous logic returned False if fragment wasn't found.
        # Changed to register-if-missing and always return True if update occurs or status matches,
        # as the goal is to ensure the fragment HAS the desired status.

    def get_fragment_status(self, fragment_name: str) -> Optional[str]:
        """
        Retrieves the current status of a fragment.

        Returns:
            The status string or None if the fragment is not registered.
        """
        metadata = self.fragment_metadata.get(fragment_name)
        if metadata:
            return metadata.get('status')
        self.logger.debug(f"Fragment '{fragment_name}' not found in metadata when getting status.")
        return None

    def list_fragments(self, status_filter: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Lists registered fragments, optionally filtering by status.
        Returns a copy to prevent external modification of the internal state.
        """
        if status_filter:
            return {name: meta.copy() for name, meta in self.fragment_metadata.items() if meta.get('status') == status_filter}
        else:
            return {name: meta.copy() for name, meta in self.fragment_metadata.items()}

    # --- SEMANTIC MEMORY ---

    def search_semantic_memory(self, query: str) -> List[Dict[str, Any]]:
        """
        Busca na memória semântica (FAISS) usando o embedding do texto.
        """
        self.logger.info(f"Buscando memória semântica para: '{query[:50]}...'")
        embedding = self.get_embedding(query)
        if embedding is None:
            self.logger.error("Falha ao gerar embedding para busca semântica.")
            return []
        try:
            results = self.search_index(
                index_path_base=self.semantic_index_path,
                query_embedding=embedding,
                top_k=self.semantic_top_k,
            )
            return results or []
        except Exception as e:
            self.logger.exception("Erro ao buscar na memória semântica:")
            return []

    # --- EPISODIC MEMORY ---

    def get_recent_episodes(self) -> List[Dict[str, Any]]:
        """
        Recupera os episódios mais recentes da memória episódica.
        """
        try:
            episodes = self.retrieve_recent_episodes(limit=self.episodic_limit)
            return [dict(row) for row in episodes] if episodes else []
        except Exception as e:
            self.logger.exception("Erro ao recuperar episódios recentes:")
            return []

    def record_episodic_event(self, context: str, action: str, outcome: str, metadata: Optional[dict] = None):
        """
        Registra um evento na memória episódica.
        """
        try:
            self.add_episodic_record(context, action, outcome, metadata)
        except Exception as e:
            self.logger.exception("Erro ao registrar evento episódico:")

    # --- EXTENSÃO FUTURA: Métodos para outros tipos de memória/backends podem ser adicionados aqui ---

    async def learn_from_task(self, learning_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processa os dados de uma tarefa concluída, persiste a experiência e recomenda
        o próximo passo no ciclo de aprendizado/evolução.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"[MemoryManager] Recebido learning_data para task_id (se disponível): {learning_data.get('task_id', 'N/A')}")
        
        # --- 1. Persistir a Experiência (Exemplo: Logar em JSONL) ---
        try:
            # Definir caminho para o log de conclusão de tarefas
            task_completion_log_path = LEARNING_LOGS_DIR / "task_completion_log.jsonl"
            task_completion_log_path.parent.mkdir(parents=True, exist_ok=True) # Garante que o diretório exista
            
            # Remover dados que podem não ser serializáveis diretamente (ex: objetos complexos)
            serializable_data = learning_data.copy()
            # Adicionar/remover campos conforme necessário para o log
            # Exemplo: remover histórico detalhado se for muito grande para logar sempre
            # serializable_data.pop('history', None) 

            with open(task_completion_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(serializable_data) + '\n')
            logger.debug(f"[MemoryManager] learning_data persistido em {task_completion_log_path}")
        except Exception as e:
            logger.error(f"[MemoryManager] Falha ao persistir learning_data: {e}", exc_info=True)
            # Continuar mesmo se o log falhar? Ou retornar erro? Por enquanto, apenas logar.

        # --- 2. Analisar Resultado e Recomendar Próximo Passo ---
        recommendation = None
        final_status = learning_data.get('final_status')

        if final_status == 'error':
            logger.info("[MemoryManager] Tarefa falhou. Recomendando ciclo de reflexão sobre falha.")
            # Poderia chamar learning_cycle.process_execution_failure aqui, mas vamos delegar de volta
            recommendation = {'trigger': 'reflection_on_failure', 'context': learning_data}
        elif final_status == 'success':
            logger.info("[MemoryManager] Tarefa concluída com sucesso. Analisando eficiência...")
            # Lógica simples: se demorou muitos passos, talvez precise otimizar
            steps_taken = learning_data.get('steps_taken', 0)
            if steps_taken > 10: # Limite arbitrário, pode vir da config
                 logger.info("[MemoryManager] Tarefa levou muitos passos. Recomendando ciclo de auto-evolução/otimização.")
                 recommendation = {'trigger': 'self_evolution', 'context': learning_data}
            else:
                 logger.info("[MemoryManager] Tarefa concluída eficientemente. Nenhuma ação de evolução imediata recomendada.")
                 # Poderia retornar um trigger para 'reflection_on_success' se existir
                 recommendation = {'trigger': 'reflection_on_success', 'context': learning_data} # Opcional

        else:
            logger.warning(f"[MemoryManager] Status final desconhecido: {final_status}. Nenhuma recomendação.")

        return recommendation # Retorna a recomendação para o Orchestrator decidir