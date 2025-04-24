import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from a3x.core.memory.memory_manager import MemoryManager
from a3x.core.context import SharedTaskContext

logger = logging.getLogger(__name__)

class DatasetBuilder:
    """Constrói datasets para treinamento do A3Net a partir do ContextStore."""

    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.dataset_dir = Path("a3x/a3net/data")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    async def build_dataset(self, context: SharedTaskContext) -> str:
        """Constrói um dataset a partir do contexto atual."""
        try:
            # 1. Coletar dados do contexto
            context_data = await self._collect_context_data(context)
            
            # 2. Processar dados
            processed_data = await self._process_data(context_data)
            
            # 3. Gerar dataset
            dataset_path = await self._generate_dataset(processed_data)
            
            return dataset_path
            
        except Exception as e:
            logger.exception("Erro ao construir dataset:")
            raise

    async def _collect_context_data(self, context: SharedTaskContext) -> List[Dict[str, Any]]:
        """Coleta dados relevantes do contexto."""
        data = []
        
        # Coletar histórico de execução
        history = await context.get_history()
        for fragment_name, result in history:
            data.append({
                "fragment": fragment_name,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
        
        # Coletar dados da memória semântica
        semantic_data = self.memory_manager.search_semantic_memory(
            context.initial_objective or ""
        )
        data.extend(semantic_data)
        
        return data

    async def _process_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processa os dados brutos para o formato de treinamento."""
        processed = []
        for item in raw_data:
            # Implementar lógica de processamento
            processed.append({
                "input": self._prepare_input(item),
                "target": self._prepare_target(item)
            })
        return processed

    def _prepare_input(self, item: Dict[str, Any]) -> str:
        """Prepara o input para treinamento."""
        # Implementar lógica de preparação
        return ""

    def _prepare_target(self, item: Dict[str, Any]) -> str:
        """Prepara o target para treinamento."""
        # Implementar lógica de preparação
        return ""

    async def _generate_dataset(self, data: List[Dict[str, Any]]) -> str:
        """Gera o arquivo JSONL do dataset."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_path = self.dataset_dir / f"dataset_{timestamp}.jsonl"
        
        with open(dataset_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return str(dataset_path) 