import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from a3x.core.config import MULTIMODAL_STORAGE_DIR # Import constant

logger = logging.getLogger(__name__)

class MultimodalCollector:
    """
    Coletor e integrador multi-modal para o A³X.
    Permite ingestão e conexão de dados de texto, imagens, web, APIs externas e sensores.
    """

    def __init__(self, storage_dir: Path = MULTIMODAL_STORAGE_DIR): # Use constant for default storage_dir
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def collect_text(self, text: str, source: str = "manual", meta: Optional[Dict[str, Any]] = None):
        """
        Armazena texto com metadados.
        """
        entry = {"type": "text", "content": text, "source": source, "meta": meta or {}}
        self._save_entry(entry)

    def collect_image(self, image_path: Path, description: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
        """
        Armazena referência a imagem e metadados (pode ser expandido para embeddings ou OCR).
        """
        entry = {"type": "image", "path": str(image_path), "description": description, "meta": meta or {}}
        self._save_entry(entry)

    def collect_web(self, url: str, content: str, meta: Optional[Dict[str, Any]] = None):
        """
        Armazena conteúdo web (ex: scraping, API) com metadados.
        """
        entry = {"type": "web", "url": url, "content": content, "meta": meta or {}}
        self._save_entry(entry)

    def collect_api(self, api_name: str, data: Any, meta: Optional[Dict[str, Any]] = None):
        """
        Armazena dados de APIs externas.
        """
        entry = {"type": "api", "api_name": api_name, "data": data, "meta": meta or {}}
        self._save_entry(entry)

    def collect_sensor(self, sensor_type: str, value: Any, meta: Optional[Dict[str, Any]] = None):
        """
        Armazena dados de sensores físicos ou virtuais.
        """
        entry = {"type": "sensor", "sensor_type": sensor_type, "value": value, "meta": meta or {}}
        self._save_entry(entry)

    def _save_entry(self, entry: Dict[str, Any]):
        """
        Salva entrada multimodal em arquivo JSONL.
        """
        file_path = self.storage_dir / "multimodal_log.jsonl"
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"[MultimodalCollector] Entrada salva: {entry['type']}")

    def load_entries(self, entry_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Carrega entradas multimodais do log, filtrando por tipo se necessário.
        """
        file_path = self.storage_dir / "multimodal_log.jsonl"
        if not file_path.exists():
            return []
        entries = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if not entry_type or entry.get("type") == entry_type:
                        entries.append(entry)
                except Exception as e:
                    logger.warning(f"[MultimodalCollector] Falha ao carregar entrada: {e}")
        return entries
