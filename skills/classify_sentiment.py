# /home/arthur/Projects/A3X/skills/classify_sentiment.py
import logging
from typing import Dict, Any, List, Optional
import time

# Configure logger
logger = logging.getLogger(__name__)

# Model name from Hugging Face model hub
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

# Global variable to hold the pipeline instance (lazy loading)
_pipeline_instance = None
_pipeline_loading_error = None

def _get_device_id() -> int:
    """Determina o ID do dispositivo (GPU se disponível, senão CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            # Check CUDA (NVIDIA) or ROCm (AMD) availability
            logger.info("GPU (CUDA/ROCm) detectada. Usando GPU para classificação de sentimento.")
            return 0 # Use the first available GPU
    except ImportError:
        logger.warning("PyTorch não encontrado. Classificação de sentimento será executada na CPU.")
    except Exception as e:
        logger.error(f"Erro ao verificar GPU com PyTorch: {e}. Usando CPU.")

    logger.info("Nenhuma GPU compatível detectada ou PyTorch ausente. Usando CPU para classificação de sentimento.")
    return -1 # transformers pipeline uses -1 for CPU

def _load_pipeline_internal():
    """Carrega o pipeline de classificação de sentimento (chamado apenas uma vez)."""
    global _pipeline_instance, _pipeline_loading_error
    if _pipeline_instance is not None or _pipeline_loading_error is not None:
        return

    try:
        from transformers import pipeline
        logger.info(f"[Sentiment] Carregando pipeline para '{MODEL_NAME}'...")
        start_time = time.time()
        device_id = _get_device_id()
        # Using device= explicitly might be more reliable than device_id for newer transformers
        device_name = "cpu" if device_id == -1 else f"cuda:{device_id}" 
        _pipeline_instance = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            # device=device_id # Older way
            device=device_name
        )
        end_time = time.time()
        load_duration = end_time - start_time
        logger.info(f"[Sentiment] Pipeline '{MODEL_NAME}' carregado em {load_duration:.2f} segundos no dispositivo '{device_name}'.")

    except ImportError:
        logger.error("[Sentiment] Biblioteca 'transformers' não encontrada. Instale com 'pip install transformers[torch]' ou 'transformers[tensorflow]'.")
        _pipeline_loading_error = ImportError("transformers not found")
    except Exception as e:
        logger.exception("[Sentiment] Erro inesperado ao carregar o pipeline de sentimento:")
        _pipeline_loading_error = e

def skill_classify_sentiment(action_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classifica o sentimento de um texto usando o modelo nlptown/bert-base-multilingual-uncased-sentiment.
    Retorna uma pontuação de 1 (muito negativo) a 5 (muito positivo) e o score de confiança.

    Expected action_input parameters:
      - text (str): O texto a ser classificado (required).
    """
    global _pipeline_instance, _pipeline_loading_error
    logger.debug(f"Executando skill_classify_sentiment com input: {action_input}")

    text_content = action_input.get("text")

    # --- Basic Validation ---
    if not text_content or not isinstance(text_content, str):
        return {"status": "error", "action": "sentiment_classification_failed", "data": {"message": "Error: 'text' parameter is required and must be a string."}}

    # --- Load Pipeline (if needed) ---
    if _pipeline_instance is None and _pipeline_loading_error is None:
        logger.debug("[Sentiment] Tentando carregar pipeline em skill_classify_sentiment.")
        _load_pipeline_internal()

    if _pipeline_loading_error is not None:
        logger.error(f"[Sentiment] Pipeline não pôde ser carregado anteriormente: {_pipeline_loading_error}")
        return {"status": "error", "action": "sentiment_classification_failed", "data": {"message": f"Failed to load sentiment analysis model: {_pipeline_loading_error}"}}
    if _pipeline_instance is None:
        logger.error("[Sentiment] Instância do pipeline é None inesperadamente.")
        return {"status": "error", "action": "sentiment_classification_failed", "data": {"message": "Sentiment analysis model instance is unexpectedly None."}}

    # --- Classification Logic ---
    try:
        logger.info(f"[Sentiment] Classificando sentimento para texto (length: {len(text_content)} characters)...")
        start_time = time.time()

        # O pipeline retorna uma lista de dicionários, pegamos o primeiro resultado
        results = _pipeline_instance(text_content)
        if not results or not isinstance(results, list):
            raise ValueError("Pipeline did not return expected list format.")
            
        result = results[0] # Assume single text input
        label = result.get('label') # e.g., '5 stars'
        score = result.get('score') # e.g., 0.987

        end_time = time.time()
        logger.info(f"[Sentiment] Classificação concluída em {end_time - start_time:.4f} segundos. Resultado: {label}, Score: {score:.4f}")

        # Extrair a pontuação numérica do label
        sentiment_rating = None
        if label and isinstance(label, str):
            try:
                sentiment_rating = int(label.split()[0]) # Pega o número antes de ' stars'
            except (ValueError, IndexError):
                logger.warning(f"[Sentiment] Não foi possível extrair a pontuação numérica do label: '{label}'")

        return {
            "status": "success",
            "action": "sentiment_classified",
            "data": {
                "sentiment_label": label,
                "sentiment_rating": sentiment_rating, # 1 to 5
                "confidence_score": score
            }
        }

    except Exception as e:
        logger.exception("[Sentiment] Erro inesperado durante a classificação de sentimento:")
        return {"status": "error", "action": "sentiment_classification_failed", "data": {"message": f"Unexpected error during sentiment classification: {e}"}}
