import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_DIM

# Configuração de logging
logging.basicConfig(
    filename='logs/utils.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Variável global para o modelo (lazy loading)
_model = None

def _get_model():
    """
    Carrega o modelo sentence-transformers se ainda não estiver carregado.
    
    Returns:
        SentenceTransformer: Modelo carregado
    """
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("Modelo sentence-transformers carregado com sucesso")
        except Exception as e:
            logging.error(f"Erro ao carregar modelo: {str(e)}")
            raise
    return _model

def generate_embedding(text: str) -> np.ndarray:
    """
    Gera embedding para um texto usando o modelo all-MiniLM-L6-v2.
    
    Args:
        text: Texto para gerar o embedding
        
    Returns:
        np.ndarray: Embedding do texto
        
    Raises:
        Exception: Se houver erro ao gerar o embedding
    """
    try:
        # Carrega o modelo se necessário
        model = _get_model()
        
        # Gera o embedding
        embedding = model.encode(text, convert_to_numpy=True)
        
        # Verifica dimensão
        if embedding.shape[0] != EMBEDDING_DIM:
            logging.warning(f"Dimensão do embedding ({embedding.shape[0]}) diferente da esperada ({EMBEDDING_DIM})")
        
        # Converte para float32
        embedding = embedding.astype(np.float32)
        
        logging.info(f"Embedding gerado com sucesso para texto: {text[:50]}...")
        return embedding
        
    except Exception as e:
        logging.error(f"Erro ao gerar embedding: {str(e)}")
        raise 