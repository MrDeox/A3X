import logging
import time
import numpy as np
import os
import datetime
from typing import Optional  # <<< Added Optional import

logger = logging.getLogger(__name__)

# Local imports
# from core.config import PROJECT_ROOT

# --- Model Configuration ---
# Model name from Hugging Face model hub
# MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2") # Default Original
MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
)  # <<< NEW MODEL
# Cache directory for models (optional, defaults to ~/.cache/huggingface/hub)
# CACHE_DIR = os.environ.get("HF_HOME")

# <<< ADDED Definition for NORMALIZE_EMBEDDINGS >>>
NORMALIZE_EMBEDDINGS = True  # Set based on usage in calculate_similarity

# Global variable to hold the model instance (lazy loading)
_model_instance = None
_model_loading_error = None

# Global variable for embedding dimension (determined after model load)
EMBEDDING_DIM = None


# --- Helper Functions ---
def _get_device() -> str:
    """Determina o dispositivo disponível (CPU ou CUDA/ROCm via PyTorch)."""
    try:
        import torch

        # Check for CUDA (NVIDIA)
        if torch.cuda.is_available():
            logger.info("CUDA (NVIDIA GPU) detectado. Usando GPU para embeddings.")
            return "cuda"
        # Check for ROCm (AMD) - requires PyTorch built with ROCm support
        # hasattr check is important as direct access might error if not compiled in
        if (
            hasattr(torch.version, "hip")
            and torch.version.hip
            and torch.cuda.is_available()
        ):
            # torch.cuda.is_available() returns true for ROCm too if installed correctly
            logger.info("ROCm (AMD GPU) detectado. Usando GPU para embeddings.")
            return "cuda"  # PyTorch uses 'cuda' device name even for ROCm
    except ImportError:
        logger.warning("PyTorch não encontrado. Embeddings serão executados na CPU.")
    except Exception as e:
        logger.error(
            f"Erro ao verificar disponibilidade de GPU com PyTorch: {e}. Usando CPU."
        )

    logger.info(
        "Nenhuma GPU compatível (CUDA/ROCm) detectada ou PyTorch ausente. Usando CPU para embeddings."
    )
    return "cpu"


def _load_model_internal():
    """Função interna para carregar o modelo SentenceTransformer na memória (chamada apenas uma vez)."""
    global _model_instance, _model_loading_error, EMBEDDING_DIM
    if _model_instance is not None or _model_loading_error is not None:
        return  # Already loaded or failed

    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"[Embeddings] Carregando modelo '{MODEL_NAME}'...")
        start_time = datetime.datetime.now()
        device = _get_device()
        _model_instance = SentenceTransformer(MODEL_NAME, device=device)
        end_time = datetime.datetime.now()
        load_duration = (end_time - start_time).total_seconds()
        logger.info(
            f"[Embeddings] Modelo '{MODEL_NAME}' carregado em {load_duration:.2f} segundos no dispositivo '{device}'."
        )

        # Determinar a dimensão do embedding dinamicamente
        try:
            # Tenta obter a dimensão do primeiro módulo de embedding
            if hasattr(
                _model_instance, "get_sentence_embedding_dimension"
            ) and callable(_model_instance.get_sentence_embedding_dimension):
                EMBEDDING_DIM = _model_instance.get_sentence_embedding_dimension()
            # Fallback: verifica o atributo 'embedding_length' ou similar (pode variar)
            elif hasattr(_model_instance, "embedding_length"):
                EMBEDDING_DIM = _model_instance.embedding_length
            # Outro fallback comum é verificar a dimensão do word_embedding_model
            elif (
                hasattr(_model_instance, "0")
                and hasattr(_model_instance[0], "auto_model")
                and hasattr(_model_instance[0].auto_model, "config")
            ):
                EMBEDDING_DIM = _model_instance[0].auto_model.config.hidden_size
            else:
                # Se nada funcionar, tenta gerar um embedding e ver o tamanho
                logger.warning(
                    "[Embeddings] Não foi possível determinar a dimensão via atributos. Tentando via inferência..."
                )
                test_embedding = _model_instance.encode(["teste"])
                EMBEDDING_DIM = test_embedding.shape[1]

            if EMBEDDING_DIM:
                logger.info(
                    f"[Embeddings] Dimensão do embedding determinada: {EMBEDDING_DIM}"
                )
            else:
                raise ValueError("Não foi possível determinar a dimensão do embedding.")

        except Exception as dim_err:
            logger.warning(
                f"[Embeddings] Não foi possível determinar dinamicamente a dimensão do embedding: {dim_err}."
            )
            # Tenta um valor padrão conhecido
            # Ajuste este bloco se mudar o modelo padrão!
            # <<< UPDATED DEFAULTS >>>
            if "e5-base" in MODEL_NAME.lower():
                EMBEDDING_DIM = 768
                logger.info(
                    f"[Embeddings] Usando dimensão padrão {EMBEDDING_DIM} para {MODEL_NAME}."
                )
            elif (
                "minilm" in MODEL_NAME.lower()
            ):  # Covers all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2
                EMBEDDING_DIM = 384
                logger.info(
                    f"[Embeddings] Usando dimensão padrão {EMBEDDING_DIM} para {MODEL_NAME}."
                )
            elif "bert-base" in MODEL_NAME.lower():  # Mantém para outros BERTs base
                EMBEDDING_DIM = 768
                logger.info(
                    f"[Embeddings] Usando dimensão padrão {EMBEDDING_DIM} para {MODEL_NAME}."
                )
            elif "e5-large" in MODEL_NAME.lower():  # Adiciona E5-large
                EMBEDDING_DIM = 1024
                logger.info(
                    f"[Embeddings] Usando dimensão padrão {EMBEDDING_DIM} para {MODEL_NAME}."
                )
            # Mantenha outros padrões se necessário
            # ...
            else:
                logger.error(
                    f"[Embeddings] Falha ao determinar dimensão e nenhum padrão conhecido para {MODEL_NAME}. VSS pode falhar!"
                )
                _model_loading_error = RuntimeError(
                    f"Falha ao determinar dimensão do embedding para {MODEL_NAME}."
                )
                _model_instance = None
                return

    except ImportError:
        logger.error(
            "[Embeddings] Biblioteca 'sentence-transformers' não encontrada. Instale com 'pip install sentence-transformers'."
        )
        _model_loading_error = ImportError("sentence-transformers not found")
    except Exception as e:
        logger.exception(
            "[Embeddings] Erro inesperado ao carregar o modelo de embedding:"
        )
        _model_loading_error = e


def get_embedding_dim() -> Optional[int]:
    """Retorna a dimensão do embedding do modelo carregado."""
    if (
        EMBEDDING_DIM is None
        and _model_instance is None
        and _model_loading_error is None
    ):
        # Tenta carregar o modelo se ainda não foi tentado
        logger.info(
            "[Embeddings] get_embedding_dim chamado antes do modelo ser carregado. Tentando carregar agora..."
        )
        _load_model_internal()
    # Retorna a dimensão se carregada com sucesso, senão None
    return EMBEDDING_DIM


def get_embedding(text: str) -> list[float] | None:
    """
    Gera o vetor de embedding para um dado texto, carregando o modelo se necessário.
    """
    global _model_instance, _model_loading_error

    # Tenta carregar o modelo na primeira chamada (ou se falhou antes e foi resetado)
    if _model_instance is None and _model_loading_error is None:
        logger.debug("[Embeddings] Tentando carregar modelo em get_embedding.")
        _load_model_internal()

    # Verifica se houve erro no carregamento ou se a instância não foi criada
    if _model_loading_error is not None:
        logger.error(
            f"[Embeddings] Modelo não pôde ser carregado anteriormente: {_model_loading_error}"
        )
        return None
    if _model_instance is None:
        # Log específico se chegou aqui sem erro, mas sem instância (deve ser raro)
        logger.error("[Embeddings] Instância do modelo é None inesperadamente.")
        return None

    # Se chegou aqui, o modelo deve estar carregado
    try:
        logger.debug(f"[Embeddings] Gerando embedding para texto: '{text[:100]}...'")
        start_time = time.time()
        texts_to_encode = [text]
        embeddings = _model_instance.encode(
            texts_to_encode, normalize_embeddings=NORMALIZE_EMBEDDINGS
        )
        end_time = time.time()
        logger.debug(
            f"[Embeddings] Embedding gerado em {end_time - start_time:.4f} segundos."
        )

        embedding_list = embeddings[0].tolist()
        return embedding_list

    except Exception as e:
        logger.error(f"[Embeddings] Erro ao gerar embedding: {e}", exc_info=True)
        return None


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calcula a similaridade de cosseno entre dois embeddings."""
    try:
        # Assume-se que os embeddings já estão normalizados (NORMALIZE_EMBEDDINGS=True em get_embedding)
        # Se NORMALIZE_EMBEDDINGS for False, descomente a normalização abaixo:
        # norm1 = np.linalg.norm(embedding1)
        # norm2 = np.linalg.norm(embedding2)
        # if norm1 == 0 or norm2 == 0:
        #     return 0.0
        # embedding1 = embedding1 / norm1
        # embedding2 = embedding2 / norm2

        # Calcula similaridade de cosseno (produto escalar de vetores normalizados)
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)  # Converte para float Python padrão
    except Exception as e:
        logger.error(f"Erro ao calcular similaridade: {e}", exc_info=True)
        return 0.0  # Retorna 0 em caso de erro


# Opcional: Pré-carregar ao iniciar? Avaliar impacto no startup.
# logger.info("[Embeddings] Módulo carregado, tentando pré-carregar modelo...")
# _load_model_internal()


if __name__ == "__main__":
    # Teste rápido do módulo
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s Embeddings Test] %(message)s"
    )
    logger.info("Iniciando teste do módulo de embeddings...")

    test_texts = [
        "O céu é azul durante o dia.",
        "O céu costuma ser azul em dias claros.",
        "Gatos gostam de caixas de papelão.",
        "Meu gato se chama Mingau.",
    ]

    embeddings = []
    for text in test_texts:
        emb = get_embedding(text)
        if emb:
            logger.info(f"Embedding gerado para: '{text}' (Dim: {len(emb)})")
            embeddings.append(
                np.array(emb)
            )  # Converte para numpy para calculate_similarity
        else:
            logger.error(f"Falha ao gerar embedding para: '{text}'")

    if len(embeddings) == 4:
        logger.info("\nCalculando similaridades:")
        sim_0_1 = calculate_similarity(embeddings[0], embeddings[1])
        sim_0_2 = calculate_similarity(embeddings[0], embeddings[2])
        sim_2_3 = calculate_similarity(embeddings[2], embeddings[3])

        logger.info(f"Similaridade (Céu 1 vs Céu 2): {sim_0_1:.4f}")
        logger.info(f"Similaridade (Céu 1 vs Gato 1): {sim_0_2:.4f}")
        logger.info(f"Similaridade (Gato 1 vs Gato 2): {sim_2_3:.4f}")

        # Verificações básicas
        if sim_0_1 > sim_0_2 and sim_2_3 > sim_0_2:
            logger.info("\nResultados de similaridade parecem consistentes.")
        else:
            logger.warning(
                "\nResultados de similaridade podem não estar como esperado."
            )
    else:
        logger.error(
            "\nTeste de similaridade não pôde ser concluído devido a falhas na geração de embeddings."
        )

    logger.info("Teste do módulo de embeddings concluído.")
