import logging
import time
from sentence_transformers import SentenceTransformer
import numpy as np
import os # Adicionado para configurar variável de ambiente

logger = logging.getLogger(__name__)

# --- Configuração do Modelo ---
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
NORMALIZE_EMBEDDINGS = True
EMBEDDING_DIM = None # Será definido dinamicamente ou por fallback

# --- Variável Global para o Modelo Carregado ---
_model_instance = None
_model_loading_error = None

def _load_model_internal():
    """Função interna para carregar o modelo SentenceTransformer na memória (chamada apenas uma vez)."""
    global _model_instance, EMBEDDING_DIM, _model_loading_error
    if _model_instance is not None or _model_loading_error is not None: # Já carregado ou falhou antes
        logger.debug("[Embeddings] Modelo já carregado ou falha anterior detectada. Pulando carregamento.")
        return

    try:
        logger.info(f"[Embeddings] Carregando modelo '{MODEL_NAME}' pela primeira vez...")
        start_time = time.time()
        _model_instance = SentenceTransformer(MODEL_NAME)
        end_time = time.time()
        logger.info(f"[Embeddings] Modelo '{MODEL_NAME}' carregado com sucesso em {end_time - start_time:.2f} segundos.")

        # Determina e loga a dimensão do embedding
        try:
            dummy_embedding = _model_instance.encode(["dim_test"])
            EMBEDDING_DIM = dummy_embedding.shape[1]
            logger.info(f"[Embeddings] Dimensão dos embeddings do modelo '{MODEL_NAME}': {EMBEDDING_DIM}")
            # Validação crucial para VSS
            if EMBEDDING_DIM is None:
                 raise ValueError("A dimensão do embedding não pôde ser determinada.")
        except Exception as dim_err:
            logger.warning(f"[Embeddings] Não foi possível determinar dinamicamente a dimensão do embedding: {dim_err}.")
            # Tenta um valor padrão conhecido
            # Ajuste este bloco se mudar o modelo padrão!
            if 'bert-base' in MODEL_NAME.lower(): # Verifica se é um BERT base
                 EMBEDDING_DIM = 768 # <<< DIMENSÃO PADRÃO PARA BERT-BASE
                 logger.info(f"[Embeddings] Usando dimensão padrão {EMBEDDING_DIM} para {MODEL_NAME}.")
            elif 'bert-large' in MODEL_NAME.lower():
                 EMBEDDING_DIM = 1024 # Padrão para BERT-large
                 logger.info(f"[Embeddings] Usando dimensão padrão {EMBEDDING_DIM} para {MODEL_NAME}.")
            elif 'multilingual-e5-large' in MODEL_NAME.lower():
                 EMBEDDING_DIM = 1024 # Padrão para E5-large
                 logger.info(f"[Embeddings] Usando dimensão padrão {EMBEDDING_DIM} para {MODEL_NAME}.")
            # Mantenha outros padrões se necessário
            # elif MODEL_NAME == 'ibm-granite/granite-embedding-125m-english':
            #      EMBEDDING_DIM = 768
            #      logger.info(f"[Embeddings] Usando dimensão padrão {EMBEDDING_DIM} para {MODEL_NAME}.")
            else:
                 logger.error(f"[Embeddings] Falha ao determinar dimensão e nenhum padrão conhecido para {MODEL_NAME}. VSS pode falhar!")
                 _model_loading_error = RuntimeError(f"Falha ao determinar dimensão do embedding para {MODEL_NAME}.")
                 # Limpa a instância se a dimensão não puder ser definida, forçando erro em get_embedding
                 _model_instance = None
                 return # Sai da função _load_model_internal

    except Exception as e:
        logger.error(f"[Embeddings] FALHA CRÍTICA ao carregar o modelo de embedding '{MODEL_NAME}': {e}", exc_info=True)
        _model_loading_error = e
        _model_instance = None # Garante que a instância é None se o carregamento falhar

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
         logger.error(f"[Embeddings] Modelo não pôde ser carregado anteriormente: {_model_loading_error}")
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
        embeddings = _model_instance.encode(texts_to_encode, normalize_embeddings=NORMALIZE_EMBEDDINGS)
        end_time = time.time()
        logger.debug(f"[Embeddings] Embedding gerado em {end_time - start_time:.4f} segundos.")

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
        return 0.0 # Retorna 0 em caso de erro

# Opcional: Pré-carregar ao iniciar? Avaliar impacto no startup.
# logger.info("[Embeddings] Módulo carregado, tentando pré-carregar modelo...")
# _load_model_internal()


if __name__ == "__main__":
    # Teste rápido do módulo
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s Embeddings Test] %(message)s')
    logger.info("Iniciando teste do módulo de embeddings...")

    test_texts = [
        "O céu é azul durante o dia.",
        "O céu costuma ser azul em dias claros.",
        "Gatos gostam de caixas de papelão.",
        "Meu gato se chama Mingau."
    ]

    embeddings = []
    for text in test_texts:
        emb = get_embedding(text)
        if emb:
            logger.info(f"Embedding gerado para: '{text}' (Dim: {len(emb)})")
            embeddings.append(np.array(emb)) # Converte para numpy para calculate_similarity
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
            logger.warning("\nResultados de similaridade podem não estar como esperado.")
    else:
        logger.error("\nTeste de similaridade não pôde ser concluído devido a falhas na geração de embeddings.")

    logger.info("Teste do módulo de embeddings concluído.") 