from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def adjust_llm_parameters(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ajusta automaticamente os parâmetros do LLM com base no histórico de respostas para melhorar a geração de conteúdo.
    
    Args:
        context: Objeto de contexto contendo memória e informações do agente.
    
    Returns:
        Dict[str, Any]: Resultado do ajuste, incluindo os novos parâmetros e uma mensagem de status.
    """
    if context is None:
        context = {}
    
    # Verificar histórico de respostas do LLM na memória do agente
    memory = context.get('mem', {})
    llm_history = memory.get('llm_responses', [])
    
    # Analisar falhas recentes (exemplo: respostas vazias ou muito curtas)
    if not llm_history:
        logger.info("[AdjustLLM] Nenhum histórico de respostas do LLM encontrado. Usando parâmetros padrão.")
        default_params = {"temperature": 0.7, "n_predict": 1024, "top_k": 40, "top_p": 0.9}
        return {
            "status": "success",
            "message": f"Nenhum histórico disponível. Parâmetros mantidos como padrão: {default_params}",
            "new_parameters": default_params
        }
    
    recent_responses = llm_history[-5:] if len(llm_history) >= 5 else llm_history
    empty_responses = [resp for resp in recent_responses if not resp or len(str(resp).strip()) < 10]
    failure_rate = len(empty_responses) / len(recent_responses) if recent_responses else 0
    
    # Ajustar parâmetros com base na taxa de falha
    new_parameters = {
        "temperature": 0.7,  # Valor padrão
        "n_predict": 1024,   # Valor padrão
        "top_k": 40,         # Valor padrão
        "top_p": 0.9         # Valor padrão
    }
    
    if failure_rate > 0.5:
        new_parameters["temperature"] = 0.9  # Aumentar para respostas mais criativas
        new_parameters["n_predict"] = 2048   # Mais tokens para completar a resposta
        logger.info(f"[AdjustLLM] Ajustando parâmetros do LLM devido a alta taxa de falha ({failure_rate*100:.1f}%): {new_parameters}")
        message = f"Alta taxa de falha detectada ({failure_rate*100:.1f}%). Parâmetros ajustados para {new_parameters}"
    else:
        logger.info(f"[AdjustLLM] Parâmetros do LLM mantidos nos valores padrão: {new_parameters}")
        message = f"Taxa de falha aceitável ({failure_rate*100:.1f}%). Parâmetros mantidos como padrão: {new_parameters}"
    
    # Atualizar memória ou configuração do agente com os novos parâmetros
    if 'llm_config' not in memory:
        memory['llm_config'] = {}
    memory['llm_config'].update(new_parameters)
    
    # Registrar a heurística aprendida para uso futuro
    heuristic = {
        "context": "LLM response failure rate",
        "failure_rate": failure_rate,
        "adjustment": new_parameters,
        "timestamp": str(logging.LogRecord('root', logging.INFO, None, None, None, None, None).asctime),
        "status": "experimental"
    }
    if 'heuristics' not in memory:
        memory['heuristics'] = []
    memory['heuristics'].append(heuristic)
    logger.info(f"[AdjustLLM] Heurística registrada: {heuristic}")
    
    return {
        "status": "success",
        "message": message,
        "new_parameters": new_parameters,
        "heuristic": heuristic
    } 