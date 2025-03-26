"""
Implementa o Núcleo Cognitivo (Cognitive Kernel) do A³X, inspirado na Global Workspace Theory.
Responsável por orquestrar o fluxo de informações e a ativação dos diferentes módulos cognitivos.
"""

import logging
from typing import Dict, Any, Optional, Callable
from core.analyzer import analyze_intent

# Configurar logger específico para o módulo
logger = logging.getLogger(__name__)

class CognitiveKernel:
    """
    Núcleo Cognitivo do A³X, responsável por coordenar o fluxo de informações
    entre os diferentes módulos cognitivos.
    """
    
    def __init__(self, intent_parser: Optional[Callable[[str], Dict[str, Any]]] = None):
        """
        Inicializa o Núcleo Cognitivo.
        Configura o espaço de trabalho e prepara placeholders para módulos futuros.
        
        Args:
            intent_parser: Função para análise de intenção. Se None, usa analyze_intent do core.
        """
        # Espaço de trabalho global (contexto atual)
        self.current_context: Dict[str, Any] = {}
        
        # Parser de intenção
        self.intent_parser = intent_parser or analyze_intent
        logger.info("Parser de intenção configurado")
        
        # Placeholders para módulos futuros
        self.memory: Optional[Any] = None
        self.planner: Optional[Any] = None
        self.executor: Optional[Any] = None
        self.reflector: Optional[Any] = None
        
        logger.info("Cognitive Kernel inicializado")
        
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa um novo input, atualizando o contexto e preparando para distribuição
        aos módulos apropriados.
        
        Args:
            input_data: Dicionário contendo os dados do input (ex: comando do usuário)
            
        Returns:
            Dict[str, Any]: Status do processamento inicial
        """
        logger.info(f"Novo input recebido: {input_data}")
        
        # Atualiza o contexto com o novo input
        self.current_context['last_input'] = input_data
        
        # Analisa a intenção do input
        content = input_data.get('content', '')
        parsed_intent = self.intent_parser(content)
        self.current_context['parsed_intent'] = parsed_intent
        logger.info(f"Intenção parseada: {parsed_intent}")
        
        # Retorna status com a intenção parseada
        return {
            'status': 'intent_parsed',
            'parsed_intent': parsed_intent
        } 