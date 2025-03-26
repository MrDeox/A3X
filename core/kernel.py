"""Kernel cognitivo do A³X."""

import logging
from typing import Optional, Dict, Any, Callable
from core.executor import Executor
from core.analyzer import analyze_intent
from memory.system import MemorySystem

# Configurar logger
logger = logging.getLogger(__name__)

class CognitiveKernel:
    """
    Kernel cognitivo do A³X.
    Orquestra o fluxo de informações e ativa os módulos cognitivos.
    """
    
    def __init__(
        self,
        executor: Optional[Executor] = None,
        memory_system: Optional[MemorySystem] = None,
        intent_parser: Optional[Callable] = None
    ):
        """
        Inicializa o kernel cognitivo.
        
        Args:
            executor: Executor opcional para processar comandos
            memory_system: Sistema de memória opcional
            intent_parser: Parser de intenção opcional
        """
        self.memory_system = memory_system
        if not memory_system:
            logger.warning("Sistema de memória não configurado")
            
        self.executor = executor
        if executor is None and memory_system is not None:
            self.executor = Executor(memory_system=self.memory_system)
            logger.info("Executor padrão criado com sistema de memória")
            
        self.intent_parser = intent_parser or analyze_intent
        if intent_parser:
            logger.info("Parser de intenção customizado configurado")
        else:
            logger.info("Usando parser de intenção padrão")
            
        self.current_context: Dict[str, Any] = {}
            
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa a entrada do usuário.
        
        Args:
            input_data: Dicionário com os dados de entrada
            
        Returns:
            Dict com o resultado do processamento
        """
        if not input_data or 'content' not in input_data:
            return {
                'status': 'error',
                'message': 'Entrada vazia ou inválida',
                'parsed_intent': None,
                'execution_result': None
            }
            
        content = input_data['content']
        if not content:
            return {
                'status': 'error',
                'message': 'Conteúdo vazio',
                'parsed_intent': None,
                'execution_result': None
            }
        
        # Analisa a intenção
        try:
            parsed_intent = self.intent_parser(content)
        except Exception as e:
            logger.error(f"Erro ao analisar intenção: {str(e)}")
            return {
                'status': 'error',
                'message': f'Erro ao analisar intenção: {str(e)}',
                'parsed_intent': None,
                'execution_result': None
            }
            
        self.current_context['parsed_intent'] = parsed_intent
            
        # Executa se tiver executor configurado
        if self.executor:
            try:
                execution_result = self.executor.execute(intent=parsed_intent)
                self.current_context['execution_result'] = execution_result
                
                return {
                    'status': 'executed',
                    'parsed_intent': parsed_intent,
                    'execution_result': execution_result
                }
            except Exception as e:
                error_result = {
                    'status': 'error',
                    'action': parsed_intent.get('type', 'unknown'),
                    'message': str(e)
                }
                self.current_context['execution_result'] = error_result
                
                return {
                    'status': 'error',
                    'parsed_intent': parsed_intent,
                    'execution_result': error_result
                }
        else:
            error_result = {
                'status': 'error',
                'action': 'unknown',
                'message': 'Executor não configurado'
            }
            self.current_context['execution_result'] = error_result
            
            return {
                'status': 'error',
                'parsed_intent': parsed_intent,
                'execution_result': error_result
            }