#!/usr/bin/env python3
"""
Executor Principal do A³X - Núcleo de controle inteligente do sistema.
Responsável por interpretar comandos e acionar os módulos apropriados.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from memory.system import MemorySystem
from memory.models import SemanticMemoryEntry
from core import code_runner
from core import llm

# Configurar logger
logger = logging.getLogger(__name__)

class Executor:
    """
    Executor de comandos do A³X.
    Responsável por executar diferentes tipos de comandos e manter o histórico.
    """
    
    def __init__(self, memory_system: Optional[MemorySystem] = None):
        """
        Inicializa o executor.
        
        Args:
            memory_system: Sistema de memória para armazenar resultados.
                         Se None, não armazena resultados.
        """
        self.memory_system = memory_system
        self.command_history: List[Dict[str, Any]] = []
        logger.info("Executor inicializado")
        if memory_system:
            logger.info("Sistema de memória configurado")
        else:
            logger.warning("Sistema de memória não configurado")
    
    def execute(self, intent: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Executa uma intenção e retorna o resultado."""
        if not intent:
            return {
                'status': 'error',
                'action': 'unknown',
                'message': 'Intenção não fornecida'
            }
            
        intent_type = intent.get('type')
        
        # Processa o comando e registra no histórico
        result = self.process_command(intent)
        
        # Adiciona entrada no histórico
        command_entry = {
            'timestamp': datetime.now(),
            'intent': intent,
            'result': result
        }
        self.command_history.append(command_entry)
        
        return result
        
    def process_command(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa a intenção do comando e executa."""
        intent_type = intent.get('type')
        
        if intent_type == 'memory':
            return self._handle_memory(intent)
        elif intent_type == 'python':
            return self._handle_python(intent)
        elif intent_type == 'terminal':
            return self._handle_terminal(intent)
        elif intent_type == 'question':
            return self._handle_question(intent)
        elif intent_type == 'instruction':
            return self._handle_instruction(intent)
        else:
            return {
                'status': 'error',
                'action': 'unknown',
                'message': f'Tipo de intenção {intent_type} não suportado'
            }
            
    def _handle_memory(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Processa operações de memória."""
        if not self.memory_system:
            return {
                'status': 'error',
                'action': intent.get('action', 'unknown'),
                'message': 'Sistema de memória não configurado'
            }
            
        action = intent.get('action')
        target = intent.get('target')
        
        if action == 'store':
            content = intent.get('content')
            entry = SemanticMemoryEntry(
                key=target,
                value=content,
                timestamp=datetime.now()
            )
            self.memory_system.store(target, entry)
            return {
                'status': 'success',
                'action': 'store',
                'message': f'Valor armazenado em {target}'
            }
            
        elif action == 'retrieve':
            value = self.memory_system.retrieve(target)
            if value:
                return {
                    'status': 'success',
                    'action': 'retrieve',
                    'value': value
                }
            else:
                return {
                    'status': 'error',
                    'action': 'retrieve',
                    'message': f'Valor não encontrado para chave {target}'
                }
                
        return {
            'status': 'error',
            'action': action,
            'message': f'Ação de memória {action} não suportada'
        }
            
    def _handle_python(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Executa código Python."""
        code = intent.get('content', '').strip()
        result = code_runner.run_python_code(code)
        
        if isinstance(result, dict) and 'error' in result:
            return {
                'status': 'error',
                'action': 'execute_python',
                'message': result['error']
            }
            
        if isinstance(result, str):
            return {
                'status': 'success',
                'action': 'execute_python',
                'output': result.strip()
            }
            
        return {
            'status': 'error',
            'action': 'execute_python',
            'message': 'Resultado inesperado da execução'
        }
        
    def _handle_terminal(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Executa comando no terminal."""
        command = intent.get('content', '').strip()
        result = code_runner.execute_terminal_command(command)
        
        if isinstance(result, dict):
            if 'error' in result:
                return {
                    'status': 'error',
                    'action': 'execute_terminal',
                    'message': result['error']
                }
            elif 'output' in result:
                return {
                    'status': 'success',
                    'action': 'execute_terminal',
                    'output': result['output'].strip()
                }
                
        return {
            'status': 'error',
            'action': 'execute_terminal',
            'message': 'Resultado inesperado do comando'
        }
        
    def _handle_question(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Processa perguntas usando LLM."""
        question = intent.get('content', '').strip()
        response = llm.run_llm(question)
        
        if isinstance(response, str):
            return {
                'status': 'success',
                'action': 'ask',
                'response': response.strip()
            }
            
        return {
            'status': 'error',
            'action': 'ask',
            'message': 'Erro ao processar pergunta'
        }
        
    def _handle_instruction(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Processa instruções usando LLM."""
        instruction = intent.get('content', '').strip()
        response = llm.run_llm(instruction)
        
        if isinstance(response, str):
            return {
                'status': 'success',
                'action': 'instruction',
                'response': response.strip()
            }
            
        return {
            'status': 'error',
            'action': 'instruction',
            'message': 'Erro ao processar instrução'
        }

if __name__ == "__main__":
    # Exemplo de uso
    from memory.system import MemorySystem
    
    # Inicializa o sistema de memória
    memory_system = MemorySystem()
    
    # Cria o executor
    executor = Executor(memory_system=memory_system)
    
    # Teste com diferentes tipos de comandos
    test_intents = [
        {
            'type': 'memory',
            'action': 'store',
            'target': 'test_key',
            'content': 'test_value'
        },
        {
            'type': 'python',
            'content': 'print("Hello, World!")'
        },
        {
            'type': 'terminal',
            'content': 'ls'
        },
        {
            'type': 'question',
            'content': 'Qual é a capital do Brasil?'
        }
    ]
    
    for intent in test_intents:
        print(f"\nExecutando intenção: {intent}")
        result = executor.execute(intent)
        print(f"Resultado: {result}") 