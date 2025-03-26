"""
Analisador de intenções do A³X.
Responsável por analisar comandos em linguagem natural e identificar suas intenções.
"""

import re
import logging

# Configurar logger
logger = logging.getLogger(__name__)

def analyze_intent(text: str) -> dict:
    """
    Analisa o texto de entrada para identificar a intenção do usuário.
    
    Args:
        text: Texto a ser analisado
        
    Returns:
        dict: Dicionário com a intenção identificada
    """
    if not text:
        return {
            'type': 'unknown',
            'action': 'unknown',
            'target': None,
            'content': text
        }
    
    # Padrões de regex para identificar intenções
    patterns = {
        'greeting': r'^(oi|olá|bom dia|boa tarde|boa noite|hey|hi|hello)[\s!]*$',
        'memory_store': r'(?:guarde|salve|armazene|memorize|lembre)\s+(?:que\s+)?(?:o\s+)?(?:valor\s+)?(?:da\s+)?(?:chave\s+)?([^\s]+)\s+(?:é|como|sendo)\s+(.+)',
        'memory_retrieve': r'(?:recupere|busque|leia|me\s+diga|qual\s+é)\s+(?:o\s+)?(?:valor\s+)?(?:da\s+)?(?:chave\s+)?([^\s]+)',
        'terminal': r'(?:execute|rode|faça)\s+(?:o\s+)?(?:comando\s+)?(?:no\s+)?(?:terminal\s+)?(.+)',
        'python': r'(?:execute|rode)\s+(?:o\s+)?(?:código\s+)?python\s+(.+)',
        'question': r'(?:me\s+diga|me\s+fale|me\s+explique|qual|quem|quando|onde|por\s+que|como|o\s+que|quais)\s+(.+)'
    }
    
    # Tenta identificar a intenção
    for intent_type, pattern in patterns.items():
        match = re.match(pattern, text.lower().strip())
        if match:
            if intent_type == 'greeting':
                return {
                    'type': 'greeting',
                    'action': 'greet',
                    'target': None,
                    'content': match.group(0)
                }
            elif intent_type == 'memory_store':
                return {
                    'type': 'memory',
                    'action': 'store',
                    'target': match.group(1),
                    'content': match.group(2)
                }
            elif intent_type == 'memory_retrieve':
                return {
                    'type': 'memory',
                    'action': 'retrieve',
                    'target': match.group(1),
                    'content': None
                }
            elif intent_type == 'terminal':
                return {
                    'type': 'terminal',
                    'action': 'execute',
                    'target': None,
                    'content': match.group(1)
                }
            elif intent_type == 'python':
                return {
                    'type': 'python',
                    'action': 'execute',
                    'target': None,
                    'content': match.group(1)
                }
            elif intent_type == 'question':
                return {
                    'type': 'question',
                    'action': 'ask',
                    'target': None,
                    'content': match.group(1)
                }
    
    # Se não identificou nenhuma intenção conhecida
    return {
        'type': 'unknown',
        'action': 'unknown',
        'target': None,
        'content': text
    } 