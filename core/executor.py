#!/usr/bin/env python3
"""
Executor Principal do A³X - Núcleo de controle inteligente do sistema.
Responsável por interpretar comandos e acionar os módulos apropriados.
"""

import re
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

# Importações dos módulos
from llm.inference import run_llm
from cli import execute
from core import run_python_code
from memory import store, retrieve

# Configuração de logging
logging.basicConfig(
    filename='logs/executor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Padrões de comando
python_code = re.compile(r'^(rode|execute|execute o comando|rode o comando)\s+python\s+(.+)$', re.IGNORECASE)
shell_command = re.compile(r'^(execute|rode|rode o comando)\s+(.+)$', re.IGNORECASE)
llm_query = re.compile(r'^(.+)$')
memory_store = re.compile(r'^lembre\s+(\w+)\s+(.+)$')
memory_retrieve = re.compile(r'^recupere\s+(\w+)$')

class Executor:
    """
    Executor Principal do A³X.
    Responsável por processar comandos em linguagem natural e acionar os módulos apropriados.
    """
    
    def __init__(self):
        """Inicializa o Executor com suas dependências e configurações."""
        self.command_history: List[Dict[str, Any]] = []
        self.last_result: Optional[str] = None
        self.context: Dict[str, Any] = {}
        
        # Padrões de reconhecimento (ordenados por prioridade)
        self.patterns = {
            'python_code': r'^(run|execute|rode)\s+python\s+',
            'terminal_command': r'^(execute|rode|execute o comando|rode o comando)\s+',
            'memory_store': r'^(lembre|memorize|store)\s+',
            'memory_retrieve': r'^(recupere|busque|retrieve)\s+',
            'question': r'^(qual|como|quando|onde|por que|quem)\s+',
            'instruction': r'^(faça|crie|implemente|desenvolva)\s+'
        }
    
    def _log(self, message: str, level: str = "INFO") -> None:
        """Registra uma mensagem no log interno."""
        print(f"[{level}] {message}")
        logging.info(message)
    
    def _extract_code(self, command: str) -> Optional[str]:
        """Extrai código Python do comando."""
        match = python_code.match(command)
        if match:
            return match.group(2)
        return None
    
    def _extract_command(self, text: str) -> str:
        """Extrai comando shell do texto."""
        # Remove prefixos comuns
        for prefix in ['execute o comando', 'rode o comando', 'execute', 'rode']:
            if text.lower().startswith(prefix):
                command = text[len(prefix):].strip()
                print(f"DEBUG - Comando extraído no Executor: '{command}'")
                return command
        print(f"DEBUG - Comando original no Executor: '{text}'")
        return text
    
    def _extract_key_value(self, text: str) -> tuple[str, str]:
        """Extrai chave e valor do texto para armazenamento."""
        # Remove prefixos
        for prefix in ['lembre', 'memorize', 'store']:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
                break
                
        # Divide em chave e valor
        parts = text.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError("Formato inválido. Use: lembre chave valor")
            
        return parts[0], parts[1]
    
    def process_command(self, input_text: str) -> str:
        """
        Processa um comando em linguagem natural.
        
        Args:
            input_text: Texto do comando
            
        Returns:
            str: Resultado da execução
        """
        try:
            # Registra comando no histórico
            self.command_history.append({
                'input': input_text,
                'timestamp': None  # TODO: Adicionar timestamp
            })
            
            # Identifica tipo de comando
            command_type = None
            for cmd_type, pattern in self.patterns.items():
                if re.match(pattern, input_text.lower()):
                    command_type = cmd_type
                    break
            
            if not command_type:
                # Se não identificou, trata como pergunta
                command_type = 'question'
            
            # Processa comando
            if command_type == 'terminal_command':
                command = self._extract_command(input_text)
                result = execute(command)
                
            elif command_type == 'python_code':
                code = self._extract_code(input_text)
                result = run_python_code(code)
                
            elif command_type == 'memory_store':
                key, value = self._extract_key_value(input_text)
                store(key, value)
                result = "Valor armazenado com sucesso"
                
            elif command_type == 'memory_retrieve':
                key = input_text.split(maxsplit=1)[1]
                result = retrieve(key) or "Chave não encontrada"
                
            elif command_type == 'question':
                result = run_llm(input_text)
                
            elif command_type == 'instruction':
                result = run_llm(input_text)
                
            else:
                result = "Comando não reconhecido"
            
            # Atualiza último resultado
            self.last_result = result
            
            # Registra resultado no histórico
            self.command_history[-1]['result'] = result
            
            return result
            
        except Exception as e:
            error_msg = f"Erro ao processar comando: {str(e)}"
            self._log(error_msg, "ERROR")
            return error_msg

if __name__ == "__main__":
    # Exemplo de uso
    executor = Executor()
    
    # Teste com diferentes tipos de comandos
    test_commands = [
        "Execute o comando ls na pasta atual",
        "Qual é a capital do Brasil?",
        "Lembre que preciso comprar pão",
        "Recupere o que pedi para lembrar sobre pão",
        "rode python print('Hello, World!')"
    ]
    
    for cmd in test_commands:
        print(f"\nProcessando comando: {cmd}")
        result = executor.process_command(cmd)
        print(f"Resultado: {result}") 