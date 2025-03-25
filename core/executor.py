#!/usr/bin/env python3
"""
Executor Principal do A³X - Núcleo de controle inteligente do sistema.
Responsável por interpretar comandos e acionar os módulos apropriados.
"""

import re
from typing import Optional, Dict, Any
from pathlib import Path

# Importações dos módulos
from llm.inference import run_llm
from cli import execute
from core import run_python_code
from memory import store, retrieve

class Executor:
    """
    Executor Principal do A³X.
    Responsável por processar comandos em linguagem natural e acionar os módulos apropriados.
    """
    
    def __init__(self):
        """Inicializa o Executor com suas dependências e configurações."""
        self.command_history: list = []
        self.last_result: Optional[str] = None
        self.context: Dict[str, Any] = {}
        
        # Padrões de reconhecimento
        self.patterns = {
            'terminal_command': r'^(execute|rode|execute o comando|rode o comando)\s+',
            'python_code': r'^(run|execute|rode)\s+python\s+',
            'memory_store': r'^(lembre|memorize|store)\s+',
            'memory_retrieve': r'^(recupere|busque|retrieve)\s+',
            'question': r'^(qual|como|quando|onde|por que|quem)\s+',
            'instruction': r'^(faça|crie|implemente|desenvolva)\s+'
        }
    
    def _log(self, message: str, level: str = "INFO") -> None:
        """Registra uma mensagem no log interno."""
        print(f"[{level}] {message}")
    
    def _think(self, input_text: str) -> Dict[str, Any]:
        """
        [Pensar] Analisa o texto e identifica a intenção.
        
        Args:
            input_text: Texto do comando
            
        Returns:
            Dict com informações sobre a intenção e ação necessária
        """
        self._log(f"Analisando comando: {input_text}")
        
        # Identifica o tipo de comando
        command_type = None
        for cmd_type, pattern in self.patterns.items():
            if re.match(pattern, input_text.lower()):
                command_type = cmd_type
                break
        
        # Se não encontrou padrão, assume ser uma pergunta geral
        if not command_type:
            command_type = 'general_question'
        
        self._log(f"Tipo de comando identificado: {command_type}")
        
        return {
            'type': command_type,
            'original_text': input_text,
            'context': self.context
        }
    
    def _decide(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        [Decidir] Determina qual ação executar baseado na análise.
        
        Args:
            analysis: Resultado da análise do comando
            
        Returns:
            Dict com a decisão tomada e parâmetros necessários
        """
        self._log("Decidindo ação apropriada...")
        
        command_type = analysis['type']
        text = analysis['original_text']
        
        # Mapeia tipo de comando para ação
        action_map = {
            'terminal_command': {
                'module': 'cli',
                'action': 'execute',
                'params': {'command': re.sub(self.patterns['terminal_command'], '', text)}
            },
            'python_code': {
                'module': 'core',
                'action': 'run_python_code',
                'params': {'code': re.sub(self.patterns['python_code'], '', text)}
            },
            'memory_store': {
                'module': 'memory',
                'action': 'store',
                'params': {'key': text.split()[1], 'value': ' '.join(text.split()[2:])}
            },
            'memory_retrieve': {
                'module': 'memory',
                'action': 'retrieve',
                'params': {'key': re.sub(self.patterns['memory_retrieve'], '', text)}
            },
            'question': {
                'module': 'llm',
                'action': 'run_llm',
                'params': {'prompt': text, 'max_tokens': 128}
            },
            'instruction': {
                'module': 'llm',
                'action': 'run_llm',
                'params': {'prompt': text, 'max_tokens': 256}
            },
            'general_question': {
                'module': 'llm',
                'action': 'run_llm',
                'params': {'prompt': text, 'max_tokens': 128}
            }
        }
        
        decision = action_map.get(command_type)
        if not decision:
            decision = {
                'module': 'llm',
                'action': 'run_llm',
                'params': {'prompt': text, 'max_tokens': 128}
            }
        
        self._log(f"Ação decidida: {decision['module']}.{decision['action']}")
        return decision
    
    def _execute(self, decision: Dict[str, Any]) -> str:
        """
        [Executar] Aciona o módulo correspondente com os parâmetros definidos.
        
        Args:
            decision: Decisão tomada com módulo e parâmetros
            
        Returns:
            str: Resultado da execução
        """
        self._log("Executando ação...")
        
        module = decision['module']
        action = decision['action']
        params = decision['params']
        
        try:
            if module == 'cli':
                result = execute(**params)
            elif module == 'core':
                result = run_python_code(**params)
            elif module == 'memory':
                if action == 'store':
                    result = store(**params)
                else:
                    result = retrieve(**params)
            elif module == 'llm':
                result = run_llm(**params)
            else:
                raise ValueError(f"Módulo desconhecido: {module}")
            
            self._log("Ação executada com sucesso")
            return str(result)
            
        except Exception as e:
            self._log(f"Erro na execução: {str(e)}", "ERROR")
            raise
    
    def _evaluate(self, result: str) -> bool:
        """
        [Avaliar] Verifica se o resultado é satisfatório.
        
        Args:
            result: Resultado da execução
            
        Returns:
            bool: True se o resultado é satisfatório
        """
        self._log("Avaliando resultado...")
        
        # Verifica se o resultado está vazio
        if not result or result.isspace():
            self._log("Resultado vazio", "WARNING")
            return False
            
        # Verifica se houve erro explícito
        if "error" in result.lower() or "exception" in result.lower():
            self._log("Erro detectado no resultado", "WARNING")
            return False
            
        self._log("Resultado avaliado como satisfatório")
        return True
    
    def _correct(self, result: str, decision: Dict[str, Any]) -> Optional[str]:
        """
        [Corrigir] Tenta uma abordagem alternativa se necessário.
        
        Args:
            result: Resultado anterior
            decision: Decisão anterior
            
        Returns:
            Optional[str]: Novo resultado ou None se não houver correção
        """
        self._log("Tentando correção...")
        
        # Se for uma pergunta, tenta reformular
        if decision['module'] == 'llm':
            new_params = decision['params'].copy()
            new_params['prompt'] = f"Por favor, reformule sua resposta de forma mais clara e direta: {new_params['prompt']}"
            try:
                return run_llm(**new_params)
            except Exception as e:
                self._log(f"Erro na correção: {str(e)}", "ERROR")
        
        return None
    
    def process_command(self, input_text: str) -> str:
        """
        Processa um comando em linguagem natural.
        
        Args:
            input_text: Texto do comando
            
        Returns:
            str: Resultado do processamento
        """
        try:
            # [Pensar]
            analysis = self._think(input_text)
            
            # [Decidir]
            decision = self._decide(analysis)
            
            # [Executar]
            result = self._execute(decision)
            
            # [Avaliar]
            if not self._evaluate(result):
                # [Corrigir]
                corrected_result = self._correct(result, decision)
                if corrected_result:
                    result = corrected_result
            
            # Atualiza histórico e contexto
            self.command_history.append(input_text)
            self.last_result = result
            self.context['last_command'] = input_text
            self.context['last_result'] = result
            
            return result
            
        except Exception as e:
            error_msg = f"Erro no processamento: {str(e)}"
            self._log(error_msg, "ERROR")
            return f"Preciso de mais contexto para agir com precisão. Pode reformular?\nErro: {error_msg}"

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