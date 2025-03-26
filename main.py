#!/usr/bin/env python3
"""
Interface interativa do A³X - Executor de comandos em linguagem natural.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from core import Executor

# Configuração de cores
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Configuração de diretórios
DATA_DIR = Path("data")
HISTORY_FILE = DATA_DIR / "history.json"

class A3XInterface:
    """Interface interativa do A³X."""
    
    def __init__(self):
        """Inicializa a interface."""
        self.executor = Executor()
        self.history = []
        self.history_file = Path("data/history.json")
        self.is_multiline = False
        self.multiline_buffer = []
        self._load_history()
        
    def _load_history(self):
        """Carrega o histórico de comandos."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except json.JSONDecodeError:
                print(f"{Colors.WARNING}Aviso: Histórico corrompido, iniciando novo histórico.{Colors.ENDC}")
                self.history = []
    
    def _save_history(self):
        """Salva o histórico de comandos."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"{Colors.WARNING}Aviso: Não foi possível salvar o histórico: {str(e)}{Colors.ENDC}")
            
    def _format_time(self, seconds):
        """Formata o tempo de execução."""
        if seconds < 1:
            return f"{seconds*1000:.1f}ms"
        return f"{seconds:.2f}s"
        
    def _show_help(self):
        """Mostra a ajuda."""
        print(f"\n{Colors.BOLD}A³X - Interface Interativa{Colors.ENDC}")
        print("\nComandos disponíveis:")
        print(f"{Colors.GREEN}!help{Colors.ENDC} - Mostra esta ajuda")
        print(f"{Colors.GREEN}!clear{Colors.ENDC} - Limpa o terminal")
        print(f"{Colors.GREEN}!exit{Colors.ENDC} - Sai do programa")
        print(f"{Colors.GREEN}!history{Colors.ENDC} - Mostra o histórico de comandos")
        print("\nModo multilinha:")
        print("Digite ... para entrar no modo multilinha")
        print("Digite >>> para sair do modo multilinha")
        print("\nExemplos de uso:")
        print("1. Cálculo simples: 2 + 2")
        print("2. Modo multilinha:")
        print("   ...")
        print("   x = 10")
        print("   y = 20")
        print("   print(x + y)")
        print("   >>>")
        print("\nDica: Use Ctrl+C para interromper um comando\n")
        
    def _get_next_command(self):
        """Obtém o próximo comando do usuário."""
        if self.is_multiline:
            while True:
                try:
                    line = input(f"{Colors.CYAN}...{Colors.ENDC} " if not self.multiline_buffer else f"{Colors.CYAN}...{Colors.ENDC} ")
                    if line.strip() == ">>>":
                        break
                    self.multiline_buffer.append(line)
                except KeyboardInterrupt:
                    print("\nModo multilinha cancelado.")
                    self.multiline_buffer = []
                    self.is_multiline = False
                    return None
            command = "\n".join(self.multiline_buffer)
            self.multiline_buffer = []
            self.is_multiline = False
            return command
        else:
            try:
                command = input(f"{Colors.BLUE}A³X>{Colors.ENDC} ").strip()
                if command == "...":
                    self.is_multiline = True
                    return self._get_next_command()
                return command
            except KeyboardInterrupt:
                print("\nComando cancelado.")
                return None
    
    def _process_special_command(self, command):
        """Processa comandos especiais."""
        if command == "!help":
            self._show_help()
        elif command == "!clear":
            self._clear_screen()
        elif command == "!exit":
            print(f"\n{Colors.GREEN}Até logo!{Colors.ENDC}")
            self._save_history()
            raise SystemExit(0)
        elif command == "!history":
            if not self.history:
                print(f"\n{Colors.WARNING}Nenhum comando no histórico.{Colors.ENDC}\n")
            else:
                print(f"\n{Colors.BOLD}Histórico de comandos:{Colors.ENDC}")
                for i, entry in enumerate(self.history[-10:], 1):
                    print(f"{i}. {entry['command']} ({entry['timestamp']})")
                print()
        else:
            return False
        return True
    
    def _process_command(self, command):
        """Processa um comando."""
        start_time = time.time()
        result = self.executor.process_command(command)
        execution_time = time.time() - start_time
        
        # Salva no histórico
        self.history.append({
            'command': command,
            'result': str(result),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'execution_time': execution_time
        })
        self._save_history()
        
        return result, execution_time
    
    def _clear_screen(self):
        """Limpa o terminal."""
        os.system('clear' if os.name == 'posix' else 'cls')
        
    def run(self):
        """Executa o loop principal da interface."""
        self._clear_screen()
        print(f"{Colors.BOLD}Bem-vindo ao A³X!{Colors.ENDC}")
        print("Digite !help para ver os comandos disponíveis.")
        
        while True:
            try:
                command = self._get_next_command()
                if not command:
                    continue
                
                if command.startswith('!'):
                    if self._process_special_command(command):
                        continue
                
                result, execution_time = self._process_command(command)
                print(f"\n{Colors.GREEN}Resultado:{Colors.ENDC}")
                print(result)
                print(f"\n{Colors.CYAN}Tempo de execução: {self._format_time(execution_time)}{Colors.ENDC}\n")
                
            except KeyboardInterrupt:
                print("\nUse !exit para sair.")
            except Exception as e:
                print(f"\n{Colors.FAIL}Erro: {str(e)}{Colors.ENDC}\n")

if __name__ == "__main__":
    interface = A3XInterface()
    interface.run() 