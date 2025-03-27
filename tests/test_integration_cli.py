import pytest
import subprocess
import os
import sys
import time

# Adiciona o diretório raiz ao path para encontrar o assistant_cli
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Define o caminho para o executável python do venv (ajuste se necessário)
PYTHON_EXEC = os.path.join(project_root, 'venv', 'bin', 'python') # OU sys.executable se rodar pytest de dentro do venv
CLI_SCRIPT = os.path.join(project_root, 'assistant_cli.py')

def run_cli_with_input_file(commands: list[str], filename="temp_test_input.txt") -> str:
    """Helper para criar arquivo de input, rodar CLI e retornar output."""
    filepath = os.path.join(project_root, filename)
    try:
        # Cria arquivo de input
        with open(filepath, 'w', encoding='utf-8') as f:
            for cmd in commands:
                f.write(cmd + '\n')

        # Executa o CLI com timeout maior
        result = subprocess.run(
            [PYTHON_EXEC, CLI_SCRIPT, '-i', filepath],
            capture_output=True,
            text=True,
            check=False, # Não lança exceção no erro do CLI
            timeout=120 # Timeout aumentado para 2 minutos
        )
        return result.stdout + result.stderr # Retorna stdout e stderr combinados
    finally:
        # Limpa o arquivo de input
        if os.path.exists(filepath):
            os.remove(filepath)

def test_generate_then_execute():
    """Testa a sequência de gerar código e executá-lo."""
    commands = [
        "gere código python que define x=100 e y=50 e imprime x",
        "execute o código anterior"
    ]
    
    print("\nIniciando teste de geração e execução de código...")
    start_time = time.time()
    
    output = run_cli_with_input_file(commands, "test_gen_exec.txt")
    elapsed_time = time.time() - start_time
    
    print(f"\nTempo de execução: {elapsed_time:.2f} segundos")
    print(f"\n--- Saída CLI para test_generate_then_execute --- \n{output}\n--------------------------------------------")

    # Verificações básicas na saída (podem precisar de ajuste)
    assert "[Skill: Generate Code]" in output, "Não encontrou a skill de geração de código"
    assert "x = 100" in output or "x=100" in output, "Não encontrou a definição de x"
    assert "y = 50" in output or "y=50" in output, "Não encontrou a definição de y"
    assert "print(x)" in output, "Não encontrou o print de x"
    assert "[Skill: Execute Code (Restricted)]" in output, "Não encontrou a skill de execução de código"
    assert "[Skill: Execute Code] Análise AST passou." in output, "A análise AST falhou"
    assert "[Skill: Execute Code] Executando via Firejail..." in output, "Não encontrou execução via Firejail"
    # Verifica se a saída capturada '100' está presente no log ou na resposta NLG
    assert "[LLM indisponível] Código de" in output or " saída:\n---\n100\n---" in output or "\n100\n" in output, "Não encontrou a saída esperada (100)"

# Adicione mais testes aqui depois... 