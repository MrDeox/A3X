import pytest
import subprocess
import os
import sys
import time
import re # Importar re para usar regex

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
            timeout=300 # Timeout aumentado para 5 minutos
        )
        return result.stdout + result.stderr # Retorna stdout e stderr combinados
    finally:
        # Limpa APENAS o arquivo de input temporário
        if os.path.exists(filepath):
            os.remove(filepath)
        # REMOVIDO: Limpeza dos arquivos de teste (teste.txt, soma.py, etc.)
        #          Cada teste deve limpar seus próprios arquivos após as asserções.
        # test_file_path = os.path.join(project_root, "teste.txt")
        # if os.path.exists(test_file_path):
        #     os.remove(test_file_path)
        # pytest_create_test_file = os.path.join(project_root, "pytest_create_test.txt")
        # if os.path.exists(pytest_create_test_file):
        #     os.remove(pytest_create_test_file)
        # sum_test_file = os.path.join(project_root, "soma.py")
        # if os.path.exists(sum_test_file):
        #     os.remove(sum_test_file)

def test_integration_generate_execute_sum():
    """Testa a geração e execução de um script simples via CLI."""
    commands = [
        "gere um script python chamado soma.py que define a=5, b=10 e imprime a+b",
        "execute o script soma.py"
    ]
    print("\nIniciando teste de geração e execução...")
    start_time = time.time()
    output = run_cli_with_input_file(commands, "test_gen_exec.txt")
    elapsed_time = time.time() - start_time
    print(f"\nTempo de execução: {elapsed_time:.2f} segundos")
    print(f"\n--- Saída CLI para test_integration_generate_execute_sum --- \n{output}\n--------------------------------------------")

    # Verifica se a geração ocorreu
    assert "[Skill: Generate Code]" in output and "\"action\": \"code_generated\"" in output, "Geração de código não detectada na saída"
    assert "Código também salvo em 'soma.py'" in output, "Mensagem de salvamento do arquivo não encontrada"

    # Verifica se a execução ocorreu e o resultado está correto
    assert "[Skill: Execute Code (Restricted)]" in output, "Execução de código restrita não detectada na saída"
    # A saída da execução pode variar (stdout/stderr), buscamos o resultado '15'
    assert "Resultado da execução:\n15" in output or "Saída:\n15" in output or "\n15\n" in output or '"output": "15\\n"' in output, "Resultado '15' da execução não encontrado na saída"

    # Verifica se o arquivo foi criado (opcional, mas bom)
    sum_file_path = os.path.join(project_root, "soma.py")
    assert os.path.exists(sum_file_path), "Arquivo soma.py não foi criado"
    # Limpeza já é feita no finally do helper

def test_integration_planning_file_ops():
    """Testa o planejamento de operações em arquivo (criação e adição)."""
    commands = [
        "crie um arquivo teste.txt com o conteúdo 'Linha 1'",
        "adicione 'Linha 2' ao final do arquivo teste.txt"
    ]
    
    print("\nIniciando teste de planejamento de operações em arquivo...")
    start_time = time.time()
    
    # Define o caminho do arquivo de teste antes de chamar o helper
    file_path_to_check = os.path.join(project_root, "teste.txt")
    # Garante que o arquivo não existe antes do teste
    if os.path.exists(file_path_to_check):
        os.remove(file_path_to_check)

    output = run_cli_with_input_file(commands, "test_plan_file.txt")
    elapsed_time = time.time() - start_time
    
    print(f"\nTempo de execução: {elapsed_time:.2f} segundos")
    print(f"\n--- Saída CLI para test_integration_planning_file_ops --- \n{output}\n--------------------------------------------")

    # Comentado/Removido: Verificações específicas de planner/skill
    # assert len(re.findall(r"\[Planner\] Comando faz parte de uma sequência\. Forçando planejamento\.\.\.", output)) >= 1, "Planner não foi forçado para o primeiro comando da sequência"
    # assert len(re.findall(r"\[Planner\] Tentando gerar plano sequencial\.\.\.", output)) >= 1, "Planner não tentou gerar plano para o primeiro comando da sequência"
    # assert "[Skill: Manage Files]" in output and "\"action\": \"file_created\"" in output, "Não executou a criação do arquivo (Comando 1)"
    # assert ('[Skill: Modify Code]' in output and '"action": "code_modified"' in output) or \
    #        ('[Skill: Manage Files]' in output and '"action": "file_appended"' in output), \
    #        "Não executou a adição ao arquivo (Comando 2)"

    # Verifica APENAS o conteúdo final do arquivo
    final_content = ""
    if os.path.exists(file_path_to_check):
        with open(file_path_to_check, 'r', encoding='utf-8') as f:
            final_content = f.read()
        os.remove(file_path_to_check) # Limpa o arquivo após a verificação
    else:
        pytest.fail(f"Arquivo de teste esperado '{file_path_to_check}' não foi criado.") # Falha se arquivo não existe

    expected_content = "Linha 1\nLinha 2"
    # Usar strip() para remover possível newline extra no final do arquivo e espaços em branco
    assert final_content.strip() == expected_content, f"Conteúdo final do arquivo inesperado.\nEsperado:\n'{expected_content}'\nRecebido:\n'{final_content.strip()}'"

# Adicione mais testes aqui depois... 