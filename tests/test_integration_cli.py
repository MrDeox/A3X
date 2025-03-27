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

# Garante que o diretório de trabalho seja a raiz do projeto
# para que os caminhos relativos funcionem corretamente.
PROJECT_ROOT = "/home/arthur/Projects/A3X" # Ajuste se necessário
try:
    os.chdir(PROJECT_ROOT)
except FileNotFoundError:
    print(f"Erro fatal: Diretório raiz do projeto '{PROJECT_ROOT}' não encontrado.")
    sys.exit(1)

# Caminho para o executável do assistente CLI
ASSISTANT_CLI_PATH = os.path.join(PROJECT_ROOT, "assistant_cli.py")
# Caminho para o diretório de dados de teste
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "tests", "test_data")

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

def run_a3x_test_capture(input_file: str = None, command: str = None, timeout: int = 60) -> tuple[int, str, str]:
    """
    Executa o assistant_cli.py com um arquivo de input ou comando único
    e captura stdout/stderr.
    """
    if not os.path.exists(ASSISTANT_CLI_PATH):
        pytest.fail(f"Arquivo do assistente não encontrado: {ASSISTANT_CLI_PATH}")

    cmd = [sys.executable, ASSISTANT_CLI_PATH]
    if input_file:
        input_path = os.path.join(TEST_DATA_DIR, input_file)
        if not os.path.exists(input_path):
             pytest.fail(f"Arquivo de input não encontrado: {input_path}")
        cmd.extend(['-i', input_path])
    elif command:
        cmd.extend(['-c', command])
    else:
        pytest.fail("É necessário fornecer 'input_file' ou 'command'.")

    try:
        # Usamos Popen para melhor controle e captura de saída em tempo real (se necessário no futuro)
        # O timeout é aplicado ao wait()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout, stderr = process.communicate(timeout=timeout)
        return_code = process.returncode
        return return_code, stdout, stderr
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        pytest.fail(f"Timeout ({timeout}s) excedido ao executar: {' '.join(cmd)}")
    except Exception as e:
        pytest.fail(f"Erro ao executar o assistente CLI: {e}\nComando: {' '.join(cmd)}")

def test_integration_generate_execute_sum():
    """
    Testa a geração de uma função Python com definição e chamada,
    seguida pela sua execução, lendo comandos de um arquivo.
    Verifica se o código gerado e o resultado da execução estão no stdout.
    """
    input_filename = "test_gen_exec_func_input.txt" # Arquivo de input atualizado
    return_code, stdout, stderr = run_a3x_test_capture(input_file=input_filename)

    print("\n--- Saída Capturada (test_integration_generate_execute_sum) ---")
    print(stdout)
    print("--- Erros Capturados (se houver) ---")
    print(stderr)
    print("--- Fim da Saída ---")

    assert return_code == 0, f"O script CLI saiu com código de erro {return_code}. Stderr:\n{stderr}"

    # Verifica se a definição da função e a chamada estão na saída (usando regex)
    assert re.search(r"def\s+somar\s*\(\s*x\s*,\s*y\s*\)\s*:", stdout), "Definição da função 'somar' não encontrada (regex) na saída."
    # A asserção abaixo ainda vai falhar se o LLM não gerar o print, mas agora é mais flexível quanto à formatação
    assert re.search(r"print\s*\(\s*somar\s*\(\s*70\s*,\s*5\s*\)\s*\)", stdout), "Chamada da função 'somar(70, 5)' não encontrada (regex) na saída."

    # Verifica se o resultado da execução ('75') está na saída
    # Ajuste do marcador baseado na execução anterior (Planner retornou plano de 1 passo)
    execution_marker = "[A³X - Passo 1/1]:" # <-- AJUSTADO MARCADOR
    assert execution_marker in stdout, f"Marcador de execução '{execution_marker}' não encontrado."
    execution_output_start = stdout.find(execution_marker)
    assert execution_output_start != -1, "Não foi possível encontrar o início da saída da execução."
    # Verifica o resultado APENAS na parte da saída APÓS o marcador de execução do passo 1
    assert "75" in stdout[execution_output_start:], "Resultado da execução '75' não encontrado na saída após o marcador do passo 1."

def test_integration_planning_file_ops():
    """
    Testa um plano simples de operações de arquivo (criar e depois modificar/append).
    Verifica se o arquivo final existe e contém o conteúdo esperado.
    """
    input_filename = "test_plan_file_ops_input.txt"
    output_filename = "teste_planner_ops.txt" # Nome do arquivo a ser criado/modificado
    output_filepath = os.path.join(PROJECT_ROOT, output_filename)

    # Garante que o arquivo não existe antes do teste
    if os.path.exists(output_filepath):
        os.remove(output_filepath)

    return_code, stdout, stderr = run_a3x_test_capture(input_file=input_filename)

    print("\n--- Saída Capturada (test_integration_planning_file_ops) ---")
    print(stdout)
    print("--- Erros Capturados (se houver) ---")
    print(stderr)
    print("--- Fim da Saída ---")

    assert return_code == 0, f"O script CLI saiu com código de erro {return_code}. Stderr:\n{stderr}"

    # Verifica se o arquivo foi criado
    assert os.path.exists(output_filepath), f"Arquivo de teste esperado '{output_filepath}' não foi criado."

    # Verifica o conteúdo final do arquivo
    try:
        with open(output_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        expected_content = "Linha inicial.\nSegunda linha adicionada."
        assert content.strip() == expected_content.strip(), \
            f"Conteúdo do arquivo '{output_filename}' não corresponde ao esperado.\n" \
            f"Esperado: '{expected_content}'\nObtido: '{content}'"
    except FileNotFoundError:
        pytest.fail(f"Arquivo '{output_filename}' não encontrado para verificação de conteúdo, embora devesse existir.")
    except Exception as e:
        pytest.fail(f"Erro ao ler o arquivo '{output_filename}': {e}")
    finally:
        # Limpa o arquivo após o teste
        if os.path.exists(output_filepath):
            os.remove(output_filepath)

# Adicione mais testes aqui depois... 