import pytest
import subprocess
from unittest.mock import patch, MagicMock

# Importar a função da skill
from core.code_safety import is_safe_ast
from skills.execute_code import skill_execute_code, PYTHON_EXEC_TIMEOUT

# Configuração básica para os testes
DEFAULT_LANGUAGE = "python"
DEFAULT_CODE = "print('Hello')"

# Mock para subprocess.CompletedProcess
# Permite simular os resultados de subprocess.run
def create_mock_process(stdout="", stderr="", returncode=0):
    proc = MagicMock(spec=subprocess.CompletedProcess)
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc

# == Testes de Sucesso ==

def test_execute_python_success(mocker):
    """Testa a execução bem-sucedida de código Python simples."""
    code = "print('Success!')\nprint('Done')"
    expected_stdout = "Success!\nDone"
    action_input = {"action": "execute", "code": code, "language": DEFAULT_LANGUAGE}

    mocker.patch('skills.execute_code.is_safe_ast', return_value=(True, "Safe"))
    mock_run = mocker.patch('subprocess.run', return_value=create_mock_process(stdout=expected_stdout, returncode=0))

    result = skill_execute_code(action_input)

    assert result['status'] == "success"
    assert result['action'] == "code_executed"
    assert result['data']['stdout'] == expected_stdout
    assert result['data']['stderr'] == ""
    assert result['data']['returncode'] == 0
    assert "Código executado com sucesso" in result['data']['message']

    # Verificar se subprocess.run foi chamado corretamente (verificar args)
    args, kwargs = mock_run.call_args
    assert args[0][0] == "firejail" # Comando base
    assert args[0][-1] == code     # Código passado como último argumento
    assert kwargs.get('timeout') == PYTHON_EXEC_TIMEOUT # Timeout padrão

def test_execute_with_stderr_on_success(mocker):
    """Testa a execução bem-sucedida que também gera stderr (ex: warnings)."""
    code = "import sys; print('Output'); print('Warning', file=sys.stderr)"
    expected_stdout = "Output"
    expected_stderr = "Warning"
    action_input = {"action": "execute", "code": code, "language": DEFAULT_LANGUAGE}

    mocker.patch('skills.execute_code.is_safe_ast', return_value=(True, "Safe"))
    mock_run = mocker.patch('subprocess.run', return_value=create_mock_process(stdout=expected_stdout, stderr=expected_stderr, returncode=0))

    result = skill_execute_code(action_input)

    assert result['status'] == "success"
    assert result['action'] == "code_executed"
    assert result['data']['stdout'] == expected_stdout
    assert result['data']['stderr'] == expected_stderr # Stderr é incluído mesmo no sucesso
    assert result['data']['returncode'] == 0

# == Testes de Falha na Execução (Dentro do Sandbox) ==

def test_execute_python_syntax_error(mocker):
    """Testa a execução de código Python com erro de sintaxe."""
    # Usar aspas triplas para definir o código com erro de sintaxe interna
    code = """print('Hello")"""
    # Usar aspas triplas duplas para evitar conflito com aspas simples no stderr simulado
    expected_stderr_content = "SyntaxError: EOL while scanning string literal"
    simulated_stderr = f'''File "<string>", line 1
    print('Hello")
                 ^
{expected_stderr_content}
'''
    action_input = {"action": "execute", "code": code, "language": DEFAULT_LANGUAGE}

    mocker.patch('skills.execute_code.is_safe_ast', return_value=(True, "Safe")) # AST não pega todos erros de sintaxe
    mock_run = mocker.patch('subprocess.run', return_value=create_mock_process(stdout="", stderr=simulated_stderr, returncode=1))

    result = skill_execute_code(action_input)

    assert result['status'] == "error"
    assert result['action'] == "execution_failed"
    assert result['data']['stdout'] == ""
    # Check for the key part of the error, not the exact multi-line string
    assert expected_stderr_content in result['data']['stderr']
    assert result['data']['returncode'] == 1
    assert "erro de sintaxe" in result['data']['message'] # Verifica a causa inferida
    assert "SyntaxError" in result['data']['message']

def test_execute_python_runtime_error(mocker):
    """Testa a execução de código Python com erro de runtime."""
    code = "x = 1 / 0"
    expected_stderr_content = "ZeroDivisionError: division by zero"
    simulated_stderr = f"""Traceback (most recent call last):
  File "<string>", line 1, in <module>
{expected_stderr_content}
""" # Exemplo
    action_input = {"action": "execute", "code": code, "language": DEFAULT_LANGUAGE}

    mocker.patch('skills.execute_code.is_safe_ast', return_value=(True, "Safe"))
    mock_run = mocker.patch('subprocess.run', return_value=create_mock_process(stdout="", stderr=simulated_stderr, returncode=1))

    result = skill_execute_code(action_input)

    assert result['status'] == "error"
    assert result['action'] == "execution_failed"
    assert result['data']['stdout'] == ""
    # Check for the key part of the error
    assert expected_stderr_content in result['data']['stderr']
    assert result['data']['returncode'] == 1
    assert "erro de runtime" in result['data']['message']
    assert "ZeroDivisionError" in result['data']['message']

# == Testes de Falha na Invocação / Ambiente ==

def test_execute_timeout(mocker):
    """Testa o cenário de timeout durante a execução."""
    code = "import time; time.sleep(2)"
    timeout_value = 1 # Segundos para o teste
    action_input = {"action": "execute", "code": code, "language": DEFAULT_LANGUAGE, "timeout": timeout_value}

    mocker.patch('skills.execute_code.is_safe_ast', return_value=(True, "Safe"))
    # Mock subprocess.run para levantar TimeoutExpired
    mock_run = mocker.patch('subprocess.run', side_effect=subprocess.TimeoutExpired(cmd="firejail ...", timeout=timeout_value))

    result = skill_execute_code(action_input)

    assert result['status'] == "error"
    assert result['action'] == "execution_failed"
    # Compare against float representation of timeout
    assert f"excedeu o limite de tempo ({float(timeout_value)}s)" in result['data']['message']
    assert "stdout" not in result['data'] # Não deve haver stdout/stderr em timeout
    assert "returncode" not in result['data']

    args, kwargs = mock_run.call_args
    assert kwargs.get('timeout') == timeout_value # Verifica se o timeout correto foi passado

def test_execute_firejail_not_found(mocker):
    """Testa o cenário onde o comando firejail não é encontrado."""
    action_input = {"action": "execute", "code": DEFAULT_CODE, "language": DEFAULT_LANGUAGE}

    mocker.patch('skills.execute_code.is_safe_ast', return_value=(True, "Safe"))
    # Mock subprocess.run para levantar FileNotFoundError
    mock_run = mocker.patch('subprocess.run', side_effect=FileNotFoundError("[Errno 2] No such file or directory: 'firejail'"))

    result = skill_execute_code(action_input)

    assert result['status'] == "error"
    assert result['action'] == "execution_failed"
    assert "Erro de ambiente" in result['data']['message']
    assert "'firejail' ou 'python3' não encontrado" in result['data']['message']

# == Testes de Validação de Input e Segurança ==

def test_execute_unsupported_language():
    """Testa a tentativa de executar uma linguagem não suportada."""
    action_input = {"action": "execute", "code": "echo hello", "language": "bash"}

    # Não precisa mockar subprocess pois a validação é antes
    result = skill_execute_code(action_input)

    assert result['status'] == "error"
    assert result['action'] == "execution_failed"
    assert "Linguagem não suportada: 'bash'" in result['data']['message']

def test_execute_missing_code():
    """Testa a chamada sem o parâmetro 'code'."""
    action_input = {"action": "execute", "language": DEFAULT_LANGUAGE}
    result = skill_execute_code(action_input)
    assert result['status'] == "error"
    assert result['action'] == "execution_failed"
    assert "Parâmetro 'code' ausente" in result['data']['message']

def test_execute_invalid_timeout_type(mocker):
    """Testa a chamada com um tipo de timeout inválido."""
    action_input = {"action": "execute", "code": DEFAULT_CODE, "language": DEFAULT_LANGUAGE, "timeout": "dez"}
    mocker.patch('skills.execute_code.is_safe_ast', return_value=(True, "Safe"))
    mock_run = mocker.patch('subprocess.run', return_value=create_mock_process())
    result = skill_execute_code(action_input)
    # A execução deve prosseguir com o timeout padrão
    assert result['status'] == "success" 
    args, kwargs = mock_run.call_args
    assert kwargs.get('timeout') == PYTHON_EXEC_TIMEOUT # Verifica se usou o default

def test_execute_negative_timeout(mocker):
    """Testa a chamada com um timeout negativo."""
    action_input = {"action": "execute", "code": DEFAULT_CODE, "language": DEFAULT_LANGUAGE, "timeout": -5}
    mocker.patch('skills.execute_code.is_safe_ast', return_value=(True, "Safe"))
    mock_run = mocker.patch('subprocess.run', return_value=create_mock_process())
    result = skill_execute_code(action_input)
    # A execução deve prosseguir com o timeout padrão
    assert result['status'] == "success"
    args, kwargs = mock_run.call_args
    assert kwargs.get('timeout') == PYTHON_EXEC_TIMEOUT # Verifica se usou o default

def test_execute_ast_unsafe_import(mocker):
    """Testa a execução de código bloqueado pela AST (import)."""
    code = "import os\nprint(os.listdir('.'))"
    action_input = {"action": "execute", "code": code, "language": DEFAULT_LANGUAGE}

    # <<< NO NEED TO MOCK: Use the real function imported from core.code_safety >>>
    # mocker.patch('skills.execute_code._is_safe_ast', return_value=(False, "Imports não são permitidos (Import)"))
    mock_run = mocker.patch('subprocess.run') # Mock para garantir que não seja chamado

    result = skill_execute_code(action_input)

    assert result['status'] == "error"
    assert result['action'] == "execution_failed"
    assert "Execução bloqueada por motivos de segurança (análise AST)" in result['data']['message']
    # Check for the specific message detail from is_safe_ast (real function)
    assert "Nó AST não permitido: Import" in result['data']['message']
    mock_run.assert_not_called() # Garante que subprocess.run não foi invocado

def test_execute_ast_unsafe_attribute(mocker):
    """Testa a execução de código bloqueado pela AST (acesso a atributo _)."""
    code = "print(().__class__.__bases__)" # Exemplo de acesso perigoso
    action_input = {"action": "execute", "code": code, "language": DEFAULT_LANGUAGE}

    # <<< NO NEED TO MOCK: Use the real function >>>
    mock_run = mocker.patch('subprocess.run')
    result = skill_execute_code(action_input)

    assert result['status'] == "error"
    assert result['action'] == "execution_failed"
    assert "Execução bloqueada por motivos de segurança (análise AST)" in result['data']['message']
    # Check for the specific message detail from is_safe_ast (real function)
    assert "Nó AST não permitido: Attribute" in result['data']['message']
    mock_run.assert_not_called()

def test_execute_ast_syntax_error(mocker):
    """Testa código que falha na própria análise AST por erro de sintaxe."""
    code = "print('Oops"
    action_input = {"action": "execute", "code": code, "language": DEFAULT_LANGUAGE}

    # <<< NO NEED TO MOCK: Use the real function >>>
    mock_run = mocker.patch('subprocess.run')
    result = skill_execute_code(action_input)

    assert result['status'] == "error"
    assert result['action'] == "execution_failed"
    assert "Execução bloqueada por motivos de segurança (análise AST)" in result['data']['message']
    assert "Erro de sintaxe no código:" in result['data']['message']
    mock_run.assert_not_called()
