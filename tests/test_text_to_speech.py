# /home/arthur/Projects/A3X/tests/test_text_to_speech.py
import os
import pathlib
import pytest # Assume pytest está disponível no ambiente de teste
import sys

# Adiciona o diretório raiz ao sys.path para encontrar 'skills'
script_dir = pathlib.Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    from skills.text_to_speech import skill_text_to_speech
except ImportError:
    pytest.skip("Skipping TTS tests, skill_text_to_speech not found.", allow_module_level=True)

# Define o diretório raiz do workspace para construir caminhos relativos seguros
WORKSPACE_ROOT = project_root.resolve()
TEST_OUTPUT_DIR = WORKSPACE_ROOT / "tests" / "output_test_files" / "tts"

# Fixture para garantir que o diretório de saída exista antes do teste
@pytest.fixture(scope="module", autouse=True)
def ensure_output_dir():
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# TODO: Este teste falhará até que um modelo .onnx e o executável Piper válidos sejam configurados.
#       Pode ser necessário mockar subprocess.run ou usar um modelo/executável de teste real.
def test_tts_basic_generation(monkeypatch):
    """
    Testa a geração básica de TTS, verificando se a skill retorna sucesso
    e um caminho de arquivo de saída. Não valida o conteúdo do áudio.
    """
    # Define o caminho para o executável e modelo real
    piper_executable = WORKSPACE_ROOT / "tools" / "piper" / "piper"
    real_model_path = WORKSPACE_ROOT / "tools" / "piper" / "pt_BR-edresson-low.onnx"
    
    # Verifica se os arquivos necessários existem antes de rodar o teste
    if not piper_executable.is_file():
        pytest.skip(f"Piper executable not found at {piper_executable}")
    if not real_model_path.is_file():
        pytest.skip(f"Voice model not found at {real_model_path}")
    if not real_model_path.with_suffix(".onnx.json").is_file():
        pytest.skip(f"Voice model config (.json) not found for {real_model_path}")

    # Define a variável de ambiente PIPER_EXECUTABLE_PATH *apenas para este teste*
    monkeypatch.setenv("PIPER_EXECUTABLE", str(piper_executable))

    action_input = {
        "text": "Olá mundo, este é um teste.",
        "voice_model_path": str(real_model_path.relative_to(WORKSPACE_ROOT)), # Usa o modelo real (caminho relativo)
        "output_dir": str(TEST_OUTPUT_DIR.relative_to(WORKSPACE_ROOT)), # Passa caminho relativo
        "filename": "test_basic_edresson" # Nome de arquivo diferente para evitar conflitos
    }

    result = skill_text_to_speech(action_input)

    print(f"TTS Result: {result}") # Log para debug no pytest

    assert result is not None, "Resultado não pode ser None"
    assert isinstance(result, dict), "Resultado deve ser um dicionário"

    # Verifica se a skill retornou sucesso (mesmo que Piper falhe por config errada,
    # a validação inicial e estrutura devem funcionar). Ajustar asserts conforme necessário.
    # Neste momento, esperamos falha pois Piper não está configurado.
    # Vamos assertar o erro esperado por enquanto, ou pular o teste.
    # Pular é melhor até termos a config.

    # --- Asserts para quando o teste for habilitado e Piper configurado ---
    assert result.get("status") == "success", f"Esperado status 'success', mas recebeu '{result.get('status')}''. Mensagem: {result.get('data', {}).get('message')}"
    assert "data" in result, "Chave 'data' esperada no resultado"
    assert "output_filepath" in result.get("data", {}), "Chave 'output_filepath' esperada nos dados do resultado"
    
    output_file_rel_path = result["data"]["output_filepath"]
    assert isinstance(output_file_rel_path, str), "'output_filepath' deve ser uma string"
    
    output_file_abs_path = WORKSPACE_ROOT / output_file_rel_path
    assert output_file_abs_path.exists(), f"Arquivo de saída esperado não encontrado em: {output_file_abs_path}"
    assert output_file_abs_path.is_file(), f"Caminho de saída não é um arquivo: {output_file_abs_path}"
    assert output_file_abs_path.suffix == ".wav", "Arquivo de saída deve ter extensão .wav"

    # Limpeza (opcional, pode ser feito em fixture teardown)
    if output_file_abs_path.exists():
        try:
            output_file_abs_path.unlink()
        except OSError as e:
            print(f"Warning: Could not delete test output file {output_file_abs_path}: {e}")

# Adicionar mais testes aqui (ex: nome de arquivo customizado, diretório inválido, texto vazio etc.)
