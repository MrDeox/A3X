# /home/arthur/Projects/A3X/tests/test_detect_objects.py
import pytest
import sys
import pathlib

# Adiciona o diretório raiz ao sys.path para encontrar 'skills'
script_dir = pathlib.Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Define o diretório raiz do workspace e o caminho da imagem de teste
WORKSPACE_ROOT = project_root.resolve()
TEST_IMAGE_NAME = "ChatGPT Image 30 de mar. de 2025, 14_23_07.png" 
TEST_IMAGE_PATH_REL = pathlib.Path("tests") / "test_data" / TEST_IMAGE_NAME
TEST_IMAGE_PATH_ABS = (WORKSPACE_ROOT / TEST_IMAGE_PATH_REL).resolve()

try:
    # Renomeado para evitar conflito com o nome do módulo
    from skills.detect_objects import skill_detect_objects 
except ImportError as e:
    pytest.skip(f"Skipping detection tests, failed to import skill_detect_objects: {e}", allow_module_level=True)

# Verifica se a imagem de teste existe antes de rodar os testes
@pytest.fixture(scope="module", autouse=True)
def check_test_image():
    if not TEST_IMAGE_PATH_ABS.is_file():
        pytest.skip(f"Test image not found at {TEST_IMAGE_PATH_ABS}", allow_module_level=True)

# Teste 1: Detecção na imagem de texto (espera-se nenhum objeto COCO)
def test_detection_on_text_image():
    """Test object detection on the text image, expecting no COCO objects."""
    action_input = {
        "image_path": str(TEST_IMAGE_PATH_REL), # Passa caminho relativo
        "confidence_threshold": 0.5 # Um limiar razoável
    }
    result = skill_detect_objects(action_input)
    print(f"Detection Result: {result}") # Log
    
    assert result["status"] == "success"
    assert "data" in result
    assert "detected_objects" in result["data"]
    assert isinstance(result["data"]["detected_objects"], list)
    # Para esta imagem específica, esperamos que NENHUM objeto padrão do COCO seja detectado
    assert len(result["data"]["detected_objects"]) == 0, "Expected no objects detected in the text image"

# Teste 2: Caminho inválido
def test_detection_invalid_path():
    """Test with a non-existent image path."""
    action_input = {
        "image_path": "tests/test_data/non_existent_image.png",
    }
    result = skill_detect_objects(action_input)
    assert result["status"] == "error"
    assert "Image file not found" in result["data"]["message"]

# Teste 3: Sem caminho da imagem
def test_detection_missing_path():
    """Test with missing image_path parameter."""
    action_input = {}
    result = skill_detect_objects(action_input)
    assert result["status"] == "error"
    assert "'image_path' parameter is required" in result["data"]["message"]

# TODO: Adicionar um teste com uma imagem que REALMENTE contenha objetos detectáveis (ex: gato, cachorro, carro)
#       e verificar se a lista de resultados contém os objetos esperados com confiança razoável.
# def test_detection_with_objects():
#     # Criar/adicionar uma imagem com objetos (ex: tests/test_data/cat_image.jpg)
#     image_with_objects = "tests/test_data/cat_image.jpg" 
#     action_input = {
#         "image_path": image_with_objects,
#         "confidence_threshold": 0.5
#     }
#     result = skill_detect_objects(action_input)
#     assert result["status"] == "success"
#     assert len(result["data"]["detected_objects"]) > 0
#     # Verificar se um 'cat' foi detectado
#     found_cat = any(obj['class_name'] == 'cat' for obj in result["data"]["detected_objects"])
#     assert found_cat, "Expected to detect a cat in the test image"

