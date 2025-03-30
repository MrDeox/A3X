# /home/arthur/Projects/A3X/tests/test_translate_text.py
import pytest
import sys
import pathlib

# Adiciona o diretório raiz ao sys.path para encontrar 'skills'
script_dir = pathlib.Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    from skills.translate_text import skill_translate_text
except ImportError as e:
    pytest.skip(f"Skipping translation tests, failed to import skill_translate_text: {e}", allow_module_level=True)

# Test case 1: English to Portuguese
@pytest.mark.parametrize("text_en, expected_pt_fragment", [
    ("Hello world", "Olá mundo"),
    ("This is a test.", "Isto é um teste"),
])
def test_translation_en_pt(text_en, expected_pt_fragment):
    """Tests basic English to Portuguese translation."""
    action_input = {
        "text": text_en,
        "source_language": "en",
        "target_language": "pt"
    }
    result = skill_translate_text(action_input)
    
    print(f"EN->PT Result: {result}") # Log for debugging
    assert result["status"] == "success"
    assert "data" in result
    assert "translated_text" in result["data"]
    # Check if the expected fragment is in the result (allows for minor variations)
    assert expected_pt_fragment.lower() in result["data"]["translated_text"].lower()

# Test case 2: Portuguese to English
@pytest.mark.parametrize("text_pt, expected_en_fragment", [
    ("Olá mundo", "world"),
    ("Isto é um teste.", "This is a test"),
])
def test_translation_pt_en(text_pt, expected_en_fragment):
    """Tests basic Portuguese to English translation."""
    action_input = {
        "text": text_pt,
        "source_language": "pt",
        "target_language": "en"
    }
    result = skill_translate_text(action_input)

    print(f"PT->EN Result: {result}") # Log for debugging
    assert result["status"] == "success"
    assert "data" in result
    assert "translated_text" in result["data"]
    assert expected_en_fragment.lower() in result["data"]["translated_text"].lower()

# Test case 3: Same language
def test_translation_same_language():
    """Tests when source and target languages are the same."""
    action_input = {
        "text": "No translation needed.",
        "source_language": "en",
        "target_language": "en"
    }
    result = skill_translate_text(action_input)

    assert result["status"] == "success"
    assert result["action"] == "translation_skipped"
    assert result["data"]["translated_text"] == action_input["text"]

# Test case 4: Missing parameter
def test_translation_missing_param():
    """Tests error handling for missing parameters."""
    action_input = {
        "source_language": "en",
        "target_language": "pt"
    }
    result = skill_translate_text(action_input)
    assert result["status"] == "error"
    assert "text" in result["data"]["message"]

# Test case 5: Unsupported language pair (based on current skill config)
def test_translation_unsupported_pair():
    """Tests error handling for an unsupported language pair."""
    action_input = {
        "text": "This should fail.",
        "source_language": "ja", # Japanese (not configured in skill)
        "target_language": "en"
    }
    result = skill_translate_text(action_input)
    assert result["status"] == "error"
    assert "not supported by NLLB mapping" in result["data"]["message"]

