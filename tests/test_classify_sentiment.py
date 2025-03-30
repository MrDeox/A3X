# /home/arthur/Projects/A3X/tests/test_classify_sentiment.py
import pytest
import sys
import pathlib

# Adiciona o diretório raiz ao sys.path para encontrar 'skills'
script_dir = pathlib.Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    from skills.classify_sentiment import skill_classify_sentiment
except ImportError as e:
    pytest.skip(f"Skipping sentiment tests, failed to import skill_classify_sentiment: {e}", allow_module_level=True)

# --- Test Cases ---

@pytest.mark.parametrize("text, expected_min_rating, description", [
    ("This is a wonderful library, I love it!", 4, "Positive English"),
    ("That was a terrible experience, very disappointing.", 1, "Negative English"),
    ("Que serviço fantástico, estou muito satisfeito!", 4, "Positive Portuguese"),
    ("A comida estava horrível e o atendimento foi péssimo.", 1, "Negative Portuguese"),
    ("The weather is okay today.", 2, "Neutral-ish English") # Neutral might be 3 stars, check range
])
def test_sentiment_classification_valid(text, expected_min_rating, description):
    """Tests sentiment classification for various valid inputs."""
    action_input = {"text": text}
    result = skill_classify_sentiment(action_input)
    print(f"Sentiment Result ({description}): {result.get('status')}, Rating: {result.get('data', {}).get('sentiment_rating')}, Label: {result.get('data', {}).get('sentiment_label')}")

    assert result["status"] == "success", f"Test Failed ({description}): Status was not success."
    assert result["action"] == "sentiment_classified", f"Test Failed ({description}): Action was incorrect."
    assert "data" in result, f"Test Failed ({description}): Missing data key."
    data = result["data"]

    assert "sentiment_label" in data, f"Test Failed ({description}): Missing sentiment_label."
    assert isinstance(data["sentiment_label"], str), f"Test Failed ({description}): Label should be string."
    assert "star" in data["sentiment_label"], f"Test Failed ({description}): Label format unexpected (should contain 'star' or 'stars')."

    assert "sentiment_rating" in data, f"Test Failed ({description}): Missing sentiment_rating."
    assert isinstance(data["sentiment_rating"], int), f"Test Failed ({description}): Rating should be int."
    assert 1 <= data["sentiment_rating"] <= 5, f"Test Failed ({description}): Rating out of range (1-5)."
    # Check if rating aligns with expectation (e.g., positive >= 4, negative <= 2)
    if expected_min_rating >= 4: # Expecting positive
        assert data["sentiment_rating"] >= expected_min_rating, f"Test Failed ({description}): Expected positive rating (>= {expected_min_rating}), got {data['sentiment_rating']}."
    elif expected_min_rating == 1: # Expecting strictly negative (rating 1 or 2 is also acceptable negative)
        assert data["sentiment_rating"] <= 2, f"Test Failed ({description}): Expected negative rating (<= 2), got {data['sentiment_rating']}."
    else: # Expecting neutral-ish (allow 2, 3, 4 for input expected_min_rating=2)
         assert 2 <= data["sentiment_rating"] <= 4, f"Test Failed ({description}): Expected neutral rating (2-4), got {data['sentiment_rating']}."


    assert "confidence_score" in data, f"Test Failed ({description}): Missing confidence_score."
    assert isinstance(data["confidence_score"], float), f"Test Failed ({description}): Score should be float."
    assert 0.0 <= data["confidence_score"] <= 1.0, f"Test Failed ({description}): Score out of range (0-1)."


def test_sentiment_classification_empty_text():
    """Tests handling of empty text input."""
    action_input = {"text": ""}
    result = skill_classify_sentiment(action_input)
    assert result["status"] == "error"
    assert result["action"] == "sentiment_classification_failed"
    assert "text' parameter is required" in result["data"]["message"]

def test_sentiment_classification_invalid_input():
    """Tests handling of non-string text input."""
    action_input = {"text": ["not a string"]}
    result = skill_classify_sentiment(action_input)
    assert result["status"] == "error"
    assert result["action"] == "sentiment_classification_failed"
    assert "must be a string" in result["data"]["message"]

def test_sentiment_classification_missing_key():
    """Tests handling of missing 'text' key."""
    action_input = {"content": "some other content"}
    result = skill_classify_sentiment(action_input)
    assert result["status"] == "error"
    assert result["action"] == "sentiment_classification_failed"
    assert "text' parameter is required" in result["data"]["message"]

