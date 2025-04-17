from a3x.core.planner import json_find_gpt  # Assuming planner.py is importable

# Test cases based on expected behavior


def test_find_json_in_code_block_with_specifier():
    """Tests finding JSON within ```json ... ``` block."""
    text = """Some text before
```json
[
  "step1",
  "step2"
]
```
Some text after"""
    expected = """[
  "step1",
  "step2"
]"""
    assert json_find_gpt(text) == expected


def test_find_json_in_code_block_without_specifier():
    """Tests finding JSON within ``` ... ``` block."""
    text = """Blah blah
```
{"key": "value"}
```
More blah"""
    expected = '{"key": "value"}'
    assert json_find_gpt(text) == expected


def test_find_raw_json_list():
    """Tests finding raw JSON list starting with [."""
    text = '["item1", "item2"]'
    expected = '["item1", "item2"]'
    assert json_find_gpt(text) == expected


def test_find_raw_json_object():
    """Tests finding raw JSON object starting with {."""
    text = '{"a": 1, "b": 2}'
    expected = '{"a": 1, "b": 2}'
    assert json_find_gpt(text) == expected


def test_no_json_present():
    """Tests input string with no JSON blocks or structures."""
    text = "This is just a plain text string without any JSON."
    assert json_find_gpt(text) is None


def test_invalid_json_in_block():
    """Tests if it extracts the content even if it's invalid JSON (parsing is separate)."""
    text = """Intro
```json
{"key": "unterminated string
```
Outro"""
    # Note: Extracts the content as is, including the partial marker if not closed
    expected = '{"key": "unterminated string'
    assert json_find_gpt(text) == expected


def test_multiple_json_blocks():
    """Tests finding the *first* JSON block when multiple exist."""
    text = """First block:
```json
["first"]
```
Second block:
```
{"second": true}
```"""
    expected = '["first"]'
    assert json_find_gpt(text) == expected


def test_json_at_start_of_string_block():
    """Tests JSON block at the very beginning."""
    text = """```json
{"start": true}
```
Trailing text."""
    expected = '{"start": true}'
    assert json_find_gpt(text) == expected


def test_json_at_end_of_string_block():
    """Tests JSON block at the very end."""
    text = """Leading text.
```
{"end": true}
```"""
    expected = '{"end": true}'
    assert json_find_gpt(text) == expected


def test_raw_json_with_surrounding_text_should_not_match():
    """Tests that raw JSON surrounded by text isn't matched (needs block or start/end)."""
    text = 'Here is some {"invalid": "JSON"} in the middle.'
    assert json_find_gpt(text) is None


def test_empty_string_input():
    """Tests behavior with an empty input string."""
    text = ""
    assert json_find_gpt(text) is None


def test_json_with_escaped_quotes_in_block():
    """Tests JSON with escaped quotes inside a block."""
    # Using raw string literal r'' might be easier here
    text = r"""Some text ```json
{"quote": "this is a \"quote\" example"}
``` More text"""
    expected = '{"quote": "this is a "quote" example"}'
    assert json_find_gpt(text) == expected
