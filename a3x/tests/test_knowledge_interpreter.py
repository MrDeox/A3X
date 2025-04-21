#!/usr/bin/env python
# coding: utf-8

import unittest
import logging

# Adjust the import path based on where this test file lives relative to the core module
# Assuming tests are run from the project root (e.g., python -m unittest discover a3x/tests)
from a3x.a3net.core.knowledge_interpreter_fragment import KnowledgeInterpreterFragment

# Suppress logging during tests if desired
# Enable debug logging for test run
logging.basicConfig(level=logging.DEBUG)
# logging.disable(logging.INFO) # Optionally disable INFO and lower to only see DEBUG

class TestKnowledgeInterpreterFragment(unittest.TestCase):

    def setUp(self):
        """Set up a knowledge interpreter instance for testing."""
        self.interpreter = KnowledgeInterpreterFragment(
            fragment_id="test_interpreter",
            description="Test instance"
        )

    def test_multi_sentence_extraction(self):
        """Test extracting multiple commands from multi-sentence text."""
        test_text = """
        Análise do log: A confiança para frag_alpha foi baixa (0.4). 
        Sugestão: treinar fragmento 'frag_alpha' por 5 épocas.
        Além disso, talvez criar fragmento 'frag_alpha_spec' com base em 'frag_alpha'.
        E também refletir sobre o fragmento 'frag_beta'.
        Por fim, pergunte ao fragmento 'frag_delta' com [0, 1, 0].
        """
        
        expected_commands = [
            "treinar fragmento 'frag_alpha' por 5 épocas",
            "criar fragmento 'frag_alpha_spec' com base em 'frag_alpha'",
            "refletir sobre fragmento 'frag_beta' como a3l",
            "perguntar ao fragmento 'frag_delta' com [0, 1, 0]"
        ]
        
        # Get commands and the sentences list for debugging
        extracted_commands, sentences = self.interpreter.interpret_knowledge(test_text)
        
        # Print sentences for debugging
        print("\n--- Debug: Sentences from test_multi_sentence_extraction ---")
        for i, s in enumerate(sentences):
            print(f"Sentence {i}: \"{s}\"")
        print("--- End Debug ---")
        
        # Use assertCountEqual to compare contents regardless of order, 
        # or assertListEqual if order is strictly important.
        # Since the implementation iterates sentences and then patterns, 
        # the order should be predictable based on sentence order, 
        # so assertListEqual might be appropriate if that order is desired.
        # However, the implementation also removes duplicates, which setUp handles.
        self.assertListEqual(extracted_commands, expected_commands)

    def test_no_commands_extraction(self):
        """Test text with no recognizable commands."""
        test_text = "O sistema parece estável. Nenhuma ação necessária neste momento."
        expected_commands = []
        extracted_commands, _ = self.interpreter.interpret_knowledge(test_text) # Ignore sentences here
        self.assertListEqual(extracted_commands, expected_commands)

    def test_invalid_epochs_extraction(self):
        """Test that invalid epoch numbers are ignored."""
        test_text = "Treinar fragmento 'invalid_epoch' por 0 épocas. Depois treinar 'good_epoch' por 3 épocas."
        expected_commands = ["treinar fragmento 'good_epoch' por 3 épocas"]
        extracted_commands, _ = self.interpreter.interpret_knowledge(test_text) # Ignore sentences here
        self.assertListEqual(extracted_commands, expected_commands)

    def test_duplicate_command_extraction(self):
        """Test that duplicate commands are removed."""
        test_text = "Refletir sobre 'frag_dup'. \nRefletir sobre 'frag_dup'."
        expected_commands = ["refletir sobre fragmento 'frag_dup' como a3l"]
        extracted_commands, _ = self.interpreter.interpret_knowledge(test_text) # Ignore sentences here
        self.assertListEqual(extracted_commands, expected_commands)

if __name__ == '__main__':
    # You can run this file directly to execute the tests
    unittest.main() 