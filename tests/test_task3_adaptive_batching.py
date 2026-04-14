
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import os
import sys
import json

# Add the project root to sys.path so we can import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.dataset_generator import DatasetGenerator, GenerationConfig

class TestTask3AdaptiveBatching(unittest.TestCase):
    def setUp(self):
        self.config = GenerationConfig(
            provider="ollama",
            model="test-model",
            example_dataset_path="examples/sample_dataset.jsonl",
            constraints_path="constraints/constraints.txt",
            dataset_goal="Test goal",
            output_dir="./test_output",
            batch_size=10,
            total_entries=10
        )
        
        # Ensure example files exist for the test
        Path("examples").mkdir(exist_ok=True, parents=True)
        Path("constraints").mkdir(exist_ok=True, parents=True)
        Path("examples/sample_dataset.jsonl").write_text('{"test": "data"}', encoding="utf-8")
        Path("constraints/constraints.txt").write_text('Test constraints', encoding="utf-8")
        
        # Mock dependencies to avoid real LLM calls and complex setup
        with patch('app.dataset_generator.OllamaProvider'), \
             patch('app.dataset_generator.PromptBuilder'), \
             patch('app.dataset_generator.DatasetEvaluator'), \
             patch('app.dataset_generator.ProgressTracker'):
            self.generator = DatasetGenerator(self.config)

    def tearDown(self):
        # Close logging handlers to release file locks
        import logging
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
        if hasattr(self, 'generator') and hasattr(self.generator, 'logger'):
            for handler in self.generator.logger.handlers[:]:
                handler.close()
                self.generator.logger.removeHandler(handler)

    def test_generate_batch_includes_diversity_hint(self):
        """Step 2: Verify that _generate_batch injects a diversity hint into the prompt."""
        self.generator.generation_prompt = "Base Prompt"
        # Make the first call return a valid JSON so it doesn't go into correction
        self.generator.provider.generate.return_value = '[{"id": 1}]'
        
        self.generator._generate_batch(5)
        
        # Check if the prompt sent to provider contains "DIVERSITY HINT:"
        found_hint_in_any_call = False
        for call in self.generator.provider.generate.call_args_list:
            # Handle both positional and keyword arguments
            args, kwargs = call
            prompt = kwargs.get("prompt") or (args[0] if args else "")
            if "DIVERSITY HINT:" in prompt:
                found_hint_in_any_call = True
                break
        
        self.assertTrue(found_hint_in_any_call, "DIVERSITY HINT: not found in any LLM call")
        
        # Verify it's one of the expected hints
        diversity_hints = [
            "Focus on edge cases and rare scenarios.",
            "Use a different tone or persona than previous entries.",
            "Ensure the examples are complex and multi-faceted.",
            "Keep the interaction brief and highly direct.",
            "Incorporate unexpected but valid user inputs."
        ]
        found_expected_hint = False
        for call in self.generator.provider.generate.call_args_list:
            args, kwargs = call
            prompt = kwargs.get("prompt") or (args[0] if args else "")
            for hint in diversity_hints:
                if f"DIVERSITY HINT: {hint}" in prompt:
                    found_expected_hint = True
                    break
            if found_expected_hint:
                break
        self.assertTrue(found_expected_hint, "No expected diversity hint found in any prompt")

    def test_generate_adaptive_batching_reduces_size(self):
        """Step 1: Verify that generate() reduces batch size when too few entries are returned."""
        # Set total entries to 10, initial batch size to 10
        self.generator.config.total_entries = 10
        self.generator.config.batch_size = 10
        
        # Mock internal methods to control flow
        with patch.object(self.generator, '_generate_batch') as mock_gen, \
             patch.object(self.generator, '_evaluate_and_correct_batch') as mock_eval, \
             patch.object(self.generator, '_save_dataset'), \
             patch.object(self.generator, '_load_analysis_inputs') as mock_load:
            
            mock_load.return_value = ("examples", "constraints")
            self.generator.prompt_builder.build_generation_prompt.return_value = "Test"
            # First call returns 2 entries (less than 10 // 2)
            # Second call should use batch size 5
            mock_gen.side_effect = [
                [{"id": 1}, {"id": 2}], # 1st call: returns 2
                [{"id": 3}]            # 2nd call: returns 1
            ]
            mock_eval.side_effect = [
                [{"id": 1}, {"id": 2}], 
                [{"id": 3}]
            ]
            self.generator.generation_prompt = "Test"
            
            # We need to make sure it doesn't run forever
            self.generator.config.total_entries = 3
            
            self.generator.generate()
            
            # Check mock_gen calls
            self.assertEqual(mock_gen.call_count, 2)
            
            # 1st call: entries_needed = 3 (min of 10 and 3)
            self.assertEqual(mock_gen.call_args_list[0][0][0], 3)
            
    def test_generate_adaptive_batching_logic_deep(self):
        """Step 1: More precise check on current_batch_size reduction."""
        self.generator.config.total_entries = 20
        self.generator.config.batch_size = 10
        
        with patch.object(self.generator, '_generate_batch') as mock_gen, \
             patch.object(self.generator, '_evaluate_and_correct_batch') as mock_eval, \
             patch.object(self.generator, '_save_dataset'), \
             patch.object(self.generator, '_load_analysis_inputs') as mock_load:
            
            mock_load.return_value = ("examples", "constraints")
            self.generator.prompt_builder.build_generation_prompt.return_value = "Test"
            # 1st call: needs 10, returns 4 ( < 10 // 2 ). current_batch_size should become 5.
            # 2nd call: needs min(5, 20-4) = 5.
            mock_gen.side_effect = [
                [{"id": i} for i in range(4)], # 4 entries
                [{"id": i} for i in range(16)] # finish it
            ]
            mock_eval.side_effect = [
                [{"id": i} for i in range(4)],
                [{"id": i} for i in range(16)]
            ]
            self.generator.generation_prompt = "Test"
            
            self.generator.generate()
            
            # 1st call entries_needed = 10
            self.assertEqual(mock_gen.call_args_list[0][0][0], 10)
            # 2nd call entries_needed = 5 (due to reduction)
            self.assertEqual(mock_gen.call_args_list[1][0][0], 5)

if __name__ == "__main__":
    unittest.main()
