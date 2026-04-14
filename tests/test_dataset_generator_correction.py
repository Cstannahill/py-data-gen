
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import os
import sys
import json

# Add the project root to sys.path so we can import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.dataset_generator import DatasetGenerator, GenerationConfig

class TestDatasetGeneratorCorrection(unittest.TestCase):
    def setUp(self):
        self.config = GenerationConfig(
            provider="ollama",
            model="test-model",
            example_dataset_path="examples/sample_dataset.jsonl",
            constraints_path="constraints/constraints.txt",
            dataset_goal="Test goal",
            output_dir="./test_output"
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

    def test_correct_entry_passes_goal_and_constraints(self):
        # Entry that needs correction
        entry = {"instruction": "test", "input": "", "output": "test"}
        feedback = "The output is too short."

        # Mock build_correction_prompt to see what it's called with
        self.generator.prompt_builder.build_correction_prompt.return_value = "Mock Correction Prompt"
        
        # Mock provider.generate to avoid real LLM call
        self.generator.provider.generate.return_value = '{"instruction": "test", "input": "", "output": "test corrected"}'
        
        # Mock evaluator.evaluate to pass the corrected entry
        self.generator.evaluator.evaluate.return_value = (True, "")

        # Call the method
        self.generator._correct_entry(entry, feedback)

        # Verify build_correction_prompt was called with correct arguments
        # It should include goal and constraints
        self.generator.prompt_builder.build_correction_prompt.assert_called_with(
            original_entry=entry,
            feedback=feedback,
            output_format=self.config.output_format,
            goal=self.config.dataset_goal,
            constraints="Test constraints" # Loaded from file
        )

if __name__ == "__main__":
    unittest.main()
