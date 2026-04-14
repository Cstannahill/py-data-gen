
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import os
import sys

# Add the project root to sys.path so we can import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.dataset_generator import DatasetGenerator, GenerationConfig

class TestDatasetGeneratorLoading(unittest.TestCase):
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
        Path("examples").mkdir(exist_ok=True)
        Path("constraints").mkdir(exist_ok=True)
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
        for handler in self.generator.logger.handlers[:]:
            handler.close()
            self.generator.logger.removeHandler(handler)

        # Clean up test output and files created for tests
        import shutil
        if Path("./test_output").exists():
            try:
                shutil.rmtree("./test_output")
            except Exception as e:
                print(f"Warning: Could not cleanup test_output: {e}")

    def test_load_file_helper(self):
        # Verify the basic _load_file helper works
        content = self.generator._load_file("constraints/constraints.txt")
        self.assertEqual(content, "Test constraints")
        
        content = self.generator._load_file("non_existent.txt")
        self.assertIsNone(content)

    @patch('app.dataset_generator.DatasetGenerator._load_file')
    def test_build_prompt_loads_files(self, mock_load_file):
        # Mock _load_file to return specific content
        def side_effect(path):
            if path == "examples/sample_dataset.jsonl":
                return "Mock examples"
            if path == "constraints/constraints.txt":
                return "Mock constraints"
            return None
        mock_load_file.side_effect = side_effect
        
        # Mock build_generation_prompt to avoid real LLM call
        self.generator.prompt_builder.build_generation_prompt.return_value = "Mock Prompt"
        
        self.generator.build_prompt()
        
        # Verify both files were loaded
        mock_load_file.assert_any_call("examples/sample_dataset.jsonl")
        mock_load_file.assert_any_call("constraints/constraints.txt")

    @patch('app.dataset_generator.DatasetGenerator._load_file')
    @patch('app.dataset_generator.DatasetGenerator._save_dataset')
    def test_generate_loads_files_when_no_existing_prompt(self, mock_save, mock_load_file):
        # Mock _load_file
        def side_effect(path):
            if path == "examples/sample_dataset.jsonl":
                return "Mock examples"
            if path == "constraints/constraints.txt":
                return "Mock constraints"
            return None
        mock_load_file.side_effect = side_effect
        
        # Mock components to avoid deep execution
        self.generator.prompt_builder.build_generation_prompt.return_value = "Mock Prompt"
        self.generator.config.total_entries = 0 # To exit loop early
        
        self.generator.generate()
        
        # Verify both files were loaded
        mock_load_file.assert_any_call("examples/sample_dataset.jsonl")
        mock_load_file.assert_any_call("constraints/constraints.txt")

if __name__ == "__main__":
    unittest.main()
