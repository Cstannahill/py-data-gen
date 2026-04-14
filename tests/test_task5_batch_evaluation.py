
import unittest
from unittest.mock import MagicMock, patch
import json
from app.dataset_evaluator import DatasetEvaluator
from app.dataset_generator import DatasetGenerator, GenerationConfig

class TestBatchEvaluation(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MagicMock()
        self.evaluator = DatasetEvaluator(self.mock_provider, "custom")

    def test_validate_heuristics_short_entry(self):
        # Entry too short should fail heuristics
        entry = {"a": "b"} # json.dumps is '{"a": "b"}' which is 10 chars
        is_valid, feedback = self.evaluator._validate_heuristics(entry)
        self.assertFalse(is_valid)
        self.assertIn("too short", feedback.lower())

    def test_validate_heuristics_long_entry(self):
        # Meaningful entry should pass heuristics
        entry = {
            "instruction": "Explain quantum entanglement in simple terms.",
            "input": "",
            "output": "Quantum entanglement is a phenomenon where two particles become connected..."
        }
        is_valid, feedback = self.evaluator._validate_heuristics(entry)
        self.assertTrue(is_valid)
        self.assertEqual(feedback, "")

    def test_evaluate_calls_heuristics(self):
        # Mock _validate_structure to pass, but _validate_heuristics to fail
        entry = {"a": "b"}
        
        with patch.object(self.evaluator, '_validate_structure', return_value=(True, "")):
            # It should fail because it's too short
            is_valid, feedback = self.evaluator.evaluate(entry)
            self.assertFalse(is_valid)
            self.assertIn("too short", feedback.lower())
            
            # _validate_quality should NOT be called
            self.mock_provider.generate.assert_not_called()

    def test_evaluate_batch(self):
        entries = [
            {"instruction": "Valid instruction", "input": "", "output": "Valid output"},
            {"a": "b"}, # Too short
        ]
        
        # Mock LLM for the first entry
        self.mock_provider.generate.return_value = json.dumps({"is_valid": True, "feedback": "Good"})
        
        results = self.evaluator.evaluate_batch(entries)
        
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0][0])  # First should be valid
        self.assertFalse(results[1][0]) # Second should be invalid (heuristics)
        self.assertIn("too short", results[1][1].lower())

    @patch("app.dataset_generator.DatasetEvaluator")
    def test_generator_uses_evaluate_batch(self, MockEvaluator):
        # Setup mock evaluator
        mock_eval_instance = MockEvaluator.return_value
        mock_eval_instance.evaluate_batch.return_value = [
            (True, "Pass"),
            (False, "Fail")
        ]
        
        # Setup generator
        config = GenerationConfig(provider="ollama", model="test", output_dir="test_output")
        generator = DatasetGenerator(config)
        generator.evaluator = mock_eval_instance
        
        # Mock _correct_entry to avoid LLM calls
        generator._correct_entry = MagicMock(return_value=None)
        
        entries = [{"test": 1}, {"test": 2}]
        generator._evaluate_and_correct_batch(entries)
        
        # Verify evaluate_batch was called
        mock_eval_instance.evaluate_batch.assert_called_once_with(entries)
        # Verify _correct_entry was called for the second entry
        generator._correct_entry.assert_called_once_with({"test": 2}, "Fail")

if __name__ == "__main__":
    unittest.main()
