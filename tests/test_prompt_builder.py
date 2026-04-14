
import unittest
import json
import os
import sys

# Add the project root to sys.path so we can import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import MagicMock
from app.prompt_builder import PromptBuilder

class TestPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MagicMock()
        self.prompt_builder = PromptBuilder(self.mock_provider)

    def test_build_correction_prompt_with_goal_and_constraints(self):
        original_entry = {"instruction": "test", "input": "", "output": "test"}
        feedback = "The output is too short."
        output_format = "sharegpt"
        goal = "Generate high-quality coding assistant data."
        constraints = "Avoid generic responses."

        # This should fail initially because PromptBuilder.build_correction_prompt 
        # only accepts 3 arguments.
        prompt = self.prompt_builder.build_correction_prompt(
            original_entry=original_entry,
            feedback=feedback,
            output_format=output_format,
            goal=goal,
            constraints=constraints
        )

        self.assertIn("## DATASET GOAL", prompt)
        self.assertIn(goal, prompt)
        self.assertIn("## CONSTRAINTS AND REQUIREMENTS", prompt)
        self.assertIn(constraints, prompt)
        self.assertIn("## ORIGINAL ENTRY (FAILED)", prompt)
        self.assertIn(json.dumps(original_entry, indent=2), prompt)
        self.assertIn("## VALIDATION FEEDBACK", prompt)
        self.assertIn(feedback, prompt)

    def test_build_correction_prompt_without_constraints(self):
        original_entry = {"instruction": "test", "input": "", "output": "test"}
        feedback = "The output is too short."
        output_format = "sharegpt"
        goal = "Generate high-quality coding assistant data."
        constraints = None

        prompt = self.prompt_builder.build_correction_prompt(
            original_entry=original_entry,
            feedback=feedback,
            output_format=output_format,
            goal=goal,
            constraints=constraints
        )

        self.assertIn("## DATASET GOAL", prompt)
        self.assertIn(goal, prompt)
        self.assertNotIn("## CONSTRAINTS AND REQUIREMENTS", prompt)
        self.assertIn("## ORIGINAL ENTRY (FAILED)", prompt)
        self.assertIn(json.dumps(original_entry, indent=2), prompt)
        self.assertIn("## VALIDATION FEEDBACK", prompt)
        self.assertIn(feedback, prompt)

if __name__ == "__main__":
    unittest.main()
