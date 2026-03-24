"""
Dataset entry evaluation and validation
"""

import json
import logging
from typing import Tuple, Dict, Any


class DatasetEvaluator:
    """Evaluates dataset entries for structure and quality"""

    def __init__(self, llm_provider, output_format: str):
        self.provider = llm_provider
        self.output_format = output_format
        self.logger = logging.getLogger(__name__)

    def evaluate(self, entry: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluate a dataset entry

        Returns:
            Tuple of (is_valid, feedback)
        """

        # First check structure
        structure_valid, structure_feedback = self._validate_structure(entry)

        if not structure_valid:
            return False, structure_feedback

        # Then check quality
        quality_valid, quality_feedback = self._validate_quality(entry)

        if not quality_valid:
            return False, quality_feedback

        return True, "Entry passed validation"

    def _validate_structure(self, entry: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate entry structure based on format"""

        if self.output_format == "sharegpt":
            return self._validate_sharegpt_structure(entry)
        elif self.output_format == "alpaca":
            return self._validate_alpaca_structure(entry)
        else:
            # For custom formats, do basic validation
            return True, ""

    def _validate_sharegpt_structure(self, entry: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate ShareGPT format structure"""

        # Check for conversations key
        if "conversations" not in entry:
            return False, "Missing 'conversations' key"

        conversations = entry["conversations"]

        # Check if conversations is a list
        if not isinstance(conversations, list):
            return False, "'conversations' must be a list"

        # Check if conversations has at least one exchange
        if len(conversations) < 2:
            return False, "Must have at least one user-assistant exchange"

        # Check alternating pattern and message structure
        expected_from = "human"
        for idx, msg in enumerate(conversations):
            if not isinstance(msg, dict):
                return False, f"Message {idx} is not a dictionary"

            if "from" not in msg:
                return False, f"Message {idx} missing 'from' field"

            if "value" not in msg:
                return False, f"Message {idx} missing 'value' field"

            if msg["from"] not in ["human", "gpt"]:
                return False, f"Message {idx} has invalid 'from' value: {msg['from']}"

            if msg["from"] != expected_from:
                return (
                    False,
                    f"Message {idx} breaks alternating pattern (expected {expected_from}, got {msg['from']})",
                )

            if not msg["value"] or not isinstance(msg["value"], str):
                return False, f"Message {idx} has empty or invalid 'value'"

            # Toggle expected
            expected_from = "gpt" if expected_from == "human" else "human"

        return True, ""

    def _validate_alpaca_structure(self, entry: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate Alpaca format structure"""

        required_fields = ["instruction", "input", "output"]

        for field in required_fields:
            if field not in entry:
                return False, f"Missing required field: '{field}'"

        # Check that fields are strings
        for field in required_fields:
            if not isinstance(entry[field], str):
                return False, f"Field '{field}' must be a string"

        # Check that instruction and output are not empty
        if not entry["instruction"].strip():
            return False, "Field 'instruction' cannot be empty"

        if not entry["output"].strip():
            return False, "Field 'output' cannot be empty"

        # Input can be empty string, but must exist

        return True, ""

    def _validate_quality(self, entry: Dict[str, Any]) -> Tuple[bool, str]:
        """Use LLM to validate entry quality"""

        evaluation_prompt = self._build_quality_evaluation_prompt(entry)
        self.logger.info("Evaluation prompt sent to LLM:\n%s", evaluation_prompt)

        try:
            response = self.provider.generate(
                evaluation_prompt, structured_outputs=False
            )
            self.logger.info("Evaluation response raw:\n%s", response)

            # Parse response
            result = json.loads(response)
            self.logger.info(
                "Evaluation response parsed:\n%s", json.dumps(result, indent=2)
            )

            is_valid = result.get("is_valid", False)
            feedback = result.get("feedback", "No feedback provided")

            return is_valid, feedback

        except Exception as e:
            # If evaluation fails, log but don't reject entry
            self.logger.error(f"Quality evaluation error: {e}")
            return True, f"Quality evaluation error: {e}"

    def _build_quality_evaluation_prompt(self, entry: Dict[str, Any]) -> str:
        """Build prompt for quality evaluation"""

        prompt = f"""You are evaluating the quality of a dataset entry.

## Format: {self.output_format.upper()}

## Entry to Evaluate:
{json.dumps(entry, indent=2)}

## Evaluation Criteria

Assess the entry on these dimensions:

1. **Coherence**: Does the entry make logical sense? Are responses appropriate to inputs?

2. **Completeness**: Is the entry complete? Are there any abrupt endings or missing context?

3. **Quality**: Is the content well-written, informative, and valuable for training?

4. **Naturalness**: Does the interaction feel natural and realistic?

5. **Diversity**: Does this entry contribute unique value (not overly repetitive or generic)?

6. **Safety**: Is the content appropriate and free from harmful, biased, or problematic material?

## Your Task

Evaluate the entry and respond with a JSON object:

{{
  "is_valid": true/false,
  "feedback": "Detailed explanation of issues if invalid, or confirmation if valid"
}}

Requirements:
- Be thorough but fair in your assessment
- Only mark as invalid if there are significant quality issues
- Provide specific, actionable feedback for invalid entries
- Return ONLY the JSON object, no other text

Evaluate now:
"""

        return prompt


class RuleBasedEvaluator(DatasetEvaluator):
    """Faster evaluator that uses only structural rules (no LLM quality check)"""

    def evaluate(self, entry: Dict[str, Any]) -> Tuple[bool, str]:
        """Only validate structure, skip quality check"""
        return self._validate_structure(entry)
