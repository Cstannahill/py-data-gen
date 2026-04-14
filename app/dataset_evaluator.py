"""
Dataset entry evaluation and validation
"""

import json
import logging
from typing import Tuple, Dict, Any, List


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

        # Heuristic check
        heuristic_valid, heuristic_feedback = self._validate_heuristics(entry)
        if not heuristic_valid:
            return False, heuristic_feedback

        # Then check quality
        quality_valid, quality_feedback = self._validate_quality(entry)

        if not quality_valid:
            return False, quality_feedback

        return True, "Entry passed validation"

    def evaluate_batch(self, entries: List[Dict[str, Any]]) -> List[Tuple[bool, str]]:
        """
        Evaluate a batch of dataset entries efficiently.
        Performs local structural checks individually, then batches quality checks.
        """
        results: List[Optional[Tuple[bool, str]]] = [None] * len(entries)
        to_eval_quality = []
        quality_indices = []

        # Step 1: Local structural and heuristic checks
        for idx, entry in enumerate(entries):
            is_valid, feedback = self._validate_structure(entry)
            if not is_valid:
                results[idx] = (False, f"Structure: {feedback}")
                continue

            is_valid, feedback = self._validate_heuristics(entry)
            if not is_valid:
                results[idx] = (False, f"Heuristic: {feedback}")
                continue

            # If it passes local checks, queue for LLM quality check
            to_eval_quality.append(entry)
            quality_indices.append(idx)

        # Step 2: Batch LLM quality check
        if to_eval_quality:
            self.logger.info(f"Performing batch quality evaluation for {len(to_eval_quality)} entries")
            quality_results = self._validate_quality_batch(to_eval_quality)

            for idx, res in zip(quality_indices, quality_results):
                results[idx] = res

        # Fallback for any gaps (should not happen)
        final_results = []
        for r in results:
            if r is None:
                # If we still have None, something went wrong in batch logic
                final_results.append((False, "Quality check failed to execute"))
            else:
                final_results.append(r)

        return final_results

    def _validate_quality_batch(self, entries: List[Dict[str, Any]]) -> List[Tuple[bool, str]]:
        """Evaluate quality for multiple entries in a single LLM call."""
        if not entries:
            return []

        # If only one entry, use the standard individual method
        if len(entries) == 1:
            return [self._validate_quality(entries[0])]

        prompt = self._build_batch_quality_evaluation_prompt(entries)
        self.logger.info(f"Batch evaluation prompt (size {len(entries)}) sent to LLM")

        try:
            response = self.provider.generate(prompt, structured_outputs=True)

            # Robust JSON parsing
            text = response.strip()
            
            # Find the JSON array or object
            start_arr = text.find("[")
            start_obj = text.find("{")
            
            if start_arr != -1 and (start_obj == -1 or start_arr < start_obj):
                # Looks like an array
                end_arr = text.rfind("]")
                if end_arr != -1:
                    text = text[start_arr:end_arr+1]
            elif start_obj != -1:
                # Looks like an object
                end_obj = text.rfind("}")
                if end_obj != -1:
                    text = text[start_obj:end_obj+1]

            data = json.loads(text)

            # Handle list vs dict (with key) wrapper
            results_list = []
            if isinstance(data, list):
                results_list = data
            elif isinstance(data, dict):
                # Try common wrapper keys
                for key in ["results", "evaluations", "entries", "data"]:
                    if key in data and isinstance(data[key], list):
                        results_list = data[key]
                        break
                
                # If it's a single object that looks like one evaluation result, 
                # but we expected many, maybe the LLM only evaluated the first one
                if not results_list and "is_valid" in data:
                    self.logger.warning("LLM returned a single evaluation object instead of an array for a batch request.")
                    results_list = [data]

            if not results_list:
                raise ValueError("Could not extract results list from LLM response")

            # Map results back to (is_valid, feedback)
            output = []
            for i in range(len(entries)):
                if i < len(results_list):
                    res = results_list[i]
                    if not isinstance(res, dict):
                        output.append((False, "Invalid evaluation result format (not a dict)"))
                        continue
                    is_valid = res.get("is_valid", False) # Default to False if missing for safety
                    feedback = res.get("feedback", "No feedback provided by LLM")
                    output.append((is_valid, feedback))
                else:
                    # Fallback if LLM returned fewer results than entries
                    output.append((False, "LLM failed to provide feedback for this entry in batch"))

            return output

        except Exception as err:
            self.logger.error(f"Batch quality evaluation error: {err}. Falling back to individual evaluation.")
            # Fallback to individual if batching fails
            # Note: _validate_quality now returns False on error, so this is safe.
            return [self._validate_quality(entry) for entry in entries]

    def _build_batch_quality_evaluation_prompt(self, entries: List[Dict[str, Any]]) -> str:
        """Build a prompt to evaluate multiple entries at once."""
        entries_json = json.dumps(entries, indent=2)

        prompt = f"""You are an expert quality evaluator for AI training datasets.
    Evaluate the following {len(entries)} dataset entries for quality and structure.

    ## Format Context: {self.output_format.upper()}

    ## Entries to Evaluate:
    {entries_json}

    ## Evaluation Criteria:
    For each entry, assess:
    1. **Coherence**: Logical sense and appropriate mapping.
    2. **Completeness**: No abrupt endings or missing context.
    3. **Quality**: Well-written and valuable for training.
    4. **Naturalness**: Realistic interaction.
    5. **Safety**: Free from bias or harmful content.

    ## Your Task:
    Respond with a JSON array containing exactly {len(entries)} objects. 
    Each object must correspond to the entry at the same index and follow this schema:

    {{
    "is_valid": true/false,
    "feedback": "Concise explanation of issues or 'Valid' if perfect"
    }}

    CRITICAL: Return ONLY the JSON array. Ensure exactly {len(entries)} items are in the array.
    """
        return prompt

    def _validate_heuristics(self, entry: Dict[str, Any]) -> Tuple[bool, str]:
        """Fast heuristic pre-check for basic quality"""
        # Example: check if any value is extremely short
        s = json.dumps(entry)
        if len(s) < 20:
            return False, "Entry is too short to be meaningful."
        return True, ""

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

        if "conversations" not in entry:
            return False, "Missing 'conversations' key"

        conversations = entry["conversations"]

        if not isinstance(conversations, list):
            return False, "'conversations' must be a list"

        if len(conversations) < 2:
            return False, "Must have at least one user-assistant exchange"

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

            expected_from = "gpt" if expected_from == "human" else "human"

        return True, ""

    def _validate_alpaca_structure(self, entry: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate Alpaca format structure"""

        required_fields = ["instruction", "input", "output"]

        for field in required_fields:
            if field not in entry:
                return False, f"Missing required field: '{field}'"

        for field in required_fields:
            if not isinstance(entry[field], str):
                return False, f"Field '{field}' must be a string"

        if not entry["instruction"].strip():
            return False, "Field 'instruction' cannot be empty"

        if not entry["output"].strip():
            return False, "Field 'output' cannot be empty"

        return True, ""

    def _validate_quality(self, entry: Dict[str, Any]) -> Tuple[bool, str]:
        """Use LLM to validate entry quality"""

        evaluation_prompt = self._build_quality_evaluation_prompt(entry)
        self.logger.info("Evaluation prompt sent to LLM:\n%s", evaluation_prompt)

        try:
            # Try with structured outputs if available
            response = self.provider.generate(
                evaluation_prompt, structured_outputs=True
            )
            self.logger.info("Evaluation response raw:\n%s", response)

            if not response or not response.strip():
                raise ValueError("Empty response from LLM during quality evaluation")

            # Basic parsing
            text = response.strip()
            
            # Simple balancing to find JSON if model adds prose even with format:json
            if not text.startswith("{"):
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    text = text[start:end+1]

            result = json.loads(text)
            self.logger.info(
                "Evaluation response parsed:\n%s", json.dumps(result, indent=2)
            )

            is_valid = result.get("is_valid", False) # Default to False if missing for safety
            feedback = result.get("feedback", "No feedback provided")

            return is_valid, feedback

        except Exception as e:
            # If evaluation fails, log and reject entry
            self.logger.error(f"Quality evaluation error: {e}")
            return False, f"Quality evaluation error: {e}"

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
        """Validate structure and heuristics, skip quality check"""
        # First check structure
        structure_valid, structure_feedback = self._validate_structure(entry)
        if not structure_valid:
            return False, structure_feedback

        # Heuristic check
        heuristic_valid, heuristic_feedback = self._validate_heuristics(entry)
        if not heuristic_valid:
            return False, heuristic_feedback

        return True, "Entry passed validation"
