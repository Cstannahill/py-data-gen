"""
Prompt building and analysis
"""

import json
from typing import Optional, Callable


class PromptBuilder:
    def __init__(self, llm_provider):
        self.provider = llm_provider

    def build_generation_prompt(
        self,
        example_dataset: Optional[str],
        constraints: Optional[str],
        goal: str,
        output_format: str,
        entries_per_batch: int,
        *,
        status_cb: Optional[Callable[[str], None]] = None,
        token_cb: Optional[Callable[[str], None]] = None,
    ) -> str:
        if status_cb:
            status_cb("Building analysis prompt")

        analysis_prompt = self._build_analysis_prompt(
            example_dataset=example_dataset,
            constraints=constraints,
            goal=goal,
            output_format=output_format,
        )

        if status_cb:
            status_cb("Calling LLM for analysis insights")

        insights = self.provider.generate(
            analysis_prompt,
            structured_outputs=False,
            on_chunk=token_cb,
        )

        if status_cb:
            status_cb("Building final generation prompt")

        generation_prompt = self._build_final_prompt(
            example_dataset=example_dataset,
            constraints=constraints,
            goal=goal,
            output_format=output_format,
            insights=insights,
            num_entries=entries_per_batch,
        )

        if status_cb:
            status_cb("Generation prompt built")

        return generation_prompt

    def _build_analysis_prompt(
        self,
        example_dataset: Optional[str],
        constraints: Optional[str],
        goal: str,
        output_format: str,
    ) -> str:
        """Build prompt for analyzing inputs and generating insights"""

        prompt = f"""You are an expert in creating high-quality training datasets for language models.

Your task is to analyze the provided information and generate insights that will help create an excellent dataset.

## Dataset Goal
{goal}

## Output Format
{output_format}
"""

        if example_dataset:
            prompt += f"""
## Example Dataset Entries
{example_dataset}
"""

        if constraints:
            prompt += f"""
## Constraints and Requirements
{constraints}
"""

        prompt += """
## Your Task
Analyze the above information and provide:

1. **Key Patterns**: What patterns do you notice in the examples (if provided)? What makes them effective?

2. **Quality Indicators**: What specific qualities should each entry have to achieve the stated goal?

3. **Diversity Strategies**: How can we ensure the dataset has appropriate variety and doesn't become repetitive?

4. **Potential Pitfalls**: What common mistakes should be avoided when generating this type of dataset?

5. **Enhancement Suggestions**: What additional elements or characteristics would make the dataset more effective for the stated goal?

6. **Structural Requirements**: What specific structural elements must each entry contain?

Provide your analysis in a clear, structured format that can be directly used to guide dataset generation.
"""

        return prompt

    def _build_final_prompt(
        self,
        example_dataset: Optional[str],
        constraints: Optional[str],
        goal: str,
        output_format: str,
        insights: str,
        num_entries: int,
    ) -> str:
        """Build the final generation prompt"""

        format_specs = self._get_format_specifications(output_format)
        
        # Conservative limits for better compatibility with local LLMs (e.g. Ollama)
        max_examples_chars = 20000  # ~5k tokens
        max_insights_chars = 10000  # ~2.5k tokens

        safe_insights = insights
        if len(insights) > max_insights_chars:
            safe_insights = insights[:max_insights_chars] + "... (truncated)"

        prompt = f"""You are an expert dataset generator. Your mission is to generate high-quality training data entries.

## OBJECTIVE
{goal}

## OUTPUT FORMAT: {output_format.upper()}
{format_specs}

## GENERATION INSIGHTS
{safe_insights}
"""

        if example_dataset:
            safe_examples = example_dataset
            if len(example_dataset) > max_examples_chars:
                # Try to truncate at a line break
                safe_examples = example_dataset[:max_examples_chars]
                last_line = safe_examples.rfind("\n")
                if last_line != -1:
                    safe_examples = safe_examples[:last_line]
                safe_examples += "\n... (examples truncated to fit context window)"
            
            prompt += f"""
## EXAMPLE ENTRIES
{safe_examples}
"""

        if constraints:
            # Constraints are usually short, but let's be safe
            safe_constraints = constraints if len(constraints) < 5000 else constraints[:5000] + "..."
            prompt += f"""
## CONSTRAINTS AND REQUIREMENTS
{safe_constraints}
"""

        prompt += f"""
## GENERATION GUIDELINES
1. Generate the requested number of dataset entries per batch.
2. Each entry must follow the {output_format} format precisely.
3. Ensure high quality, diversity, and strict adherence to all requirements.
4. Apply the insights provided above to create effective training examples.
5. Avoid repetitive patterns; each entry should contribute unique value.

## OUTPUT SCHEMA
You must return a JSON object with an "entries" key containing an array of dataset entries.
Example:
{{
  "entries": [
    {{ ... entry 1 ... }},
    {{ ... entry 2 ... }}
  ]
}}
"""

        return prompt

    def _get_format_specifications(self, output_format: str) -> str:
        """Get format specifications for different dataset types"""

        if output_format == "sharegpt":
            return """ShareGPT format structure:
{
  "conversations": [
    {"from": "human", "value": "User message here"},
    {"from": "gpt", "value": "Assistant response here"},
    ...
  ]
}

CRITICAL Requirements:
- Use the plural key "conversations" (MUST end with 's').
- Each entry must have a "conversations" array.
- Messages alternate between "human" and "gpt" exactly.
- Each message object must have "from" and "value" fields.
"""

        elif output_format == "alpaca":
            return """Alpaca format structure:
{
  "instruction": "The task or question",
  "input": "Additional context (can be empty string)",
  "output": "The response or completion"
}

Requirements:
- Each entry must have "instruction", "input", and "output" fields.
- Use these exact keys (all lowercase).
- Instruction should be a clear directive.
- Output should be the expert response.
"""

        else:
            # Default to a generic but clear object structure if format is unknown (e.g. "jsonl")
            return f"""The entries should be valid JSON objects representing training examples.
Format: {output_format.upper()}

Requirements:
1. Each entry must be a complete, self-contained JSON object.
2. If no specific schema is provided, use a natural structure appropriate for the goal.
3. Ensure high quality and consistency across entries.
"""

    def build_correction_prompt(
        self, original_entry: dict, feedback: str, output_format: str, goal: str, constraints: Optional[str]
    ) -> str:
        """Build prompt for correcting a failed entry"""

        format_specs = self._get_format_specifications(output_format)

        prompt = f"""You are correcting a dataset entry that failed validation.

## DATASET GOAL
{goal}
"""
        if constraints:
            prompt += f"""
## CONSTRAINTS AND REQUIREMENTS
{constraints}
"""
        prompt += f"""
## FORMAT REQUIRED: {output_format.upper()}
{format_specs}

## ORIGINAL ENTRY (FAILED)
{json.dumps(original_entry, indent=2)}

## VALIDATION FEEDBACK
{feedback}

## YOUR TASK
Fix the entry to address the validation feedback while maintaining the intent and quality of the original.

Requirements:
1. Return ONLY a single corrected entry as valid JSON
2. Address all issues mentioned in the feedback
3. Maintain or improve the quality of the content
4. Ensure strict adherence to the {output_format} format
5. Do not include any text outside the JSON object

Return the corrected entry now:
"""
        return prompt
