"""
LLM Dataset Generator
A modular system for generating high-quality training datasets using LLMs
"""

import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from llm_providers import OllamaProvider, LLMProvider
from prompt_builder import PromptBuilder
from dataset_evaluator import DatasetEvaluator
from progress_tracker import ProgressTracker


@dataclass
class GenerationConfig:
    """Configuration for dataset generation"""

    provider: str = "ollama"
    model: str = "gpt-oss-20B"
    use_thinking: bool = True
    use_structured_outputs: bool = True
    total_entries: int = 100
    batch_size: int = 10
    max_correction_attempts: int = 3
    output_format: str = "sharegpt"  # sharegpt or alpaca

    example_dataset_path: Optional[str] = None
    constraints_path: Optional[str] = None
    dataset_goal: str = ""

    # Path to existing prompt file (skips analysis if provided)
    existing_prompt_path: Optional[str] = None

    output_dir: str = "./output"
    save_prompt: bool = True
    save_failures: bool = True


class DatasetGenerator:
    """Main orchestrator for dataset generation"""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.provider = self._init_provider()
        self.prompt_builder = PromptBuilder(self.provider)
        self.evaluator = DatasetEvaluator(self.provider, config.output_format)
        self.tracker = ProgressTracker(config.total_entries, config.batch_size)

        # Storage
        self.dataset: List[Dict] = []
        self.failures: List[Dict] = []
        self.generation_prompt: Optional[str] = None

    def _setup_logging(self):
        log_file = (
            self.output_dir
            / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),  # <-- fix
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _init_provider(self) -> LLMProvider:
        """Initialize LLM provider"""
        if self.config.provider == "ollama":
            return OllamaProvider(
                model=self.config.model,
                use_thinking=self.config.use_thinking,
                use_structured_outputs=self.config.use_structured_outputs,
            )
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _load_file(self, path: Optional[str]) -> Optional[str]:
        """Load content from file"""
        if not path:
            return None

        p = Path(path)

        # 1) Best default for modern text files (JSON/JSONL/etc.)
        try:
            return p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # 2) Common Windows/Excel/BOM case
            try:
                return p.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                # 3) Last-resort fallback so you can at least keep going
                #    (will replace undecodable bytes with �)
                self.logger.warning(
                    f"Could not decode file {path} as UTF-8. Falling back with replacement."
                )
                return p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            self.logger.warning(f"Could not load file {path}: {e}")
            return None

    def generate(self) -> List[Dict]:
        """Main generation workflow"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING DATASET GENERATION")
        self.logger.info("=" * 60)

        # Step 1 & 2: Analyze inputs and generate prompt
        self.logger.info("\n[STEP 1-2] Analyzing inputs and generating prompt...")

        # NEW: reuse a previously generated prompt if provided
        if self.config.existing_prompt_path:
            prompt_path = Path(self.config.existing_prompt_path)
            self.tracker.update_status(f"Loading existing prompt: {prompt_path}")

            if not prompt_path.exists():
                raise FileNotFoundError(
                    f"existing_prompt_path not found: {prompt_path}"
                )

            self.generation_prompt = prompt_path.read_text(
                encoding="utf-8", errors="replace"
            )
            self.logger.info(
                f"Loaded existing prompt: {len(self.generation_prompt)} chars"
            )

        else:
            self.tracker.update_status("Loading example dataset")
            example_data = self._load_file(self.config.example_dataset_path)
            self.logger.info(
                f"✓ Examples loaded: {len(example_data) if example_data else 0} chars"
            )

            self.tracker.update_status("Loading constraints")
            constraints = self._load_file(self.config.constraints_path)
            self.logger.info(
                f"✓ Constraints loaded: {len(constraints) if constraints else 0} chars"
            )

            # Throttled heartbeat while analysis is streaming
            analysis_chars = 0
            last_ui = time.monotonic()

            def on_analysis_chunk(s: str):
                nonlocal analysis_chars, last_ui
                analysis_chars += len(s)
                now = time.monotonic()
                if now - last_ui >= 1.0:  # update at most 1x/sec
                    self.tracker.update_status(
                        f"Analyzing inputs (LLM streaming… {analysis_chars} chars)"
                    )
                    last_ui = now

            self.tracker.update_status("Building generation prompt")

            t0 = time.perf_counter()

            self.generation_prompt = self.prompt_builder.build_generation_prompt(
                example_dataset=example_data,
                constraints=constraints,
                goal=self.config.dataset_goal,
                output_format=self.config.output_format,
                entries_per_batch=self.config.batch_size,
                status_cb=self.tracker.update_status,  # milestone updates
                token_cb=on_analysis_chunk,  # streaming heartbeat
            )
            dt = time.perf_counter() - t0

            self.logger.info(
                f"✓ Generation prompt built in {dt:.1f}s ({len(self.generation_prompt)} chars)"
            )
            self.tracker.update_status("Prompt ready; starting batch generation")

            if self.config.save_prompt:
                prompt_file = self.output_dir / "generation_prompt.txt"
                prompt_file.write_text(
                    self.generation_prompt, encoding="utf-8", errors="replace"
                )
                self.logger.info(f"✓ Prompt saved to: {prompt_file}")

        # Step 3: Generate dataset entries in batches
        total_batches = (
            self.config.total_entries + self.config.batch_size - 1
        ) // self.config.batch_size

        for batch_num in range(total_batches):
            entries_needed = min(
                self.config.batch_size, self.config.total_entries - len(self.dataset)
            )

            if entries_needed <= 0:
                break

            self.logger.info(
                f"\n[STEP 3] Generating batch {batch_num + 1}/{total_batches}"
            )
            self.tracker.update_status(
                f"Generating batch {batch_num + 1}/{total_batches}"
            )

            batch_entries = self._generate_batch(entries_needed)

            # Step 4: Evaluate and correct entries
            self.logger.info(f"[STEP 4] Evaluating batch {batch_num + 1}")
            self.tracker.update_status(f"Evaluating batch {batch_num + 1}")

            validated_entries = self._evaluate_and_correct_batch(batch_entries)

            # Add validated entries to dataset
            self.dataset.extend(validated_entries)
            self.tracker.add_entries(len(validated_entries))

            self.logger.info(
                f"✓ Batch {batch_num + 1} complete: {len(validated_entries)}/{entries_needed} entries added"
            )
            self.logger.info(
                f"Progress: {len(self.dataset)}/{self.config.total_entries} total entries"
            )

        # Save final dataset
        self._save_dataset()

        # Step 5: Save failure log
        if self.config.save_failures and self.failures:
            self._save_failures()

        self.logger.info("\n" + "=" * 60)
        self.logger.info("GENERATION COMPLETE")
        self.logger.info(f"Total entries generated: {len(self.dataset)}")
        self.logger.info(f"Total failures: {len(self.failures)}")
        self.logger.info("=" * 60)

        return self.dataset

    def _generate_batch(self, num_entries: int) -> List[Dict]:
        try:
            if not self.generation_prompt:
                raise ValueError("Generation prompt is not set.")

            prompt = (
                self.generation_prompt
                + "\n\n"
                + f"## BATCH REQUEST\nGenerate exactly {num_entries} NEW entries now.\n"
                + "Return ONLY valid JSON (an array of entries). No commentary.\n"
            )
            prompt_preview = prompt[:100].replace("\n", "\\n")
            self.logger.info(
                f"Sending generation prompt to LLM ({len(prompt)} chars) preview: {prompt_preview}"
            )
            response = self.provider.generate(prompt)  # <-- no context dict
            self.logger.info("✓ Generation response received from LLM")
            self.logger.info("Generation response raw:\n%s", response)
            entries = self._parse_generation_response(response)

            self.logger.info(f"Generated {len(entries)} entries")
            self.logger.info(
                "Generation response parsed:\n%s", json.dumps(entries, indent=2)
            )
            return entries

        except Exception as e:
            self.logger.error(f"Error generating batch: {e}")
            return []

    def _parse_generation_response(self, response: str) -> List[Dict]:
        """Parse LLM response to extract dataset entries"""
        try:
            # Try to parse as JSON array
            if response.strip().startswith("["):
                return json.loads(response)

            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
                return json.loads(json_str)

            # Try to parse as single JSON object
            return [json.loads(response)]

        except Exception as e:
            self.logger.error(f"Error parsing generation response: {e}")
            return []

    def _evaluate_and_correct_batch(self, entries: List[Dict]) -> List[Dict]:
        """Evaluate entries and correct failures"""
        validated = []

        for idx, entry in enumerate(entries):
            self.logger.info(f"  Evaluating entry {idx + 1}/{len(entries)}")

            is_valid, feedback = self.evaluator.evaluate(entry)

            if is_valid:
                validated.append(entry)
                self.logger.info(f"  ✓ Entry {idx + 1} passed")
            else:
                self.logger.warning(f"  ✗ Entry {idx + 1} failed: {feedback}")

                # Attempt correction
                corrected = self._correct_entry(entry, feedback)

                if corrected:
                    validated.append(corrected)
                    self.logger.info(f"  ✓ Entry {idx + 1} corrected and passed")
                else:
                    self._log_failure(entry, feedback, "correction_failed")
                    self.logger.error(f"  ✗ Entry {idx + 1} could not be corrected")

        return validated

    def _correct_entry(self, entry: Dict, feedback: str) -> Optional[Dict]:
        """Attempt to correct a failed entry"""
        correction_prompt = self.prompt_builder.build_correction_prompt(
            original_entry=entry,
            feedback=feedback,
            output_format=self.config.output_format,
        )
        self.logger.info("Correction prompt sent to LLM:\n%s", correction_prompt)

        for attempt in range(self.config.max_correction_attempts):
            try:
                self.tracker.update_status(f"Correcting entry (attempt {attempt + 1})")

                response = self.provider.generate(correction_prompt)
                self.logger.info(
                    "Correction response raw (attempt %s):\n%s",
                    attempt + 1,
                    response,
                )
                corrected_entries = self._parse_generation_response(response)
                self.logger.info(
                    "Correction response parsed (attempt %s):\n%s",
                    attempt + 1,
                    json.dumps(corrected_entries, indent=2),
                )

                if not corrected_entries:
                    continue

                corrected = corrected_entries[0]
                is_valid, new_feedback = self.evaluator.evaluate(corrected)

                if is_valid:
                    return corrected

                feedback = new_feedback

            except Exception as e:
                self.logger.error(f"Error during correction attempt {attempt + 1}: {e}")

        return None

    def _log_failure(self, entry: Dict, reason: str, failure_type: str):
        """Log a failed entry"""
        self.failures.append(
            {
                "entry": entry,
                "reason": reason,
                "type": failure_type,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def _save_dataset(self):
        """Save the generated dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            self.output_dir / f"dataset_{self.config.output_format}_{timestamp}.json"
        )

        with open(filename, "w") as f:
            json.dump(self.dataset, f, indent=2)

        self.logger.info(f"✓ Dataset saved to: {filename}")

    def _save_failures(self):
        """Save failure log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"failures_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(self.failures, f, indent=2)

        self.logger.info(f"✓ Failure log saved to: {filename}")


def main():
    """Example usage"""
    config = GenerationConfig(
        provider="ollama",
        model="gpt-oss-20B-heretic",
        total_entries=50,
        batch_size=5,
        dataset_goal="Create a dataset for training an assistant that helps users plan projects comprehensively",
        example_dataset_path="examples/sample_dataset.json",
        constraints_path="examples/constraints.txt",
        output_format="sharegpt",
        output_dir="./output",
    )

    generator = DatasetGenerator(config)
    dataset = generator.generate()

    print(f"\nGeneration complete! Created {len(dataset)} entries.")


if __name__ == "__main__":
    main()
