"""
LLM Dataset Generator
A modular system for generating high-quality training datasets using LLMs
"""

import json
import logging
import time
import sys
import random
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import json_repair as _json_repair

    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

from .providers import (
    AnthropicProvider,
    GeminiProvider,
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from .prompt_builder import PromptBuilder
from .dataset_evaluator import DatasetEvaluator
from .progress_tracker import ProgressTracker


@dataclass
class GenerationConfig:
    """Configuration for dataset generation"""

    provider: str = "ollama"
    model: str = "gpt-oss-20B"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    use_thinking: bool = True
    use_structured_outputs: bool = True
    total_entries: int = 100
    batch_size: int = 10
    max_correction_attempts: int = 3
    output_format: str = "sharegpt"  # sharegpt or alpaca

    example_dataset_path: Optional[str] = None
    constraints_path: Optional[str] = None
    dataset_goal: str = ""

    # API key for cloud providers (OpenRouter, OpenAI, Anthropic, Gemini, etc.)
    api_key: Optional[str] = None

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
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _init_provider(self) -> LLMProvider:
        """Initialize LLM provider"""
        api_key = self.config.api_key

        if self.config.provider == "ollama":
            if not api_key and OllamaProvider.is_cloud_model(self.config.model):
                api_key = os.environ.get("OLLAMA_API_KEY")
            return OllamaProvider(
                model=self.config.model,
                base_url=self.config.base_url,
                api_key=api_key,
                temperature=self.config.temperature,
                use_thinking=self.config.use_thinking,
                use_structured_outputs=self.config.use_structured_outputs,
            )

        # Fallback to environment variables if api_key is missing in config
        if not api_key:
            if self.config.provider == "openrouter":
                api_key = os.environ.get("OPENROUTER_API_KEY")
            elif self.config.provider == "gemini":
                api_key = os.environ.get("GEMINI_API_KEY")

        if self.config.provider == "openrouter":
            if not api_key:
                raise ValueError(
                    "api_key is required for the openrouter provider. "
                    "Pass --api-key <key>, set it in your config file, or set OPENROUTER_API_KEY env var."
                )
            return OpenRouterProvider(
                model=self.config.model,
                api_key=api_key,
                temperature=self.config.temperature,
                use_structured_outputs=self.config.use_structured_outputs,
            )
        elif self.config.provider == "gemini":
            if not api_key:
                raise ValueError(
                    "api_key is required for the gemini provider. "
                    "Pass --api-key <key>, set it in your config file, or set GEMINI_API_KEY env var."
                )
            return GeminiProvider(
                model=self.config.model,
                api_key=api_key,
                temperature=self.config.temperature,
                use_structured_outputs=self.config.use_structured_outputs,
            )
        elif self.config.provider == "openai":
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required for the openai provider. "
                    "Pass --api-key <key>, set it in your config file, or set OPENAI_API_KEY env var."
                )
            return OpenAIProvider(
                model=self.config.model,
                api_key=api_key,
            )
        elif self.config.provider == "anthropic":
            if not api_key:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required for the anthropic provider. "
                    "Pass --api-key <key>, set it in your config file, or set ANTHROPIC_API_KEY env var."
                )
            return AnthropicProvider(
                model=self.config.model,
                api_key=api_key,
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
                # 3) Last-resort fallback
                self.logger.warning(
                    f"Could not decode file {path} as UTF-8. Falling back with replacement."
                )
                return p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            self.logger.warning(f"Could not load file {path}: {e}")
            return None

    def _load_analysis_inputs(self) -> tuple[Optional[str], Optional[str]]:
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
        return example_data, constraints

    def build_prompt(self) -> Path:
        """Build and save the generation prompt. Returns the path to the saved file."""
        self.logger.info("=" * 60)
        self.logger.info("BUILDING GENERATION PROMPT")
        self.logger.info("=" * 60)

        self.tracker.only_status = True
        try:
            example_data, constraints = self._load_analysis_inputs()

            analysis_chars = 0
            last_ui = time.monotonic()

            def on_analysis_chunk(s: str):
                nonlocal analysis_chars, last_ui
                analysis_chars += len(s)
                now = time.monotonic()
                if now - last_ui >= 1.0:
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
                status_cb=self.tracker.update_status,
                token_cb=on_analysis_chunk,
            )
            dt = time.perf_counter() - t0

            self.logger.info(
                f"✓ Prompt built in {dt:.1f}s ({len(self.generation_prompt)} chars)"
            )

            prompt_file = self.output_dir / "generation_prompt.txt"
            prompt_file.write_text(
                self.generation_prompt, encoding="utf-8", errors="replace"
            )
            self.logger.info(f"✓ Prompt saved to: {prompt_file}")

            return prompt_file
        finally:
            self.tracker.only_status = False
            print()  # Finalize status line

    def generate(self) -> List[Dict]:
        """Main generation workflow"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING DATASET GENERATION")
        self.logger.info("=" * 60)

        # Step 1 & 2: Analyze inputs and generate prompt
        self.logger.info("\n[STEP 1-2] Analyzing inputs and generating prompt...")

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
            
            # Strip trailing trigger if it exists to avoid confusing the model
            # when we append our batch request in the user prompt.
            import re
            original_len = len(self.generation_prompt)
            self.generation_prompt = re.sub(
                r"Generate (?:the )?\d+ entries now:?\s*$", 
                "", 
                self.generation_prompt, 
                flags=re.IGNORECASE | re.MULTILINE
            ).strip()
            
            if len(self.generation_prompt) < original_len:
                self.logger.info("Stripped trailing 'Generate now' trigger from loaded prompt.")

            # Warn about huge prompts
            if len(self.generation_prompt) > 40000:
                self.logger.warning(
                    f"Prompt size ({len(self.generation_prompt)} chars) is very large "
                    "and may exceed the LLM context window. If generation fails, "
                    "consider using a shorter prompt or re-running build-prompt."
                )

        else:
            example_data, constraints = self._load_analysis_inputs()

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
                status_cb=self.tracker.update_status,
                token_cb=on_analysis_chunk,
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        incremental_file = self.output_dir / f"dataset_incremental_{timestamp}.jsonl"
        self.logger.info(f"Incremental save file: {incremental_file}")

        current_batch_size = self.config.batch_size
        batch_num = 0
        consecutive_failures = 0
        max_consecutive_failures = 5

        while len(self.dataset) < self.config.total_entries:
            entries_needed = min(
                current_batch_size, self.config.total_entries - len(self.dataset)
            )

            if entries_needed <= 0:
                break

            batch_num += 1
            self.logger.info(
                f"\n[STEP 3] Generating batch {batch_num} (size: {current_batch_size})"
            )
            self.tracker.update_status(
                f"Generating batch {batch_num} (Progress: {len(self.dataset)}/{self.config.total_entries})"
            )

            batch_entries = self._generate_batch(entries_needed)

            if not batch_entries:
                consecutive_failures += 1
                self.logger.warning(
                    f"Batch {batch_num} failed to generate any valid entries. "
                    f"Consecutive failures: {consecutive_failures}/{max_consecutive_failures}"
                )
                if consecutive_failures >= max_consecutive_failures:
                    self.logger.error(
                        "Too many consecutive generation failures. Aborting to prevent infinite loop."
                    )
                    break
                # Reduce batch size on failure
                current_batch_size = max(1, current_batch_size // 2)
                continue

            # Reset failures on success
            consecutive_failures = 0

            # Step 4: Evaluate and correct entries
            self.logger.info(f"[STEP 4] Evaluating batch {batch_num}")
            self.tracker.update_status(f"Evaluating batch {batch_num}")

            validated_entries = self._evaluate_and_correct_batch(batch_entries)

            # Add validated_entries to dataset even if 0 (though unlikely if batch_entries was non-empty)
            if not validated_entries:
                self.logger.warning(f"Batch {batch_num} - all entries failed validation.")
            
            self.dataset.extend(validated_entries)
            self.tracker.add_entries(len(validated_entries))

            # Incremental save
            if validated_entries:
                with open(incremental_file, "a", encoding="utf-8") as f:
                    for entry in validated_entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Adaptive batching: increase if successful, reduce if struggling
            if len(validated_entries) == entries_needed:
                # Success! Maybe increase batch size slowly back up
                if current_batch_size < self.config.batch_size:
                    current_batch_size = min(self.config.batch_size, current_batch_size + 1)
            elif len(validated_entries) < entries_needed // 2:
                self.logger.warning(
                    f"Batch {batch_num} returned only {len(validated_entries)} valid entries "
                    f"(requested {entries_needed}). Reducing batch size."
                )
                current_batch_size = max(1, current_batch_size // 2)

            self.logger.info(
                f"✓ Batch {batch_num} complete: {len(validated_entries)}/{entries_needed} entries added"
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

            diversity_hints = [
                "Focus on edge cases and rare scenarios.",
                "Use a different tone or persona than previous entries.",
                "Ensure the examples are complex and multi-faceted.",
                "Keep the interaction brief and highly direct.",
                "Incorporate unexpected but valid user inputs.",
            ]
            hint = random.choice(diversity_hints)

            # Include a few recent entries to help the LLM avoid repetition
            recent_context = ""
            if self.dataset:
                # Take up to 5 recent entries
                recent = self.dataset[-5:]
                recent_context = "\n## RECENTLY GENERATED (DO NOT REPEAT):\n"
                for i, entry in enumerate(recent):
                    # Create a very brief summary of the entry
                    summary = ""
                    if self.config.output_format == "sharegpt":
                        convs = entry.get("conversations", [])
                        if convs:
                            summary = convs[0].get("value", "")[:100]
                    elif self.config.output_format == "alpaca":
                        summary = entry.get("instruction", "")[:100]
                    
                    recent_context += f"- {summary}...\n"

            batch_prompt = (
                f"## BATCH REQUEST\n"
                f"Generate EXACTLY {num_entries} NEW and UNIQUE dataset entries now.\n"
                f"FORMAT: {self.config.output_format.upper()}\n"
                f"DIVERSITY HINT: {hint}\n"
                f"{recent_context}\n"
                f"CRITICAL: You must return a JSON object with an \"entries\" key. \n"
                f"The \"entries\" key must contain a list of EXACTLY {num_entries} objects. \n"
                f"Each object must follow the {self.config.output_format} schema.\n"
                f"Ensure every entry is different from previous ones and contributes unique value."
            )
            prompt_preview = batch_prompt[:100].replace("\n", "\\n")
            self.logger.info(
                f"Sending generation batch request to LLM ({len(batch_prompt)} chars) preview: {prompt_preview}"
            )
            response = self.provider.generate(
                prompt=batch_prompt, 
                system_prompt=self.generation_prompt,
                structured_outputs=True
            )
            self.logger.info("âś“ Generation response received from LLM")
            self.logger.info("Generation response raw:\n%s", response)
            entries = self._parse_generation_response(response, expected_count=num_entries)

            # Retry with correction prompt if parsing produced 0 entries
            for attempt in range(self.config.max_correction_attempts):
                if entries:
                    break
                self.logger.warning(
                    "Parse returned 0 entries on attempt %d/%d — sending correction prompt",
                    attempt + 1,
                    self.config.max_correction_attempts,
                )
                fix_prompt = (
                    "### ERROR: INVALID RESPONSE FORMAT\n"
                    "Your previous response did not follow the required structure or contained the wrong number of entries.\n"
                    "Here is what you returned (truncated to 1500 chars):\n\n"
                    "--- BEGIN PREVIOUS RESPONSE ---\n"
                    + response[:1500]
                    + "\n--- END PREVIOUS RESPONSE ---\n\n"
                    f"Please try again. You MUST generate EXACTLY {num_entries} NEW entries.\n"
                    "CRITICAL REQUIREMENTS:\n"
                    "1. Return a JSON object with an \"entries\" key.\n"
                    f"2. The \"entries\" key must contain an array of EXACTLY {num_entries} objects.\n"
                    "3. No introductory prose, no code fences, no commentary.\n"
                    f"4. Ensure every entry follows the {self.config.output_format} schema precisely."
                )
                response = self.provider.generate(
                    prompt=fix_prompt, 
                    system_prompt=self.generation_prompt,
                    structured_outputs=True
                )
                self.logger.info(
                    "Correction attempt %d response raw:\n%s", attempt + 1, response
                )
                entries = self._parse_generation_response(response)

            self.logger.info(f"Generated {len(entries)} entries")
            self.logger.info(
                "Generation response parsed:\n%s", json.dumps(entries, indent=2)
            )
            return entries

        except Exception as e:
            self.logger.error(f"Error generating batch: {e}")
            return []

    # Keys that signal a metadata/wrapper object rather than a dataset entry
    _WRAPPER_KEYS = frozenset({"metadata", "result", "response", "summary"})

    def _try_parse_json(self, text: str) -> Optional[List[Dict]]:
        """Attempt to parse text as a JSON array, JSONL, or single object.

        Returns a non-empty list of dicts on success, or None on failure.
        """
        text = text.strip()
        if not text:
            return None

        # JSON array
        if text.startswith("["):
            try:
                result = json.loads(text)
                if isinstance(result, list):
                    entries = [e for e in result if isinstance(e, dict)]
                    if entries:
                        return entries
            except json.JSONDecodeError:
                pass

        # Single JSON object
        if text.startswith("{"):
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    # Check if this is a wrapper with an obvious entries key
                    for key in ["entries", "data", "dataset", "examples"]:
                        if key in obj and isinstance(obj[key], list):
                            self.logger.debug(f"Extracting entries from '{key}' wrapper")
                            entries = [e for e in obj[key] if isinstance(e, dict)]
                            if entries:
                                return entries

                    # Skip pure metadata wrappers
                    if len(obj) == 1 and next(iter(obj)) in self._WRAPPER_KEYS:
                        return None
                    return [obj]
            except json.JSONDecodeError:
                pass

        # JSONL — one object per non-empty line
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) > 1:
            entries: List[Dict] = []
            for line in lines:
                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        continue
                    if len(obj) == 1 and next(iter(obj)) in self._WRAPPER_KEYS:
                        self.logger.debug(
                            "Skipping wrapper object with key: %s", next(iter(obj))
                        )
                        continue
                    entries.append(obj)
                except json.JSONDecodeError:
                    continue
            if entries:
                return entries

        # Last resort: attempt automatic JSON repair
        if _HAS_JSON_REPAIR:
            try:
                repaired = _json_repair.loads(text)
                if isinstance(repaired, list):
                    entries = [e for e in repaired if isinstance(e, dict)]
                    if entries:
                        self.logger.debug("Recovered entries via json_repair (array)")
                        return entries
                elif isinstance(repaired, dict):
                    if not (
                        len(repaired) == 1
                        and next(iter(repaired)) in self._WRAPPER_KEYS
                    ):
                        self.logger.debug("Recovered entry via json_repair (object)")
                        return [repaired]
            except Exception:
                pass

        return None

    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract the content inside every markdown code fence in text.

        Returns blocks in the order they appear, last (most-corrected) first
        so callers can try the most-complete version first.
        """
        import re

        # Match ```<optional lang>\n<content>\n```
        pattern = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
        blocks = [m.group(1).strip() for m in pattern.finditer(text)]
        # Reverse so the last (most complete/corrected) block is tried first
        return list(reversed(blocks))

    def _extract_json_patterns(self, text: str) -> List[str]:
        """Extract everything that looks like a JSON array or object using basic balancing."""
        results = []
        
        # Look for [ ... ]
        start = -1
        balance = 0
        for i, char in enumerate(text):
            if char == "[":
                if balance == 0:
                    start = i
                balance += 1
            elif char == "]":
                balance -= 1
                if balance == 0 and start != -1:
                    results.append(text[start:i+1])
                    start = -1
                    
        # Look for { ... }
        start = -1
        balance = 0
        for i, char in enumerate(text):
            if char == "{":
                if balance == 0:
                    start = i
                balance += 1
            elif char == "}":
                balance -= 1
                if balance == 0 and start != -1:
                    results.append(text[start:i+1])
                    start = -1
                    
        return results

    def _parse_generation_response(self, response: str, expected_count: int = 0) -> List[Dict]:
        """Parse LLM response to extract dataset entries."""
        text = response.strip()

        entries = []
        # --- Pass 1: try the full text as-is ---
        result = self._try_parse_json(text)
        if result:
            entries = result

        # --- Pass 2: if the whole text starts with a code fence, strip it ---
        if not entries and text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                inner = text[first_newline + 1 :]
                if inner.rstrip().endswith("```"):
                    inner = inner.rstrip()[:-3].rstrip()
                result = self._try_parse_json(inner)
                if result:
                    entries = result

        # --- Pass 3: scan for ALL embedded code fences anywhere in the text ---
        if not entries:
            for block in self._extract_code_blocks(text):
                result = self._try_parse_json(block)
                if result:
                    self.logger.debug("Recovered entries from embedded code block")
                    entries = result
                    break

        # --- Pass 4: scan for ANYTHING that looks like JSON anywhere in the text ---
        if not entries:
            for pattern in self._extract_json_patterns(text):
                result = self._try_parse_json(pattern)
                if result:
                    self.logger.debug("Recovered entries from raw text pattern")
                    entries = result
                    break

        # RESILIENCE: Auto-fix common typos and format mismatches
        for entry in entries:
            if not isinstance(entry, dict): continue
            
            if self.config.output_format == "sharegpt":
                # Fix 'conversation' -> 'conversations'
                if "conversation" in entry and "conversations" not in entry:
                    self.logger.debug("Auto-fixing 'conversation' -> 'conversations' typo")
                    entry["conversations"] = entry.pop("conversation")
                
                # Fix Alpaca-style entry being sent to ShareGPT
                if "conversations" not in entry and "instruction" in entry and "output" in entry:
                    self.logger.debug("Auto-converting Alpaca entry to ShareGPT")
                    entry["conversations"] = [
                        {"from": "human", "value": f"{entry['instruction']}\n\n{entry.get('input', '')}".strip()},
                        {"from": "gpt", "value": entry["output"]}
                    ]
                
                # Fix message field names (role/content -> from/value)
                if "conversations" in entry and isinstance(entry["conversations"], list):
                    for msg in entry["conversations"]:
                        if not isinstance(msg, dict): continue
                        if "from" not in msg and "role" in msg:
                            msg["from"] = "human" if msg["role"] in ["user", "human"] else "gpt"
                        if "value" not in msg and "content" in msg:
                            msg["value"] = msg.pop("content")
            
            elif self.config.output_format == "alpaca":
                # Fix ShareGPT-style entry being sent to Alpaca
                if "instruction" not in entry and "conversations" in entry:
                    self.logger.debug("Auto-converting ShareGPT entry to Alpaca")
                    convs = entry["conversations"]
                    if len(convs) >= 2:
                        entry["instruction"] = convs[0].get("value", "")
                        entry["input"] = ""
                        entry["output"] = convs[1].get("value", "")

        # Check if we got the expected count (if specified)
        if expected_count > 0 and len(entries) < expected_count:
            self.logger.warning(
                f"Parser extracted {len(entries)} entries, but expected {expected_count}. "
                "Returning empty list to trigger correction."
            )
            return []

        if not entries:
            self.logger.error(
                "Error parsing generation response: could not extract any JSON entries"
            )
        
        return entries

    def _evaluate_and_correct_batch(self, entries: List[Dict]) -> List[Dict]:
        """Evaluate entries and correct failures"""
        validated = []
        eval_results = self.evaluator.evaluate_batch(entries)

        for idx, (entry, (is_valid, feedback)) in enumerate(zip(entries, eval_results)):
            self.logger.info(f"  Evaluating entry {idx + 1}/{len(entries)}")

            if is_valid:
                validated.append(entry)
                self.logger.info(f"  ✓ Entry {idx + 1} passed")
            else:
                self.logger.warning(f"  ✗ Entry {idx + 1} failed: {feedback}")

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
        constraints = self._load_file(self.config.constraints_path)
        correction_prompt = self.prompt_builder.build_correction_prompt(
            original_entry=entry,
            feedback=feedback,
            output_format=self.config.output_format,
            goal=self.config.dataset_goal,
            constraints=constraints,
        )
        self.logger.info("Correction prompt sent to LLM:\n%s", correction_prompt)

        for attempt in range(self.config.max_correction_attempts):
            try:
                self.tracker.update_status(f"Correcting entry (attempt {attempt + 1})")

                # Pass generation_prompt as system prompt to maintain context
                response = self.provider.generate(
                    prompt=correction_prompt, 
                    system_prompt=self.generation_prompt,
                    structured_outputs=True
                )
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
        """Save the generated dataset."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.config.output_format == "jsonl":
            filename = self.output_dir / f"dataset_{timestamp}.jsonl"
            with open(filename, "w", encoding="utf-8") as f:
                for entry in self.dataset:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        else:
            filename = (
                self.output_dir
                / f"dataset_{self.config.output_format}_{timestamp}.json"
            )
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.dataset, f, indent=2, ensure_ascii=False)

        self.logger.info(f"✓ Dataset saved to: {filename}")

    def _save_failures(self):
        """Save failure log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"failures_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(self.failures, f, indent=2)

        self.logger.info(f"✓ Failure log saved to: {filename}")
