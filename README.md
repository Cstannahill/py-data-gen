# py-data-gen

A modular, production-ready pipeline for generating high-quality LLM training datasets. It uses a two-phase workflow — intelligent prompt synthesis followed by batched generation — with built-in evaluation, automatic correction, and resilient JSON parsing. Local models via Ollama, cloud models via OpenRouter and Gemini, and a clean provider interface designed for easy extension.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
  - [CLI Usage](#cli-usage)
  - [Library Usage](#library-usage)
- [Output Formats](#output-formats)
- [Providers](#providers)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Adding a New Provider](#adding-a-new-provider)
- [Tests](#tests)

---

## Features

| Feature                          | Description                                                                                                                                     |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Two-Phase Prompt Synthesis**   | An LLM first analyses your examples and constraints to generate a tailored generation system prompt, which then drives all subsequent batches   |
| **Prompt Reuse**                 | Save and reload a built prompt with `existing_prompt_path` to skip re-analysis and iterate faster                                               |
| **Adaptive Batch Sizing**        | Automatically reduces batch size on repeated failures and ramps back up on success to maintain forward progress                                 |
| **Multi-Pass JSON Parser**       | Four-pass parser handles raw JSON, fenced code blocks (any language tag), embedded JSON anywhere in prose, and malformed JSON via `json-repair` |
| **Parse-Failure Retries**        | When a batch yields zero parseable entries, a structured correction prompt is sent back to the LLM (up to `max_correction_attempts`)            |
| **Batch Quality Evaluation**     | Structural validation + heuristic checks + LLM quality review, batched into a single prompt to reduce API calls                                 |
| **Entry Correction**             | Entries that fail evaluation are individually corrected with targeted feedback before being discarded                                           |
| **Incremental Saves**            | Validated entries are appended to a JSONL file after every batch — no data loss on crash                                                        |
| **Diversity Injection**          | Random hints and recent-entry context sent with each batch prompt to reduce repetition                                                          |
| **Multiple Output Formats**      | ShareGPT, Alpaca, and generic JSONL with format-aware schema validation                                                                         |
| **Multi-Provider Support**       | Ollama (local + cloud), OpenRouter, and Gemini fully implemented; OpenAI and Anthropic stubs ready                                              |
| **Streaming Responses**          | All providers stream token-by-token with an optional `on_chunk` callback                                                                        |
| **Automatic Retries**            | Cloud providers retry on 429/5xx with exponential back-off                                                                                      |
| **Real-Time Progress Bar**       | In-terminal progress bar with entry count, percentage, elapsed time, and ETA                                                                    |
| **Structured Logging**           | Per-run timestamped log file alongside console output                                                                                           |
| **Failure Log**                  | All rejected entries and their feedback saved to a JSON file for offline review                                                                 |
| **Environment Variable Support** | API keys resolved from `.env`, config file, or `--api-key` flag — in that priority order                                                        |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                        CLI / Library                    │
└───────────────────────────┬─────────────────────────────┘
                            │  GenerationConfig
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   DatasetGenerator                      │
│                                                         │
│  Phase 1 — Prompt Build                                 │
│    PromptBuilder → analysis LLM call → final prompt     │
│                                                         │
│  Phase 2 — Batch Loop                                   │
│    for each batch:                                      │
│      generate → parse (4-pass) → retry if 0 entries    │
│      evaluate batch (structure + heuristic + LLM QA)   │
│      correct failed entries → incremental JSONL save    │
│      adaptive batch size adjustment                     │
│                                                         │
│  Phase 3 — Finalise                                     │
│    write dataset file  ·  write failure log             │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
    ┌──────▼──────┐           ┌───────▼──────┐
    │ LLMProvider │           │DatasetEvaluator│
    │  (abstract) │           └───────────────┘
    └──────┬──────┘
           │
   ┌───────┼────────────────────┐
   ▼       ▼                    ▼
Ollama  OpenRouter           Gemini
(local/cloud)  (hundreds    (Google AI
               of models)    Studio)
```

---

## Requirements

- Python 3.10+
- One of:
  - [Ollama](https://ollama.com) running locally (default `http://localhost:11434`)
  - An [OpenRouter](https://openrouter.ai) API key
  - A [Google AI Studio](https://aistudio.google.com) (Gemini) API key

```
requests>=2.31.0
python-dotenv>=1.0.0
json-repair>=0.30.0
```

---

## Installation

```bash
# Clone
git clone https://github.com/Cstannahill/py-data-gen
cd py-data-gen

# Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```dotenv
# Required for OpenRouter
OPENROUTER_API_KEY=sk-or-...

# Required for Gemini
GEMINI_API_KEY=AIza...

# Required for Ollama cloud models (models ending in -cloud)
OLLAMA_API_KEY=...

# Optional — OpenAI / Anthropic (providers not yet implemented)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

The CLI resolves keys automatically from `.env` on startup. You can also pass `--api-key` directly or set `api_key` in `config.json`.

---

## Configuration

### config.json

All settings can be declared in a JSON file and passed with `--config`. CLI flags override individual fields when both are provided.

```jsonc
{
  // Provider: "ollama" | "openrouter" | "gemini" | "openai" | "anthropic"
  "provider": "ollama",

  // Model name. Append "-cloud" suffix for Ollama cloud; use provider's model ID for others.
  "model": "gpt-oss-20B",

  // Enable thinking/reasoning tokens where the model supports it (Ollama only)
  "use_thinking": true,

  // Request structured JSON output from the provider
  "use_structured_outputs": true,

  // Total dataset entries to generate
  "total_entries": 1000,

  // Entries per generation batch
  "batch_size": 10,

  // Max correction attempts when a batch returns 0 parseable entries
  "max_correction_attempts": 3,

  // Output schema: "sharegpt" | "alpaca" | "jsonl"
  "output_format": "sharegpt",

  // Path to few-shot example file (.jsonl or .json)
  "example_dataset_path": "examples/alpaca_examples.jsonl",

  // Path to plain-text constraints / style guide
  "constraints_path": "constraints/constraints.txt",

  // Skip prompt-build phase and reuse an existing prompt file
  "existing_prompt_path": null,

  // Natural-language description of what the dataset should achieve
  "dataset_goal": "Train a coding assistant that explains concepts clearly.",

  // Directory for all output files
  "output_dir": "./output",

  // Persist the built generation prompt to generation_prompt.txt
  "save_prompt": true,

  // Persist rejected entries to failures_<timestamp>.json
  "save_failures": true,
}
```

### GenerationConfig (Python)

```python
from app.dataset_generator import DatasetGenerator, GenerationConfig

config = GenerationConfig(
    provider="openrouter",
    model="mistralai/mistral-7b-instruct",
    api_key="sk-or-...",          # or set OPENROUTER_API_KEY env var
    total_entries=500,
    batch_size=10,
    max_correction_attempts=3,
    output_format="sharegpt",     # "sharegpt" | "alpaca" | "jsonl"
    dataset_goal="Train a customer-support assistant for an e-commerce platform.",
    example_dataset_path="examples/sample_dataset.jsonl",
    constraints_path="constraints/constraints.txt",
    output_dir="./output/ecommerce",
    save_prompt=True,
    save_failures=True,
)
```

---

## Quick Start

### CLI Usage

```bash
# ── Local Ollama ────────────────────────────────────────────────
python cli.py generate --config config.json

# ── OpenRouter (cloud) — key loaded from .env ───────────────────
python cli.py generate \
    --provider openrouter \
    --model mistralai/mistral-7b-instruct \
    --config config.json

# ── Gemini ──────────────────────────────────────────────────────
python cli.py generate \
    --provider gemini \
    --model gemini-1.5-flash \
    --config config.json

# ── Build prompt only (no generation) ───────────────────────────
python cli.py build-prompt \
    --goal "Train a coding assistant" \
    --examples examples/sample_dataset.jsonl \
    --constraints constraints/constraints.txt \
    --output-dir ./output/my_run

# ── Generate from a previously built prompt (skip re-analysis) ──
python cli.py generate \
    --config config.json \
    --provider ollama \
    --model my-model
# (set existing_prompt_path in config.json to reuse a saved prompt)
```

#### Full CLI Reference

**Common flags (both subcommands)**

| Flag                      | Default                  | Description                                                 |
| ------------------------- | ------------------------ | ----------------------------------------------------------- |
| `--provider`              | `ollama`                 | `ollama` · `openrouter` · `gemini` · `openai` · `anthropic` |
| `--model`                 | `gpt-oss-20B`            | Provider-specific model identifier                          |
| `--api-key`               | _(env)_                  | API key; falls back to environment variable                 |
| `--base-url`              | `http://localhost:11434` | Ollama server URL                                           |
| `--temperature`           | `0.7`                    | Sampling temperature                                        |
| `--no-thinking`           | —                        | Disable thinking/reasoning tokens                           |
| `--no-structured-outputs` | —                        | Disable JSON-mode structured outputs                        |
| `--output-dir`            | `./output`               | Directory for all output files                              |
| `--config`                | —                        | Path to a JSON config file                                  |

**`generate` subcommand**

| Flag                 | Default    | Description                                      |
| -------------------- | ---------- | ------------------------------------------------ |
| `--entries`          | `100`      | Total entries to generate                        |
| `--batch-size`       | `10`       | Entries per batch                                |
| `--format`           | `sharegpt` | `sharegpt` · `alpaca` · `jsonl`                  |
| `--max-corrections`  | `3`        | Max correction attempts per failed batch         |
| `--prompt`           | —          | Path to existing prompt file (skips re-analysis) |
| `--no-save-failures` | —          | Suppress failure log output                      |

**`build-prompt` subcommand**

| Flag            | Default    | Description                                      |
| --------------- | ---------- | ------------------------------------------------ |
| `--goal`        | —          | Dataset objective (natural language)             |
| `--examples`    | —          | Path to example dataset file                     |
| `--constraints` | —          | Path to constraints/style guide file             |
| `--format`      | `sharegpt` | Output format (shapes the prompt)                |
| `--batch-size`  | `10`       | Entries per batch (used to calibrate the prompt) |

### Library Usage

```python
from app.dataset_generator import DatasetGenerator, GenerationConfig

config = GenerationConfig(
    provider="gemini",
    model="gemini-1.5-flash",
    api_key="AIza...",
    total_entries=200,
    batch_size=10,
    output_format="alpaca",
    dataset_goal="Generate instruction-following examples for a medical Q&A assistant.",
    example_dataset_path="examples/alpaca_examples.jsonl",
    constraints_path="constraints/constraints.txt",
    output_dir="./output/medical",
)

generator = DatasetGenerator(config)

# Build prompt only (returns Path to saved prompt file)
prompt_path = generator.build_prompt()

# Full generation (build prompt + generate + evaluate + save)
dataset = generator.generate()
print(f"Generated {len(dataset)} entries")
```

---

## Output Formats

### ShareGPT

Multi-turn conversation format used by most fine-tuning frameworks (Axolotl, LLaMA-Factory, etc.).

```json
{
  "conversations": [
    { "from": "human", "value": "What is gradient descent?" },
    {
      "from": "gpt",
      "value": "Gradient descent is an optimisation algorithm..."
    }
  ]
}
```

**Validation rules:** `conversations` key required; messages must alternate `human → gpt`; each message must have non-empty `from` and `value` fields.

### Alpaca

Instruction-following format popularised by Stanford Alpaca.

```json
{
  "instruction": "Explain the difference between TCP and UDP.",
  "input": "",
  "output": "TCP provides reliable, ordered delivery with connection overhead. UDP is connectionless and faster but offers no delivery guarantees..."
}
```

**Validation rules:** All three fields (`instruction`, `input`, `output`) required as strings; `instruction` and `output` must be non-empty.

### JSONL (generic)

Raw JSONL output — each line is a complete JSON object. No structural schema enforced; use this for custom formats or when the LLM defines the schema from your constraints.

---

## Providers

| Provider       | Status  | Key env var                     | Notes                                                                                                                                                                           |
| -------------- | ------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ollama**     | ✅ Full | `OLLAMA_API_KEY` _(cloud only)_ | Local (`localhost:11434`) and cloud (`ollama.com`). Append `-cloud` to model name for cloud routing. Supports streaming, structured outputs, thinking tokens, and 128k context. |
| **OpenRouter** | ✅ Full | `OPENROUTER_API_KEY`            | Access to hundreds of upstream models via OpenAI-compatible API. Streaming with exponential-back-off retries on 429/5xx.                                                        |
| **Gemini**     | ✅ Full | `GEMINI_API_KEY`                | Google AI Studio. Supports streaming via `streamGenerateContent` and JSON-mode structured outputs. Exponential-back-off on 429/503.                                             |
| **OpenAI**     | 🚧 Stub | `OPENAI_API_KEY`                | Interface defined; implementation pending.                                                                                                                                      |
| **Anthropic**  | 🚧 Stub | `ANTHROPIC_API_KEY`             | Interface defined; implementation pending.                                                                                                                                      |

All providers share a common `LLMProvider` abstract base class with `generate()` and `test_connection()` methods, making it straightforward to add new backends.

---

## How It Works

### Phase 1 — Prompt Build

1. Load the example dataset and constraints files.
2. Send an **analysis prompt** to the LLM asking it to identify patterns, quality indicators, diversity strategies, and structural requirements.
3. Incorporate the analysis insights along with the examples, constraints, format spec, and dataset goal into a **final generation system prompt**.
4. Optionally save the prompt to `generation_prompt.txt` for reuse.

> If `existing_prompt_path` is set, this phase is skipped entirely.

### Phase 2 — Batch Generation Loop

For each batch until `total_entries` is reached:

1. **Generate** — Send the saved system prompt plus a batch request (with a random diversity hint and a summary of recent entries) to the LLM.
2. **Parse** — Four-pass parser attempts to extract valid JSON entries:
   - Pass 1: full response as-is
   - Pass 2: strip markdown code fences
   - Pass 3: scan for embedded JSON arrays/objects using bracket balancing
   - Pass 4: `json-repair` as last resort
3. **Retry on zero entries** — If parsing yields nothing, a structured correction prompt is sent (up to `max_correction_attempts`).
4. **Evaluate batch** — Each entry goes through:
   - **Structural check** — format-specific field/type validation
   - **Heuristic check** — minimum length guard
   - **LLM quality review** — batched into a single API call; assesses coherence, completeness, quality, naturalness, and safety
5. **Correct failures** — Each invalid entry gets a targeted correction prompt; passing entries are kept.
6. **Incremental save** — Validated entries appended to `dataset_incremental_<timestamp>.jsonl`.
7. **Adaptive sizing** — If < 50% of requested entries are valid, batch size is halved. After 5 consecutive total failures, generation aborts.

### Phase 3 — Finalise

- Write the full dataset to `dataset_sharegpt_<timestamp>.json` (or equivalent).
- Write `failures_<timestamp>.json` with all rejected entries and their feedback if `save_failures` is enabled.
- Write a timestamped `.log` file capturing all events.

---

## Project Structure

```
py-data-gen/
├── cli.py                        # Argument parsing and CLI entry point
├── config.json                   # Default config (edit to suit your run)
├── requirements.txt
│
├── app/
│   ├── __init__.py
│   ├── dataset_generator.py      # DatasetGenerator · GenerationConfig · JSON parsers
│   ├── dataset_evaluator.py      # DatasetEvaluator — structure + heuristic + LLM QA
│   ├── prompt_builder.py         # PromptBuilder — analysis → final generation prompt
│   ├── progress_tracker.py       # In-terminal progress bar with ETA
│   └── providers/
│       ├── __init__.py
│       ├── base.py               # LLMProvider abstract base class
│       ├── ollama.py             # Ollama (local + cloud)
│       ├── openrouter.py         # OpenRouter (streaming, retries)
│       ├── gemini.py             # Google Gemini (streaming, retries)
│       ├── openai.py             # OpenAI stub (not implemented)
│       └── anthropic.py         # Anthropic stub (not implemented)
│
├── constraints/                  # Plain-text constraint / style-guide files
│   ├── constraints.txt           # Example constraints
│   ├── alpaca_constraints.txt
│   └── prompt_architect.txt
│
├── examples/                     # Few-shot example datasets
│   ├── sample_dataset.jsonl
│   ├── alpaca_examples.jsonl
│   └── alpaca_examples_full.jsonl
│
├── output/                       # Generated datasets, prompts, and logs (git-ignored)
│
└── tests/
    ├── test_dataset_generator_correction.py
    ├── test_dataset_generator_loading.py
    ├── test_prompt_builder.py
    ├── test_task3_adaptive_batching.py
    ├── test_task5_batch_evaluation.py
    └── test_task6_system_prompts.py
```

---

## Adding a New Provider

1. Create `app/providers/myprovider.py` and subclass `LLMProvider`:

```python
from .base import LLMProvider
from typing import Callable, Dict, Optional

class MyProvider(LLMProvider):
    def __init__(self, model: str, api_key: str, temperature: float = 0.7):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        *,
        structured_outputs: Optional[bool] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        # Call your API here; stream if on_chunk is provided
        ...

    def test_connection(self) -> bool:
        # Return True if the API is reachable
        ...
```

2. Export from `app/providers/__init__.py`:

```python
from .myprovider import MyProvider
__all__ = [..., "MyProvider"]
```

3. Register in `DatasetGenerator._init_provider()` (`app/dataset_generator.py`) with the provider name string and any required env-var resolution.

4. Add `"myprovider"` to the `choices` list in `cli.py` → `_add_provider_args()`.

---

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_task3_adaptive_batching.py -v
```

Test coverage includes adaptive batching, batch evaluation, JSON parsing and correction, prompt building, system prompt handling, and file loading.
