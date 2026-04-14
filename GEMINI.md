# Project Overview: py-data-gen

A robust and modular system for generating high-quality training datasets using Large Language Models (LLMs). It supports local Ollama and OpenRouter, with an extensible architecture for additional providers.

The project uses a two-step prompt generation process:
1.  **Analysis:** The LLM analyzes goals, examples, and constraints to generate insights.
2.  **Generation:** These insights are incorporated into a final prompt for generating high-quality, diverse dataset entries in batches.

It also features an automated evaluation and correction loop to ensure structural and quality compliance.

## Building and Running

### Requirements
- Python 3.8+
- Ollama (local or cloud)
- Dependencies: `pip install -r requirements.txt` (only `requests` and `python-dotenv`)

### Configuration
Configuration can be passed via command-line flags or a JSON configuration file (e.g., `config.json`). Environment variables for API keys should be stored in a `.env` file.

### Commands
- **Build Generation Prompt:**
  ```bash
  python cli.py build-prompt --goal "Your Goal" --examples path/to/examples.jsonl --constraints path/to/constraints.txt --output-dir ./output/my_run
  ```
- **Generate Dataset:**
  ```bash
  python cli.py generate --prompt ./output/my_run/generation_prompt.txt --entries 100 --batch-size 10
  ```
- **Using a Config File:**
  ```bash
  python cli.py generate --config config.json
  ```

## Development Conventions

### Architecture & File Structure
- **Core Package:** The active codebase is located in the `app/` directory.
- **Legacy Files:** Files in the root directory (e.g., `dataset_generator.py`, `llm_providers.py`) are legacy versions and should be ignored in favor of their counterparts in `app/`.
- **Modularity:**
    - `app/dataset_generator.py`: Main orchestrator (`DatasetGenerator` class).
    - `app/providers/`: LLM provider implementations (Ollama, OpenRouter, Anthropic, OpenAI).
    - `app/prompt_builder.py`: Logic for creating analysis, generation, and correction prompts.
    - `app/dataset_evaluator.py`: Logic for structural validation and LLM-based quality evaluation.
    - `app/progress_tracker.py`: Real-time progress display.

### Key Practices
- **JSON-based Communication:** Most internal data and outputs are in JSON or JSONL format.
- **Provider Interface:** New LLM providers should inherit from `LLMProvider` in `app/providers/base.py`.
- **Format Support:** Supports `sharegpt` and `alpaca` formats out of the box. Adding new formats requires updating `PromptBuilder` and `DatasetEvaluator`.
- **Quality Control:** The system uses an LLM-based evaluation step (`DatasetEvaluator._validate_quality`) to check for coherence, completeness, and naturalness.

## Important Directories
- `app/`: Source code for the main application.
- `constraints/`: Directory for text files defining dataset constraints.
- `examples/`: Directory for example dataset entries (JSON/JSONL).
- `output/`: Default directory for generated prompts, datasets, and logs.
