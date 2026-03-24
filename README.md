# LLM Dataset Generator

A robust and modular system for generating high-quality training datasets using Large Language Models. Supports local Ollama, with extensible architecture for additional providers.

## Features

- **Intelligent Prompt Generation**: Analyzes examples and constraints to generate optimized prompts
- **Multiple Format Support**: Built-in support for ShareGPT and Alpaca formats, easily extensible
- **Quality Control**: Automated evaluation and correction of generated entries
- **Batch Processing**: Efficient generation in configurable batch sizes
- **Progress Tracking**: Real-time progress display with ETA and statistics
- **Failure Logging**: Detailed tracking of failures for debugging and improvement
- **Modular Architecture**: Easy to extend with new providers, formats, and evaluators

## Requirements

- Python 3.8+
- Ollama (local or cloud)
- See `requirements.txt` for Python dependencies

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-dataset-generator

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running
ollama serve
```

## Quick Start

### Using the CLI

```bash
# Basic usage
python cli.py --goal "Train a helpful coding assistant" --entries 100

# With examples and constraints
python cli.py \
  --goal "Train a project planning assistant" \
  --examples examples/sample_dataset.json \
  --constraints examples/constraints.txt \
  --entries 50 \
  --batch-size 10

# Using a configuration file
python cli.py --config config.json
```

### Using as a Library

```python
from dataset_generator import DatasetGenerator, GenerationConfig

config = GenerationConfig(
    provider="ollama",
    model="gpt-oss-20B",
    total_entries=100,
    batch_size=10,
    output_format="sharegpt",
    dataset_goal="Train an assistant for technical documentation",
    example_dataset_path="examples/sample_dataset.json",
    constraints_path="examples/constraints.txt",
    output_dir="./output"
)

generator = DatasetGenerator(config)
dataset = generator.generate()
```

## Configuration

### GenerationConfig Parameters

| Parameter                 | Type | Default       | Description                                   |
| ------------------------- | ---- | ------------- | --------------------------------------------- |
| `provider`                | str  | "ollama"      | LLM provider to use                           |
| `model`                   | str  | "gpt-oss-20B" | Model name                                    |
| `use_thinking`            | bool | True          | Enable thinking/reasoning (if model supports) |
| `use_structured_outputs`  | bool | True          | Enable structured outputs (if model supports) |
| `total_entries`           | int  | 100           | Total entries to generate                     |
| `batch_size`              | int  | 10            | Entries per batch                             |
| `max_correction_attempts` | int  | 3             | Max attempts to correct failed entries        |
| `output_format`           | str  | "sharegpt"    | Output format (sharegpt/alpaca)               |
| `example_dataset_path`    | str  | None          | Path to example dataset                       |
| `constraints_path`        | str  | None          | Path to constraints file                      |
| `dataset_goal`            | str  | ""            | Goal/purpose of the dataset                   |
| `output_dir`              | str  | "./output"    | Output directory                              |
| `save_prompt`             | bool | True          | Save generation prompt                        |
| `save_failures`           | bool | True          | Save failure log                              |

## File Structure

```
llm-dataset-generator/
├── dataset_generator.py      # Main orchestrator
├── llm_providers.py          # LLM provider implementations
├── prompt_builder.py         # Prompt generation and analysis
├── dataset_evaluator.py      # Entry validation and quality control
├── progress_tracker.py       # Progress tracking and display
├── cli.py                    # Command-line interface
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── examples/                 # Example files
│   ├── sample_dataset.json
│   ├── constraints.txt
│   ├── alpaca_examples.json
│   └── alpaca_constraints.txt
└── output/                   # Generated datasets (created automatically)
```

## Supported Formats

### ShareGPT Format

```json
{
  "conversations": [
    { "from": "human", "value": "User message" },
    { "from": "gpt", "value": "Assistant response" }
  ]
}
```

### Alpaca Format

```json
{
  "instruction": "The task or question",
  "input": "Additional context (can be empty)",
  "output": "The response"
}
```

## Examples

See `example_usage.py` for comprehensive examples including:

- Basic ShareGPT generation
- Alpaca with constraints
- Custom formats
- High-quality datasets with aggressive QA
- Using Ollama Cloud
- Configuration from JSON

## How It Works

1. **Analysis Phase**: The system analyzes example data, constraints, and goals to generate insights
2. **Prompt Generation**: Creates an optimized prompt incorporating all inputs and insights
3. **Batch Generation**: Generates entries in batches using the LLM
4. **Evaluation**: Each entry is validated for structure and quality
5. **Correction**: Failed entries are sent back with feedback for correction
6. **Logging**: All failures are logged for later analysis

## Extending the System

### Adding a New Provider

```python
from llm_providers import LLMProvider

class MyProvider(LLMProvider):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def generate(self, prompt: str, context=None) -> str:
        # Implement your provider logic
        pass
```

### Adding a New Format

1. Update `PromptBuilder._get_format_specifications()`
2. Update `DatasetEvaluator._validate_structure()`
3. Add format-specific validation logic

### Custom Evaluator

```python
from dataset_evaluator import DatasetEvaluator

class MyEvaluator(DatasetEvaluator):
    def _validate_quality(self, entry):
        # Custom quality validation logic
        return is_valid, feedback
```

## Troubleshooting

### Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### Model Not Found

```bash
# Pull the model
ollama pull gpt-oss-20B
```

### Generation Quality Issues

- Review the failure log in `output/failures_*.json`
- Adjust constraints in your constraints file
- Provide better examples in your example dataset
- Increase `max_correction_attempts`
- Review and refine the `dataset_goal`

## Advanced Usage

### Disabling Model Features

If your model doesn't support thinking or structured outputs:

```python
config = GenerationConfig(
    model="mistral",
    use_thinking=False,
    use_structured_outputs=False,
    # ... other config
)
```

### Using Ollama Cloud

```python
from llm_providers import OllamaProvider

provider = OllamaProvider(
    model="gpt-oss-20B",
    base_url="https://cloud.ollama.ai"
)

# Use in generator
generator = DatasetGenerator(config)
generator.provider = provider
generator.prompt_builder.provider = provider
generator.evaluator.provider = provider
```

### Batch Size Optimization

- **Larger batches** (20-50): Faster generation, but more risk of quality issues
- **Smaller batches** (5-10): Slower but better quality control
- **Very small batches** (1-3): Maximum quality, slowest generation

## Performance Tips

1. Use structured outputs when available for better parsing
2. Start with smaller batches to test quality before scaling up
3. Provide clear, specific examples and constraints
4. Monitor the failure log to identify recurring issues
5. Use thinking mode for complex generation tasks

## License

[Your License Here]

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.
