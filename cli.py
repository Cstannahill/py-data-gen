#!/usr/bin/env python3
"""
Command-line interface for the dataset generator
"""

import argparse
import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads .env from the current working directory (or any parent)

from app.dataset_generator import DatasetGenerator, GenerationConfig


def load_json_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)


def _add_provider_args(parser: argparse.ArgumentParser):
    """Shared provider/model arguments used by both subcommands."""
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "openrouter", "gemini", "openai", "anthropic"],
        help="LLM provider to use (default: ollama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-20B",
        help="Model name to use (default: gpt-oss-20B)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for cloud providers. Can also be set via OPENROUTER_API_KEY, GEMINI_API_KEY, or OLLAMA_API_KEY depending on the provider/model.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Base URL for the Ollama API (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--no-thinking", action="store_true", help="Disable thinking/reasoning features"
    )
    parser.add_argument(
        "--no-structured-outputs",
        action="store_true",
        help="Disable structured output features",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for output files (default: ./output)",
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate high-quality training datasets using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── build-prompt ──────────────────────────────────────────────────────────
    bp = subparsers.add_parser(
        "build-prompt",
        help="Analyse inputs and save a generation prompt to a file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py build-prompt --goal "Train a coding assistant" \\
      --examples examples/sample_dataset.jsonl \\
      --constraints constraints/constraints.txt \\
      --output-dir ./output/my_run
""",
    )
    _add_provider_args(bp)
    bp.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file (overrides all other flags)",
    )
    bp.add_argument("--goal", type=str, help="Goal/purpose of the dataset")
    bp.add_argument("--examples", type=str, help="Path to example dataset file")
    bp.add_argument("--constraints", type=str, help="Path to constraints file")
    bp.add_argument(
        "--format",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "alpaca", "jsonl"],
        help="Output dataset format used to shape the prompt (default: sharegpt)",
    )
    bp.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Entries per batch — used to calibrate the prompt (default: 10)",
    )

    # ── generate ──────────────────────────────────────────────────────────────
    gen = subparsers.add_parser(
        "generate",
        help="Run dataset generation from a saved prompt file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py generate \\
      --prompt ./output/my_run/generation_prompt.txt \\
      --entries 100 --batch-size 10

  # Load all settings from a JSON config
  python cli.py generate --config config.json
""",
    )
    _add_provider_args(gen)
    gen.add_argument(
        "--prompt",
        type=str,
        help="Path to a prompt file produced by build-prompt (skips re-analysis)",
    )
    gen.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file (overrides all other flags)",
    )
    gen.add_argument(
        "--entries",
        type=int,
        default=100,
        help="Total number of dataset entries to generate (default: 100)",
    )
    gen.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of entries to generate per batch (default: 10)",
    )
    gen.add_argument(
        "--format",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "alpaca", "jsonl"],
        help="Output dataset format (default: sharegpt)",
    )
    gen.add_argument(
        "--max-corrections",
        type=int,
        default=3,
        help="Maximum correction attempts per failed entry (default: 3)",
    )
    gen.add_argument(
        "--no-save-failures",
        action="store_true",
        help="Do not save a failure log",
    )

    return parser


def _reconfigure_stdio():
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")  # type: ignore
    except AttributeError:
        pass


def _resolve_api_key(provider: str, model: str, api_key: str | None) -> str | None:
    if api_key:
        return api_key

    if provider == "openrouter":
        return os.environ.get("OPENROUTER_API_KEY")
    if provider == "gemini":
        return os.environ.get("GEMINI_API_KEY")
    if provider == "ollama" and model.endswith("-cloud"):
        return os.environ.get("OLLAMA_API_KEY")
    return None


def run_build_prompt(args: argparse.Namespace):
    """Handle the build-prompt subcommand."""
    _reconfigure_stdio()

    if args.config:
        print(f"Loading configuration from {args.config}")
        config_data = load_json_config(args.config)
        config = GenerationConfig(**config_data)
    else:
        if not args.goal:
            print("Error: --goal is required when not using --config")
            sys.exit(1)
        
        api_key = _resolve_api_key(args.provider, args.model, args.api_key)

        config = GenerationConfig(
            provider=args.provider,
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            use_thinking=not args.no_thinking,
            use_structured_outputs=not args.no_structured_outputs,
            output_format=args.format,
            batch_size=args.batch_size,
            example_dataset_path=args.examples,
            constraints_path=args.constraints,
            dataset_goal=args.goal,
            output_dir=args.output_dir,
            save_prompt=True,
        )

    print("\n" + "=" * 60)
    print("BUILD GENERATION PROMPT")
    print("=" * 60)
    print(f"Provider : {config.provider}  /  Model: {config.model}")
    print(f"Format   : {config.output_format}")
    print(f"Goal     : {config.dataset_goal}")
    if config.example_dataset_path:
        print(f"Examples : {config.example_dataset_path}")
    if config.constraints_path:
        print(f"Constraints: {config.constraints_path}")
    print(f"Output   : {config.output_dir}")
    print("=" * 60 + "\n")

    try:
        generator = DatasetGenerator(config)

        if not generator.provider.test_connection():
            print(
                f"ERROR: Could not reach {config.provider} provider. Check your API key and network connection."
            )
            sys.exit(1)

        prompt_path = generator.build_prompt()
        print(f"\n✓ Prompt saved to: {prompt_path}")
        print("  Pass it to the generate command with:")
        print(f'  python cli.py generate --prompt "{prompt_path}" --entries <N>')

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def run_generate(args: argparse.Namespace):
    """Handle the generate subcommand."""
    _reconfigure_stdio()

    if args.config:
        print(f"Loading configuration from {args.config}")
        config_data = load_json_config(args.config)
        config = GenerationConfig(**config_data)
    else:
        if not args.prompt:
            print("Error: --prompt <path> is required (or use --config)")
            sys.exit(1)

        api_key = _resolve_api_key(args.provider, args.model, args.api_key)

        config = GenerationConfig(
            provider=args.provider,
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            use_thinking=not args.no_thinking,
            use_structured_outputs=not args.no_structured_outputs,
            total_entries=args.entries,
            batch_size=args.batch_size,
            max_correction_attempts=args.max_corrections,
            output_format=args.format,
            existing_prompt_path=args.prompt,
            output_dir=args.output_dir,
            save_prompt=False,
            save_failures=not args.no_save_failures,
        )

    print("\n" + "=" * 60)
    print("DATASET GENERATION CONFIGURATION")
    print("=" * 60)
    print(f"Provider : {config.provider}  /  Model: {config.model}")
    print(f"Entries  : {config.total_entries}  (batch size: {config.batch_size})")
    print(f"Format   : {config.output_format}")
    print(f"Output   : {config.output_dir}")
    if config.existing_prompt_path:
        print(f"Prompt   : {config.existing_prompt_path}")
    print("=" * 60 + "\n")

    try:
        generator = DatasetGenerator(config)

        if not generator.provider.test_connection():
            print(
                f"ERROR: Could not reach {config.provider} provider. Check your API key and network connection."
            )
            sys.exit(1)

        dataset = generator.generate()

        print(f"\n✓ Successfully generated {len(dataset)} dataset entries!")
        print(f"✓ Output saved to {config.output_dir}")

    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "build-prompt":
        run_build_prompt(args)
    elif args.command == "generate":
        run_generate(args)


if __name__ == "__main__":
    main()
