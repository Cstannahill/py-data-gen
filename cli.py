#!/usr/bin/env python3
"""
Command-line interface for the dataset generator
"""

import argparse
import sys
import json
from pathlib import Path

from dataset_generator import DatasetGenerator, GenerationConfig


def load_json_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Generate high-quality training datasets using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python cli.py --goal "Train a coding assistant" --entries 100
  
  # With examples and constraints
  python cli.py --goal "Train a helpful assistant" \\
    --examples examples/sample_dataset.json \\
    --constraints examples/constraints.txt \\
    --entries 50 --batch-size 10
  
  # Using a config file
  python cli.py --config config.json
  
  # Custom model and format
  python cli.py --model mistral --format alpaca \\
    --goal "Train a technical writer" --entries 30
        """,
    )

    # Config file option
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file (overrides other arguments)",
    )

    # Core parameters
    parser.add_argument("--goal", type=str, help="Goal/purpose of the dataset")

    parser.add_argument(
        "--entries",
        type=int,
        default=100,
        help="Total number of dataset entries to generate (default: 100)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of entries to generate per batch (default: 10)",
    )

    # Input files
    parser.add_argument("--examples", type=str, help="Path to example dataset file")

    parser.add_argument(
        "--constraints", type=str, help="Path to constraints/instructions file"
    )

    # Model configuration
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama"],
        help="LLM provider to use (default: ollama)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-20B",
        help="Model name to use (default: gpt-oss-20B)",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Base URL for Ollama API (default: http://localhost:11434)",
    )

    parser.add_argument(
        "--no-thinking", action="store_true", help="Disable thinking/reasoning features"
    )

    parser.add_argument(
        "--no-structured-outputs",
        action="store_true",
        help="Disable structured output features",
    )

    # Output configuration
    parser.add_argument(
        "--format",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "alpaca"],
        help="Output dataset format (default: sharegpt)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for output files (default: ./output)",
    )

    parser.add_argument(
        "--no-save-prompt",
        action="store_true",
        help="Do not save generation prompt to file",
    )

    parser.add_argument(
        "--no-save-failures",
        action="store_true",
        help="Do not save failure log to file",
    )

    # Generation parameters
    parser.add_argument(
        "--max-corrections",
        type=int,
        default=3,
        help="Maximum correction attempts per failed entry (default: 3)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)",
    )

    return parser


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Load from config file if provided
    if args.config:
        print(f"Loading configuration from {args.config}")
        config_data = load_json_config(args.config)
        config = GenerationConfig(**config_data)
    else:
        # Validate required arguments
        if not args.goal:
            print("Error: --goal is required when not using --config")
            parser.print_help()
            sys.exit(1)

        # Build config from arguments
        config = GenerationConfig(
            provider=args.provider,
            model=args.model,
            use_thinking=not args.no_thinking,
            use_structured_outputs=not args.no_structured_outputs,
            total_entries=args.entries,
            batch_size=args.batch_size,
            max_correction_attempts=args.max_corrections,
            output_format=args.format,
            example_dataset_path=args.examples,
            constraints_path=args.constraints,
            dataset_goal=args.goal,
            output_dir=args.output_dir,
            save_prompt=not args.no_save_prompt,
            save_failures=not args.no_save_failures,
        )
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")  # type: ignore
    except AttributeError:
        pass
    # Display configuration
    print("\n" + "=" * 60)
    print("DATASET GENERATION CONFIGURATION")
    print("=" * 60)
    print(f"Provider: {config.provider}")
    print(f"Model: {config.model}")
    print(f"Total Entries: {config.total_entries}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Output Format: {config.output_format}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Goal: {config.dataset_goal}")
    if config.example_dataset_path:
        print(f"Examples: {config.example_dataset_path}")
    if config.constraints_path:
        print(f"Constraints: {config.constraints_path}")
    print("=" * 60 + "\n")

    # Create generator and run
    try:
        generator = DatasetGenerator(config)

        # Test connection first
        if not generator.provider.test_connection():
            print(f"ERROR: Could not connect to Ollama at {args.base_url}")
            print("Please ensure Ollama is running and accessible.")
            sys.exit(1)

        # Generate dataset
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


if __name__ == "__main__":
    main()
