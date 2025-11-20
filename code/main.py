"""Main entry point for FakeLenseV2"""

import argparse
import sys

from code.train import train_from_config
from code.inference import evaluate_from_config, InferenceEngine


def train(args):
    """Execute training"""
    train_from_config(args.config)


def evaluate(args):
    """Execute evaluation"""
    evaluate_from_config(args.config, args.model)


def infer_cli(args):
    """Execute single inference from command line"""
    engine = InferenceEngine(args.model)

    prediction = engine.predict(
        text=args.text,
        source=args.source or "Unknown",
        social_reactions=args.reactions or 0,
    )

    label_map = {0: "Fake News", 1: "Suspicious News", 2: "Real News"}
    print(f"\nPrediction: {label_map[prediction]}")
    print(f"Class: {prediction}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="FakeLenseV2: AI-Powered Fake News Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python -m code.main train --config config.json

  # Evaluate the model
  python -m code.main evaluate --model models/best_model.pth

  # Make a single prediction
  python -m code.main infer --model models/best_model.pth \\
      --text "Breaking news article text..." \\
      --source "Reuters" \\
      --reactions 5000
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration JSON file"
    )
    train_parser.set_defaults(func=train)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration JSON file"
    )
    eval_parser.add_argument(
        "--model", type=str, default=None, help="Path to trained model file"
    )
    eval_parser.set_defaults(func=evaluate)

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Make a prediction")
    infer_parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model file"
    )
    infer_parser.add_argument(
        "--text", type=str, required=True, help="Article text to analyze"
    )
    infer_parser.add_argument(
        "--source", type=str, default=None, help="News source name"
    )
    infer_parser.add_argument(
        "--reactions", type=float, default=None, help="Number of social media reactions"
    )
    infer_parser.set_defaults(func=infer_cli)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
