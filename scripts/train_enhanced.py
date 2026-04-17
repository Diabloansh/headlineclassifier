#!/usr/bin/env python3
"""
Launch training for the headline persuasion-route classifier.

Usage:
    python scripts/train_enhanced.py --data Dataset/1ansh.json
    python scripts/train_enhanced.py --data Dataset/1ansh.json --output-dir results_custom --epochs 25
"""

import argparse
import os
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Launch headline classifier training")
    parser.add_argument("--data", required=True, help="Path to JSON dataset")
    parser.add_argument("--test-data", default=None, help="Separate test JSON (optional)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: results_TIMESTAMP)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience (epochs without val F1 improvement)")

    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results_{timestamp}"

    # Resolve paths relative to project root
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)

    cmd_parts = [
        sys.executable, "src/train.py",
        "--data_path", args.data,
        "--output_dir", args.output_dir,
        "--patience", str(args.patience),
    ]
    if args.test_data:
        cmd_parts.extend(["--test_path", args.test_data])

    print("=" * 60)
    print("  Headline Persuasion-Route Classifier — Training")
    print("=" * 60)
    print(f"  Dataset:    {args.data}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.learning_rate}")
    print(f"  Patience:   {args.patience} epochs")
    print("=" * 60)
    print(f"  Command: {' '.join(cmd_parts)}\n")

    os.execv(sys.executable, cmd_parts)


if __name__ == "__main__":
    main()
