#!/usr/bin/env python3
"""Standalone hierarchical evaluation script.

Evaluate the full hierarchical pipeline (stage1 → stage2) on a test dataset
without re-training.

Usage:
    python scripts/evaluate_pipeline.py \
        --data Dataset/BERT_training_3000_v2.json \
        --stage1-model results/stage1/best_model.pt \
        --stage2-health-model results/stage2_health/best_model.pt \
        --stage2-tech-model results/stage2_technology/best_model.pt \
        --output-dir results/hierarchical_eval
"""

import argparse
import os
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Evaluate hierarchical pipeline")
    parser.add_argument("--data", required=True, help="Path to JSON test data")
    parser.add_argument("--stage1-model", required=True, help="Path to stage1 best_model.pt")
    parser.add_argument("--stage2-health-model", required=True, help="Path to stage2 health best_model.pt")
    parser.add_argument("--stage2-tech-model", required=True, help="Path to stage2 technology best_model.pt")
    parser.add_argument("--output-dir", default="results/hierarchical_eval", help="Output directory")
    args = parser.parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    cmd = [
        sys.executable, "src/evaluate.py",
        "--data_path", args.data,
        "--stage1_model", args.stage1_model,
        "--stage2_health_model", args.stage2_health_model,
        "--stage2_tech_model", args.stage2_tech_model,
        "--output_dir", args.output_dir,
    ]

    print("=" * 60)
    print("  Hierarchical Pipeline Evaluation")
    print("=" * 60)
    print(f"  Data:         {args.data}")
    print(f"  Stage 1:      {args.stage1_model}")
    print(f"  Stage 2 (H):  {args.stage2_health_model}")
    print(f"  Stage 2 (T):  {args.stage2_tech_model}")
    print(f"  Output:       {args.output_dir}")
    print("=" * 60 + "\n")

    subprocess.run(cmd, cwd=project_dir)


if __name__ == "__main__":
    main()
