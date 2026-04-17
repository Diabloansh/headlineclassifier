#!/usr/bin/env python3
"""Orchestrate the full hierarchical training pipeline.

Trains all three models sequentially:
  1. Stage 1  — topic classifier (health vs technology)
  2. Stage 2a — route classifier for health headlines
  3. Stage 2b — route classifier for technology headlines

Usage:
    python scripts/train_all.py --data Dataset/BERT_training_3000_v2.json
    python scripts/train_all.py --data Dataset/BERT_training_3000_v2.json --epochs 25 --patience 5
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime


def run_stage(description: str, cmd: list, cwd: str):
    """Run a training stage as a subprocess and stream output."""
    print("\n" + "=" * 70)
    print(f"  {description}")
    print("=" * 70)
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 70 + "\n")

    result = subprocess.run(cmd, cwd=cwd)

    if result.returncode != 0:
        print(f"\n*** ERROR: {description} failed with return code {result.returncode} ***")
        sys.exit(result.returncode)

    print(f"\n  Finished: {datetime.now().isoformat()}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train full hierarchical pipeline")
    parser.add_argument("--data", required=True, help="Path to JSON dataset")
    parser.add_argument("--test-data", default=None, help="Separate test JSON (optional)")
    parser.add_argument("--output-dir", default="results", help="Base output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs per stage")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    args = parser.parse_args()

    # Resolve paths
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    pipeline_start = datetime.now()

    print("\n" + "#" * 70)
    print("  HIERARCHICAL HEADLINE CLASSIFIER — Full Training Pipeline")
    print("#" * 70)
    print(f"  Dataset:    {args.data}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.learning_rate}")
    print(f"  Patience:   {args.patience}")
    print("#" * 70)

    python = sys.executable

    # ---- Stage 1: Topic classifier ----
    stage1_dir = os.path.join(args.output_dir, "stage1")
    stage1_cmd = [
        python, "src/train_stage1.py",
        "--data_path", args.data,
        "--output_dir", stage1_dir,
        "--patience", str(args.patience),
    ]
    if args.test_data:
        stage1_cmd.extend(["--test_path", args.test_data])

    run_stage("STAGE 1 — Topic Classifier (health vs technology)", stage1_cmd, project_dir)

    # ---- Stage 2a: Route classifier (health) ----
    stage2_health_dir = os.path.join(args.output_dir, "stage2_health")
    stage2_health_cmd = [
        python, "src/train_stage2.py",
        "--data_path", args.data,
        "--topic", "health",
        "--output_dir", stage2_health_dir,
        "--patience", str(args.patience),
    ]
    if args.test_data:
        stage2_health_cmd.extend(["--test_path", args.test_data])

    run_stage("STAGE 2a — Route Classifier (health)", stage2_health_cmd, project_dir)

    # ---- Stage 2b: Route classifier (technology) ----
    stage2_tech_dir = os.path.join(args.output_dir, "stage2_technology")
    stage2_tech_cmd = [
        python, "src/train_stage2.py",
        "--data_path", args.data,
        "--topic", "technology",
        "--output_dir", stage2_tech_dir,
        "--patience", str(args.patience),
    ]
    if args.test_data:
        stage2_tech_cmd.extend(["--test_path", args.test_data])

    run_stage("STAGE 2b — Route Classifier (technology)", stage2_tech_cmd, project_dir)

    # ---- Summary ----
    total_time = (datetime.now() - pipeline_start).total_seconds()

    print("\n" + "#" * 70)
    print("  PIPELINE COMPLETE")
    print("#" * 70)
    print(f"  Total time:  {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Stage 1:     {stage1_dir}/")
    print(f"  Stage 2a:    {stage2_health_dir}/")
    print(f"  Stage 2b:    {stage2_tech_dir}/")
    print(f"\n  Model paths:")
    print(f"    Stage 1:     {stage1_dir}/best_model.pt")
    print(f"    Stage 2 (H): {stage2_health_dir}/best_model.pt")
    print(f"    Stage 2 (T): {stage2_tech_dir}/best_model.pt")
    print("#" * 70)

    # ---- Run hierarchical evaluation ----
    eval_dir = os.path.join(args.output_dir, "hierarchical_eval")
    eval_cmd = [
        python, "src/evaluate.py",
        "--data_path", args.data,
        "--stage1_model", os.path.join(stage1_dir, "best_model.pt"),
        "--stage2_health_model", os.path.join(stage2_health_dir, "best_model.pt"),
        "--stage2_tech_model", os.path.join(stage2_tech_dir, "best_model.pt"),
        "--output_dir", eval_dir,
    ]
    run_stage("HIERARCHICAL EVALUATION", eval_cmd, project_dir)


if __name__ == "__main__":
    main()
