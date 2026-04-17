#!/usr/bin/env python3
"""
Generate a human-readable results log from training output files.

Reads training_log_final.json, test_metrics.json, and training_summary.json
from the results directory and produces a comprehensive results_log.txt.
"""

import json
import os
import sys
from datetime import datetime


def load_json(path):
    with open(path) as f:
        return json.load(f)


def format_results_log(results_dir: str) -> str:
    """Generate a comprehensive results log string."""

    # Load all available result files
    training_log = load_json(os.path.join(results_dir, "training_log_final.json"))
    test_metrics = load_json(os.path.join(results_dir, "test_metrics.json"))
    summary = load_json(os.path.join(results_dir, "training_summary.json"))

    config = training_log["config"]
    ts = summary["training_summary"]
    epochs = summary["per_epoch_summary"]

    lines = []
    sep = "=" * 80
    dash = "-" * 80

    # ── Header ──
    lines.append(sep)
    lines.append("  HEADLINE PERSUASION-ROUTE CLASSIFIER — TRAINING RESULTS LOG")
    lines.append(sep)
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ── Model & Hyperparameters ──
    lines.append(sep)
    lines.append("  MODEL CONFIGURATION & HYPERPARAMETERS")
    lines.append(sep)
    lines.append(f"  Model:              {config['model_name']}")
    lines.append(f"  Number of labels:   {config['num_labels']}")
    lines.append(f"  Max sequence length: {config['max_length']}")
    lines.append(f"  Dropout rate:       {config['dropout_rate']}")
    lines.append("")
    lines.append(f"  Epochs:             {config['num_epochs']}")
    lines.append(f"  Batch size:         {config['batch_size']}")
    lines.append(f"  Learning rate:      {config['learning_rate']}")
    lines.append(f"  Weight decay:       {config['weight_decay']}")
    lines.append(f"  Warmup ratio:       {config['warmup_ratio']}")
    lines.append(f"  LR scheduler:       {config['lr_scheduler']}")
    lines.append("")
    lines.append(f"  Device:             {training_log['device']}")
    lines.append(f"  Training started:   {training_log['start_time']}")
    lines.append(f"  Training ended:     {training_log['end_time']}")
    lines.append(f"  Total duration:     {training_log['total_duration_seconds']:.1f} seconds "
                 f"({training_log['total_duration_seconds']/60:.1f} minutes)")
    lines.append("")

    # ── Dataset sizes ──
    ds = training_log["dataset_sizes"]
    lines.append(dash)
    lines.append("  DATASET SPLIT SIZES")
    lines.append(dash)
    lines.append(f"  Train:  {ds['train']}")
    lines.append(f"  Val:    {ds['val']}")
    lines.append(f"  Test:   {ds['test']}")
    lines.append(f"  Total:  {ds['train'] + ds['val'] + ds['test']}")
    lines.append("")

    # ── Best epoch ──
    lines.append(dash)
    lines.append("  BEST MODEL (selected by validation macro-F1)")
    lines.append(dash)
    lines.append(f"  Best epoch:         {ts['best_epoch']}")
    lines.append(f"  Best val macro-F1:  {ts['best_val_f1']:.4f}")
    lines.append("")

    # ── Final test set results ──
    lines.append(sep)
    lines.append("  FINAL TEST SET RESULTS")
    lines.append(sep)
    lines.append(f"  Accuracy:           {test_metrics['accuracy']:.4f}")
    lines.append(f"  Macro F1:           {test_metrics['macro_f1']:.4f}")
    lines.append(f"  Weighted F1:        {test_metrics['weighted_f1']:.4f}")
    lines.append(f"  Macro Precision:    {test_metrics['macro_precision']:.4f}")
    lines.append(f"  Macro Recall:       {test_metrics['macro_recall']:.4f}")
    lines.append(f"  ROC AUC (OvR):      {test_metrics['roc_auc']:.4f}")
    lines.append("")

    # ── Per-class test metrics ──
    label_names = ["central_route", "peripheral_route", "neutral_route"]
    lines.append(dash)
    lines.append("  PER-CLASS TEST METRICS")
    lines.append(dash)
    header = f"  {'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    lines.append(header)
    lines.append("  " + "-" * 64)
    for name in label_names:
        p = test_metrics.get(f"{name}_precision", 0)
        r = test_metrics.get(f"{name}_recall", 0)
        f = test_metrics.get(f"{name}_f1", 0)
        s = test_metrics.get(f"{name}_support", 0)
        lines.append(f"  {name:<22} {p:>10.4f} {r:>10.4f} {f:>10.4f} {s:>10}")
    lines.append("")

    # ── Confusion matrix ──
    cm = test_metrics.get("confusion_matrix", [])
    lines.append(dash)
    lines.append("  CONFUSION MATRIX (rows = true label, cols = predicted label)")
    lines.append(dash)
    short_names = ["Central", "Periph.", "Neutral"]
    header_cm = f"  {'':>15}" + "".join(f"{n:>12}" for n in short_names)
    lines.append(header_cm)
    for i, row in enumerate(cm):
        row_str = f"  {short_names[i]:>15}" + "".join(f"{v:>12}" for v in row)
        lines.append(row_str)
    lines.append("")

    # ── Per-epoch training log ──
    lines.append(sep)
    lines.append("  PER-EPOCH TRAINING LOG")
    lines.append(sep)
    header_ep = (
        f"  {'Epoch':>5}  {'Train Loss':>11}  {'Train F1':>9}  {'Train Acc':>10}  "
        f"{'Val F1':>7}  {'Val Acc':>8}  {'Val Loss':>9}  {'Best':>5}  {'Time(s)':>8}"
    )
    lines.append(header_ep)
    lines.append("  " + "-" * 90)
    for ep in epochs:
        best_marker = "  *" if ep["is_best"] else ""
        lines.append(
            f"  {ep['epoch']:>5}  {ep['train_loss']:>11.4f}  {ep['train_f1']:>9.4f}  "
            f"{ep['train_acc']:>10.4f}  {ep['val_f1']:>7.4f}  {ep['val_acc']:>8.4f}  "
            f"{ep['val_loss']:>9.4f}  {best_marker:>5}  {ep['duration']:>8.1f}"
        )
    lines.append("")

    # ── Footer ──
    lines.append(sep)
    lines.append("  END OF RESULTS LOG")
    lines.append(sep)
    lines.append("")

    return "\n".join(lines)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"

    if not os.path.isdir(results_dir):
        print(f"Error: results directory '{results_dir}' not found.")
        sys.exit(1)

    log_text = format_results_log(results_dir)

    output_path = os.path.join(results_dir, "results_log.txt")
    with open(output_path, "w") as f:
        f.write(log_text)

    print(log_text)
    print(f"\n  Results log saved to: {output_path}")


if __name__ == "__main__":
    main()
