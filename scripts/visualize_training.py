#!/usr/bin/env python3
"""
Visualise training curves from training_log_final.json.

Generates:
  - Loss curves (train & val)
  - F1 curves (train & val)
  - Accuracy curves
  - Per-class F1 bar chart (test)
  - Confusion matrix heatmap (test)

Usage:
    python scripts/visualize_training.py --log results/training_log_final.json --output results/visualizations
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_log(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_loss(epochs_data, out_dir):
    epochs     = [e["epoch"] for e in epochs_data]
    train_loss = [e["train_loss"] for e in epochs_data]
    val_loss   = [e["val_metrics"]["loss"] for e in epochs_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, "o-", label="Train loss")
    ax.plot(epochs, val_loss,   "s-", label="Val loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)


def plot_f1(epochs_data, out_dir):
    epochs   = [e["epoch"] for e in epochs_data]
    train_f1 = [e["train_metrics"]["macro_f1"] for e in epochs_data]
    val_f1   = [e["val_metrics"]["macro_f1"] for e in epochs_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_f1, "o-", label="Train Macro F1")
    ax.plot(epochs, val_f1,   "s-", label="Val Macro F1")
    ax.set(xlabel="Epoch", ylabel="Macro F1", title="Training & Validation Macro F1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "f1_curves.png"), dpi=150)
    plt.close(fig)


def plot_accuracy(epochs_data, out_dir):
    epochs    = [e["epoch"] for e in epochs_data]
    train_acc = [e["train_metrics"]["accuracy"] for e in epochs_data]
    val_acc   = [e["val_metrics"]["accuracy"] for e in epochs_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_acc, "o-", label="Train Accuracy")
    ax.plot(epochs, val_acc,   "s-", label="Val Accuracy")
    ax.set(xlabel="Epoch", ylabel="Accuracy", title="Training & Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_curves.png"), dpi=150)
    plt.close(fig)


def plot_per_class_f1(test_metrics, label_names, out_dir):
    f1s = [test_metrics.get(f"{name}_f1", 0) for name in label_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(label_names, f1s, color=["#2196F3", "#FF9800", "#4CAF50"])
    ax.set(xlabel="Class", ylabel="F1 Score", title="Test F1 per Class")
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.3f}", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_class_f1.png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrix(test_metrics, label_names, out_dir):
    cm = np.array(test_metrics.get("confusion_matrix", []))
    if cm.size == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualise training results")
    parser.add_argument("--log", required=True, help="Path to training_log_final.json")
    parser.add_argument("--output", default="results/visualizations", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    log = load_log(args.log)

    epochs_data  = log["epochs"]
    test_metrics = log.get("test_metrics", {})
    label_names  = log["config"].get("label_names",
                                     ["central_route", "peripheral_route", "neutral_route"])
    # If label_names not in log, read from Config
    if not label_names:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from config import Config
        label_names = Config().label_names

    plot_loss(epochs_data, args.output)
    plot_f1(epochs_data, args.output)
    plot_accuracy(epochs_data, args.output)
    if test_metrics:
        plot_per_class_f1(test_metrics, label_names, args.output)
        plot_confusion_matrix(test_metrics, label_names, args.output)

    print(f"Visualisations saved to {args.output}/")


if __name__ == "__main__":
    main()
