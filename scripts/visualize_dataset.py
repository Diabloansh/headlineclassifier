#!/usr/bin/env python3
"""
Visualise the dataset distribution and text-length statistics.

Usage:
    python scripts/visualize_dataset.py --data Dataset/1ansh.json --output results/dataset_analysis
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True, help="Path to JSON dataset")
    parser.add_argument("--output", default="results/dataset_analysis")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    config = Config()

    with open(args.data) as f:
        data = json.load(f)

    # --- class distribution ---
    counts = {name: 0 for name in config.label_names}
    for item in data:
        for name in config.label_names:
            field = config.field_mapping[name]
            if item[field] == 1:
                counts[name] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.keys(), counts.values(),
                  color=["#2196F3", "#FF9800", "#4CAF50"])
    ax.set(xlabel="Class", ylabel="Count",
           title=f"Class Distribution (N={len(data)})")
    for bar, v in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(v), ha="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "class_distribution.png"), dpi=150)
    plt.close(fig)

    # --- topic distribution ---
    topics = {}
    for item in data:
        t = item.get("topic", "unknown")
        topics[t] = topics.get(t, 0) + 1

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(topics.keys(), topics.values(), color="#9C27B0")
    ax.set(xlabel="Topic", ylabel="Count", title="Topic Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "topic_distribution.png"), dpi=150)
    plt.close(fig)

    # --- text length distribution ---
    lengths = [len(item["text"].split()) for item in data]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=30, color="#009688", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(lengths), color="red", linestyle="--",
               label=f"Mean = {np.mean(lengths):.1f} words")
    ax.set(xlabel="Word count", ylabel="Frequency",
           title="Headline Length Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "text_length_distribution.png"), dpi=150)
    plt.close(fig)

    # --- summary stats ---
    summary = {
        "total_samples":  len(data),
        "class_counts":   counts,
        "topic_counts":   topics,
        "text_length_stats": {
            "mean":   float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "min":    int(np.min(lengths)),
            "max":    int(np.max(lengths)),
            "std":    float(np.std(lengths)),
        },
    }
    with open(os.path.join(args.output, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Dataset analysis saved to {args.output}/")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
