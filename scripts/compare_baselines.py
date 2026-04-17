#!/usr/bin/env python3
"""
Compare the trained model against two baselines:
  1. Random uniform baseline
  2. Class-prior (majority vote per class prior) baseline

Usage:
    python scripts/compare_baselines.py \
        --model results/best_model.pt \
        --data  Dataset/1ansh.json \
        --output results
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from dataset import load_data, HeadlineDataset, create_datasets
from model import build_model
from utils import compute_metrics, save_metrics, set_seed, setup_logging


def random_baseline_metrics(true_labels, num_classes, label_names, seed=42):
    rng = np.random.RandomState(seed)
    n = len(true_labels)
    # Uniform random probabilities
    probs = rng.dirichlet(np.ones(num_classes), size=n)
    return compute_metrics(probs, np.array(true_labels), label_names)


def class_prior_baseline_metrics(train_labels, test_labels, num_classes, label_names):
    from collections import Counter
    counts = Counter(train_labels)
    total  = len(train_labels)
    priors = np.array([counts.get(i, 0) / total for i in range(num_classes)])

    # Each test sample gets the same prior distribution
    n = len(test_labels)
    probs = np.tile(priors, (n, 1))
    return compute_metrics(probs, np.array(test_labels), label_names)


def model_metrics(model_path, test_texts, test_labels, config, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg       = checkpoint["config"]
    tokenizer = checkpoint["tokenizer"]

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = HeadlineDataset(test_texts, test_labels, tokenizer, cfg.max_length)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(outputs["logits"], dim=-1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    return compute_metrics(np.array(all_probs), np.array(all_labels), config.label_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="Path to best_model.pt")
    parser.add_argument("--data",   required=True, help="Path to JSON dataset")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    setup_logging()
    set_seed(42)

    config = Config()

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create the same splits used during training (deterministic seed)
    train_ds, val_ds, test_ds, tokenizer = create_datasets(args.data, config)
    test_texts  = test_ds.texts
    test_labels = test_ds.labels
    train_labels = train_ds.labels

    # 1. Random baseline
    rand_metrics = random_baseline_metrics(test_labels, config.num_labels, config.label_names)

    # 2. Class-prior baseline
    prior_metrics = class_prior_baseline_metrics(
        train_labels, test_labels, config.num_labels, config.label_names
    )

    # 3. Model
    mdl_metrics = model_metrics(args.model, test_texts, test_labels, config, device)

    # Report
    os.makedirs(args.output, exist_ok=True)

    for name, base, fname in [
        ("Random Baseline",      rand_metrics,  "random_baseline_comparison.json"),
        ("Class-Prior Baseline", prior_metrics, "class_prior_baseline_comparison.json"),
    ]:
        comparison = {
            "baseline_name":    name,
            "test_set_size":    len(test_labels),
            "baseline_metrics": base,
            "model_metrics":    mdl_metrics,
            "improvements": {
                "macro_f1_diff":  mdl_metrics["macro_f1"]  - base["macro_f1"],
                "accuracy_diff":  mdl_metrics["accuracy"]  - base["accuracy"],
                "roc_auc_diff":   mdl_metrics["roc_auc"]   - base["roc_auc"],
            },
        }
        path = os.path.join(args.output, fname)
        save_metrics(comparison, path)

        print(f"\n{'=' * 55}")
        print(f"  {name} vs Model")
        print(f"{'=' * 55}")
        print(f"  {'Metric':<18} {'Baseline':>10} {'Model':>10} {'Diff':>10}")
        print(f"  {'-'*48}")
        for m in ["macro_f1", "accuracy", "roc_auc"]:
            print(f"  {m:<18} {base[m]:>10.4f} {mdl_metrics[m]:>10.4f} "
                  f"{mdl_metrics[m] - base[m]:>+10.4f}")

    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
