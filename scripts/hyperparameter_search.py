#!/usr/bin/env python3
"""
Hyperparameter search for the headline persuasion-route classifier.

Uses **Stratified K-Fold cross-validation** so that every sample contributes
to both training and validation across folds — critical for a 300-sample dataset.

Supports two search strategies:
  • random  – sample N random configurations from the search space (default)
  • grid    – exhaustive grid over a (smaller) predefined grid

Usage
-----
    # Random search with 30 trials, 5-fold CV
    python scripts/hyperparameter_search.py \
        --data Dataset/1ansh.json \
        --output results/hp_search \
        --strategy random \
        --n_trials 30 \
        --n_folds 5

    # Grid search (uses a compact grid; see GRID_SPACE below)
    python scripts/hyperparameter_search.py \
        --data Dataset/1ansh.json \
        --output results/hp_search \
        --strategy grid

    # After the search, retrain the best config on full train+val and evaluate on test
    python scripts/hyperparameter_search.py \
        --data Dataset/1ansh.json \
        --output results/hp_search \
        --retrain_best
"""

import argparse
import copy
import itertools
import json
import logging
import os
import random as py_random
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from dataset import HeadlineDataset, load_data
from model import build_model
from utils import compute_metrics, save_metrics, set_seed, setup_logging


# ============================================================================
# Search spaces
# ============================================================================

# Full search space for random sampling
RANDOM_SEARCH_SPACE: Dict[str, list] = {
    "learning_rate":     [1e-5, 2e-5, 3e-5, 5e-5],
    "batch_size":        [8, 16, 32],
    "dropout_rate":      [0.1, 0.2, 0.3],
    "weight_decay":      [0.001, 0.01, 0.05, 0.1],
    "warmup_ratio":      [0.0, 0.05, 0.1, 0.2],
    "num_epochs":        [10, 15, 20],
    "lr_scheduler_type": ["cosine", "linear"],
    "max_length":        [64, 128],
}

# Smaller grid for exhaustive search (keeps total combos manageable)
GRID_SEARCH_SPACE: Dict[str, list] = {
    "learning_rate":     [1e-5, 2e-5, 3e-5],
    "batch_size":        [8, 16],
    "dropout_rate":      [0.1, 0.2, 0.3],
    "weight_decay":      [0.01, 0.05],
    "warmup_ratio":      [0.1],
    "num_epochs":        [15, 20],
    "lr_scheduler_type": ["cosine"],
    "max_length":        [128],
}

# Early stopping patience (within each trial)
EARLY_STOPPING_PATIENCE = 4


# ============================================================================
# Helpers
# ============================================================================

def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_config(hp: Dict[str, Any]) -> Config:
    """Create a Config with hyperparameters overridden."""
    cfg = Config()
    for key, value in hp.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _sample_random_configs(space: Dict[str, list], n: int, seed: int = 42) -> List[Dict]:
    """Sample n unique random configurations from the search space."""
    rng = py_random.Random(seed)
    configs: List[Dict] = []
    seen = set()

    max_possible = 1
    for vals in space.values():
        max_possible *= len(vals)

    n = min(n, max_possible)

    while len(configs) < n:
        cfg = {k: rng.choice(v) for k, v in space.items()}
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    return configs


def _grid_configs(space: Dict[str, list]) -> List[Dict]:
    """Enumerate all combinations in the grid."""
    keys = list(space.keys())
    values = list(space.values())
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


# ============================================================================
# Single-fold training + evaluation
# ============================================================================

def train_fold(
    hp: Dict[str, Any],
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    tokenizer,
    device: torch.device,
    fold_idx: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train one fold and return best validation metrics + epoch count."""
    config = _make_config(hp)

    train_dataset = HeadlineDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset   = HeadlineDataset(val_texts, val_labels, tokenizer, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False)

    # Class weights (inverse frequency)
    label_counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor(
        [total / (config.num_labels * label_counts.get(i, 1)) for i in range(config.num_labels)],
        dtype=torch.float32,
    ).to(device)

    model = build_model(config, class_weights=class_weights).to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps  = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    if config.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )

    best_val_f1 = 0.0
    best_val_metrics: Dict = {}
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config.num_epochs):
        # --- train ---
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            labs  = batch["labels"].to(device)

            out = model(input_ids=ids, attention_mask=mask, labels=labs)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += out["loss"].item()

        avg_loss = epoch_loss / len(train_loader)

        # --- validate ---
        model.eval()
        all_probs, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labs = batch["labels"].to(device)

                out = model(input_ids=ids, attention_mask=mask, labels=labs)
                val_loss += out["loss"].item()
                probs = torch.softmax(out["logits"], dim=-1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labs.cpu().numpy())

        val_metrics = compute_metrics(
            np.array(all_probs), np.array(all_labels), config.label_names,
        )
        val_metrics["loss"] = val_loss / max(len(val_loader), 1)

        if verbose:
            logging.info(
                f"    Fold {fold_idx+1} | Epoch {epoch+1:2d}/{config.num_epochs} | "
                f"Train loss {avg_loss:.4f} | Val F1 {val_metrics['macro_f1']:.4f} | "
                f"Val Acc {val_metrics['accuracy']:.4f}"
            )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_val_metrics = val_metrics
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            if verbose:
                logging.info(f"    Early stopping at epoch {epoch+1} (patience={EARLY_STOPPING_PATIENCE})")
            break

    return {
        "best_val_f1":       best_val_f1,
        "best_val_accuracy": best_val_metrics.get("accuracy", 0.0),
        "best_val_roc_auc":  best_val_metrics.get("roc_auc", 0.0),
        "best_epoch":        best_epoch,
        "epochs_run":        epoch + 1,
        "val_metrics":       best_val_metrics,
    }


# ============================================================================
# Cross-validated trial
# ============================================================================

def run_trial(
    trial_idx: int,
    hp: Dict[str, Any],
    texts: List[str],
    labels: List[int],
    n_folds: int,
    device: torch.device,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one hyperparameter config with stratified k-fold CV."""
    config = _make_config(hp)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results: List[Dict] = []

    texts_arr  = np.array(texts)
    labels_arr = np.array(labels)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts_arr, labels_arr)):
        fold_train_texts  = texts_arr[train_idx].tolist()
        fold_train_labels = labels_arr[train_idx].tolist()
        fold_val_texts    = texts_arr[val_idx].tolist()
        fold_val_labels   = labels_arr[val_idx].tolist()

        fold_result = train_fold(
            hp, fold_train_texts, fold_train_labels,
            fold_val_texts, fold_val_labels,
            tokenizer, device, fold_idx, verbose=verbose,
        )
        fold_results.append(fold_result)

    # Aggregate across folds
    f1_scores  = [r["best_val_f1"] for r in fold_results]
    acc_scores = [r["best_val_accuracy"] for r in fold_results]
    auc_scores = [r["best_val_roc_auc"] for r in fold_results]

    return {
        "trial":       trial_idx + 1,
        "hyperparams": hp,
        "mean_val_f1":       float(np.mean(f1_scores)),
        "std_val_f1":        float(np.std(f1_scores)),
        "mean_val_accuracy": float(np.mean(acc_scores)),
        "std_val_accuracy":  float(np.std(acc_scores)),
        "mean_val_roc_auc":  float(np.mean(auc_scores)),
        "std_val_roc_auc":   float(np.std(auc_scores)),
        "per_fold":          fold_results,
    }


# ============================================================================
# Retrain best config on full train+val, evaluate on held-out test
# ============================================================================

def retrain_best(
    best_hp: Dict[str, Any],
    data_path: str,
    output_dir: str,
    device: torch.device,
):
    """Retrain the best hyperparameter config on the full dataset and evaluate on test."""
    from dataset import create_datasets

    config = _make_config(best_hp)
    config.results_dir = output_dir
    config.model_save_path = os.path.join(output_dir, "best_model")

    train_ds, val_ds, test_ds, tokenizer = create_datasets(data_path, config)

    # Merge train + val for final training
    merged_texts  = train_ds.texts + val_ds.texts
    merged_labels = train_ds.labels + val_ds.labels

    merged_dataset = HeadlineDataset(merged_texts, merged_labels, tokenizer, config.max_length)
    test_loader    = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    train_loader   = DataLoader(merged_dataset, batch_size=config.batch_size, shuffle=True)

    # Class weights
    label_counts = Counter(merged_labels)
    total = len(merged_labels)
    class_weights = torch.tensor(
        [total / (config.num_labels * label_counts.get(i, 1)) for i in range(config.num_labels)],
        dtype=torch.float32,
    ).to(device)

    model = build_model(config, class_weights=class_weights).to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps  = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    if config.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )

    logging.info(f"Retraining best config for {config.num_epochs} epochs on "
                 f"{len(merged_dataset)} samples (train+val merged)")

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            out = model(input_ids=ids, attention_mask=mask, labels=labs)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += out["loss"].item()

        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"  Epoch {epoch+1:2d}/{config.num_epochs} | Train loss {avg_loss:.4f}")

    # Evaluate on held-out test set
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out["logits"], dim=-1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    test_metrics = compute_metrics(np.array(all_probs), np.array(all_labels), config.label_names)

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = config.model_save_path + ".pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config":           config,
        "tokenizer":        tokenizer,
        "hyperparams":      best_hp,
        "test_metrics":     test_metrics,
    }, model_path)

    save_metrics(test_metrics, os.path.join(output_dir, "best_retrained_test_metrics.json"))

    logging.info("=" * 60)
    logging.info("Retrained model — Test set results:")
    logging.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logging.info(f"  Macro F1:  {test_metrics['macro_f1']:.4f}")
    logging.info(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")
    logging.info(f"  Model saved to {model_path}")
    logging.info("=" * 60)

    return test_metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for headline persuasion-route classifier",
    )
    parser.add_argument("--data",      type=str, required=True,  help="Path to JSON dataset")
    parser.add_argument("--output",    type=str, default="results/hp_search", help="Output directory")
    parser.add_argument("--strategy",  type=str, default="random", choices=["random", "grid"],
                        help="Search strategy: 'random' or 'grid'")
    parser.add_argument("--n_trials",  type=int, default=30,
                        help="Number of random trials (ignored for grid)")
    parser.add_argument("--n_folds",   type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--seed",      type=int, default=42, help="Random seed")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print per-epoch per-fold logs")
    parser.add_argument("--retrain_best", action="store_true",
                        help="After search, retrain the best config and evaluate on test set")
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    os.makedirs(args.output, exist_ok=True)
    device = _device()
    logging.info(f"Device: {device}")

    # Load data once (used for all folds)
    config = Config()
    texts, labels = load_data(args.data, config)
    logging.info(f"Total samples: {len(texts)}")

    # Build search configs
    if args.strategy == "grid":
        hp_configs = _grid_configs(GRID_SEARCH_SPACE)
        logging.info(f"Grid search: {len(hp_configs)} configurations")
    else:
        hp_configs = _sample_random_configs(RANDOM_SEARCH_SPACE, args.n_trials, seed=args.seed)
        logging.info(f"Random search: {len(hp_configs)} configurations sampled")

    # ---- Run trials ----
    all_results: List[Dict] = []
    best_mean_f1 = 0.0
    best_trial: Optional[Dict] = None

    total_start = time.time()

    for i, hp in enumerate(hp_configs):
        trial_start = time.time()
        logging.info(
            f"\n{'='*60}\n"
            f"Trial {i+1}/{len(hp_configs)}\n"
            f"  lr={hp['learning_rate']:.1e}  bs={hp['batch_size']}  "
            f"dropout={hp['dropout_rate']}  wd={hp['weight_decay']}  "
            f"warmup={hp['warmup_ratio']}  epochs={hp['num_epochs']}  "
            f"sched={hp['lr_scheduler_type']}  maxlen={hp['max_length']}\n"
            f"{'='*60}"
        )

        result = run_trial(
            trial_idx=i,
            hp=hp,
            texts=texts,
            labels=labels,
            n_folds=args.n_folds,
            device=device,
            verbose=args.verbose,
        )

        trial_time = time.time() - trial_start
        result["trial_time_seconds"] = trial_time

        all_results.append(result)

        logging.info(
            f"  → Mean Val F1: {result['mean_val_f1']:.4f} ± {result['std_val_f1']:.4f}  |  "
            f"Acc: {result['mean_val_accuracy']:.4f} ± {result['std_val_accuracy']:.4f}  |  "
            f"AUC: {result['mean_val_roc_auc']:.4f}  |  {trial_time:.1f}s"
        )

        if result["mean_val_f1"] > best_mean_f1:
            best_mean_f1 = result["mean_val_f1"]
            best_trial = result
            logging.info(f"  *** New best configuration! (Mean F1 = {best_mean_f1:.4f}) ***")

        # Incremental save after each trial
        _save_search_results(all_results, best_trial, args, total_start)

    total_time = time.time() - total_start

    # ---- Final summary ----
    _print_summary(all_results, best_trial, total_time)
    _save_search_results(all_results, best_trial, args, total_start, final=True)

    # ---- Retrain best ----
    if args.retrain_best and best_trial is not None:
        logging.info("\nRetraining best configuration on full train+val set...")
        retrain_best(best_trial["hyperparams"], args.data, args.output, device)


def _save_search_results(
    all_results: List[Dict],
    best_trial: Optional[Dict],
    args,
    total_start: float,
    final: bool = False,
):
    """Save search results to JSON (incrementally after each trial)."""
    # Strip per-fold val_metrics detail to keep the file readable
    slim_results = []
    for r in all_results:
        slim = {k: v for k, v in r.items() if k != "per_fold"}
        slim["per_fold_f1s"] = [f["best_val_f1"] for f in r["per_fold"]]
        slim_results.append(slim)

    output = {
        "search_strategy": args.strategy,
        "n_folds":         args.n_folds,
        "n_trials_run":    len(all_results),
        "total_time_seconds": time.time() - total_start,
        "best_trial":      {
            "trial":       best_trial["trial"],
            "hyperparams": best_trial["hyperparams"],
            "mean_val_f1": best_trial["mean_val_f1"],
            "std_val_f1":  best_trial["std_val_f1"],
            "mean_val_accuracy": best_trial["mean_val_accuracy"],
            "mean_val_roc_auc":  best_trial["mean_val_roc_auc"],
        } if best_trial else None,
        "all_trials": slim_results,
    }

    suffix = "_final" if final else ""
    path = os.path.join(args.output, f"hp_search_results{suffix}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    if final:
        logging.info(f"Final results saved to {path}")

        # Also save the ranked leaderboard as a readable text file
        ranked = sorted(slim_results, key=lambda x: x["mean_val_f1"], reverse=True)
        lb_path = os.path.join(args.output, "leaderboard.txt")
        with open(lb_path, "w") as f:
            f.write(f"{'Rank':<5} {'Mean F1':<10} {'± Std':<10} {'Acc':<10} "
                    f"{'AUC':<10} {'LR':<10} {'BS':<5} {'Drop':<6} "
                    f"{'WD':<8} {'Warm':<6} {'Ep':<4} {'Sched':<8} {'MaxL':<5} {'Time':<8}\n")
            f.write("-" * 110 + "\n")
            for rank, r in enumerate(ranked, 1):
                hp = r["hyperparams"]
                f.write(
                    f"{rank:<5} {r['mean_val_f1']:<10.4f} {r['std_val_f1']:<10.4f} "
                    f"{r['mean_val_accuracy']:<10.4f} {r['mean_val_roc_auc']:<10.4f} "
                    f"{hp['learning_rate']:<10.1e} {hp['batch_size']:<5} "
                    f"{hp['dropout_rate']:<6} {hp['weight_decay']:<8} "
                    f"{hp['warmup_ratio']:<6} {hp['num_epochs']:<4} "
                    f"{hp['lr_scheduler_type']:<8} {hp['max_length']:<5} "
                    f"{r.get('trial_time_seconds', 0):<8.1f}\n"
                )
        logging.info(f"Leaderboard saved to {lb_path}")


def _print_summary(all_results: List[Dict], best_trial: Optional[Dict], total_time: float):
    """Print a concise summary of the search."""
    ranked = sorted(all_results, key=lambda x: x["mean_val_f1"], reverse=True)

    print("\n" + "=" * 70)
    print("  HYPERPARAMETER SEARCH — RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total trials: {len(all_results)}  |  Total time: {total_time:.1f}s")
    print()

    if best_trial:
        hp = best_trial["hyperparams"]
        print("  BEST CONFIGURATION:")
        print(f"    Mean Val F1:       {best_trial['mean_val_f1']:.4f} ± {best_trial['std_val_f1']:.4f}")
        print(f"    Mean Val Accuracy: {best_trial['mean_val_accuracy']:.4f}")
        print(f"    Mean Val ROC AUC:  {best_trial['mean_val_roc_auc']:.4f}")
        print()
        print("    Hyperparameters:")
        for k, v in hp.items():
            print(f"      {k:25s} = {v}")

    print()
    print("  TOP-5 CONFIGURATIONS:")
    print(f"  {'Rank':<5} {'Mean F1':<10} {'± Std':<10} {'LR':<10} {'BS':<5} "
          f"{'Drop':<6} {'WD':<8} {'Sched':<8}")
    print("  " + "-" * 65)
    for rank, r in enumerate(ranked[:5], 1):
        hp = r["hyperparams"]
        print(f"  {rank:<5} {r['mean_val_f1']:<10.4f} {r['std_val_f1']:<10.4f} "
              f"{hp['learning_rate']:<10.1e} {hp['batch_size']:<5} "
              f"{hp['dropout_rate']:<6} {hp['weight_decay']:<8} "
              f"{hp['lr_scheduler_type']:<8}")

    print("=" * 70)


if __name__ == "__main__":
    main()
