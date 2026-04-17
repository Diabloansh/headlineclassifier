"""Utility functions for headline persuasion-route classification."""

import json
import logging
import random
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# --------------------------------------------------------------------------
# Metrics for multi-class classification
# --------------------------------------------------------------------------

def compute_metrics(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    label_names: List[str],
) -> Dict:
    """Compute comprehensive metrics for multi-class classification.

    Parameters
    ----------
    probabilities : ndarray of shape (N, C)
        Softmax probabilities for each class.
    true_labels : ndarray of shape (N,)
        Integer ground-truth class indices.
    label_names : list of str
        Human-readable names for each class.

    Returns
    -------
    dict  –  flat dictionary of all metrics.
    """
    pred_labels = probabilities.argmax(axis=1)

    # Overall accuracy
    accuracy = accuracy_score(true_labels, pred_labels)

    # Precision / recall / F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, zero_division=0,
    )

    # Macro & weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0,
    )
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    # ROC AUC (one-vs-rest)
    try:
        roc_auc = roc_auc_score(
            true_labels, probabilities, multi_class="ovr", average="macro",
        )
    except ValueError:
        roc_auc = 0.0

    # Confusion matrix (as nested list for JSON serialisability)
    cm = confusion_matrix(true_labels, pred_labels).tolist()

    metrics: Dict = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm,
    }

    # Per-class metrics
    for i, name in enumerate(label_names):
        metrics[f"{name}_precision"] = float(precision[i])
        metrics[f"{name}_recall"]    = float(recall[i])
        metrics[f"{name}_f1"]        = float(f1[i])
        metrics[f"{name}_support"]   = int(support[i])

    return metrics


def print_classification_report(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    label_names: List[str],
):
    """Pretty-print sklearn classification report."""
    pred_labels = probabilities.argmax(axis=1)
    print(classification_report(true_labels, pred_labels,
                                target_names=label_names, digits=4))


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def save_metrics(metrics: Dict, filepath: str):
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath: str) -> Dict:
    with open(filepath, "r") as f:
        return json.load(f)
