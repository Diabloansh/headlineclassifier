"""Standalone evaluation script for headline persuasion-route classifier."""

import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import load_data, HeadlineDataset
from model import build_model
from utils import (
    setup_logging,
    compute_metrics,
    print_classification_report,
    save_metrics,
)


def load_trained_model(model_path: str, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config    = checkpoint["config"]
    tokenizer = checkpoint["tokenizer"]

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model, config, tokenizer


def evaluate(model, dataloader, device, config):
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            labels          = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs["logits"], dim=-1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    return probs, labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate headline classifier")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--data_path",  type=str, required=True, help="Path to JSON test data")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    setup_logging()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Load model
    model, config, tokenizer = load_trained_model(args.model_path, device)

    # Load data
    texts, labels = load_data(args.data_path, config)
    dataset = HeadlineDataset(texts, labels, tokenizer, config.max_length)
    loader  = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Evaluate
    probs, true_labels = evaluate(model, loader, device, config)
    metrics = compute_metrics(probs, true_labels, config.label_names)

    # Print
    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS — Headline Persuasion Classifier")
    print("=" * 55)
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Macro F1:   {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:{metrics['weighted_f1']:.4f}")
    print(f"  ROC AUC:    {metrics['roc_auc']:.4f}")
    print()
    print_classification_report(probs, true_labels, config.label_names)

    # Confusion matrix
    print("Confusion matrix (rows=true, cols=pred):")
    for row in metrics["confusion_matrix"]:
        print("  ", row)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_metrics(metrics, os.path.join(args.output_dir, "evaluation_metrics.json"))
    logging.info(f"Metrics saved to {args.output_dir}/evaluation_metrics.json")


if __name__ == "__main__":
    main()
