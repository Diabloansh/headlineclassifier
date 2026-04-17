"""Training script for Stage 2 — persuasion-route classification (per topic).

Usage:
    python src/train_stage2.py --data_path Dataset/BERT_training_3000_v2.json --topic health
    python src/train_stage2.py --data_path Dataset/BERT_training_3000_v2.json --topic technology
"""

import argparse
import json
import logging
import os
from datetime import datetime
from collections import Counter

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from tqdm import tqdm

from config import Stage2Config
from dataset import create_route_datasets
from model import build_model
from utils import set_seed, setup_logging, compute_metrics, save_metrics


# --------------------------------------------------------------------------
# One training epoch
# --------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scheduler, device, topic):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Training [Stage 2 — {topic}]"):
        optimizer.zero_grad()

        input_ids      = batch["input_ids"].to(device)
        attention_mask  = batch["attention_mask"].to(device)
        labels          = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# --------------------------------------------------------------------------
# Evaluation helper
# --------------------------------------------------------------------------

def evaluate_model(model, dataloader, device, config, topic):
    model.eval()
    all_probs  = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating [Stage 2 — {topic}]"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            labels          = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs["loss"].item()

            probs = torch.softmax(outputs["logits"], dim=-1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)

    metrics = compute_metrics(probs, labels, config.label_names)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 — route classifier for one topic")
    parser.add_argument("--data_path",  type=str, required=True, help="Path to JSON dataset")
    parser.add_argument("--topic",      type=str, required=True, choices=["health", "technology"],
                        help="Topic to train route classifier for")
    parser.add_argument("--test_path",  type=str, default=None,  help="Separate test JSON (optional)")
    parser.add_argument("--output_dir", type=str, default=None,  help="Output directory")
    parser.add_argument("--patience",   type=int, default=4,     help="Early stopping patience")
    args = parser.parse_args()

    # ---- setup ----
    setup_logging()
    set_seed(42)

    config = Stage2Config(topic=args.topic)
    if args.output_dir:
        config.results_dir     = args.output_dir
        config.model_save_path = os.path.join(args.output_dir, "best_model")

    # Device selection (MPS → CUDA → CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # ---- datasets (filtered to topic) ----
    train_dataset, val_dataset, test_dataset, tokenizer = create_route_datasets(
        args.data_path, config, args.topic, args.test_path,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False)

    # ---- class weights (inverse frequency) ----
    label_counts = Counter(train_dataset.labels)
    total = len(train_dataset.labels)
    class_weights = torch.tensor(
        [total / (config.num_labels * label_counts[i]) for i in range(config.num_labels)],
        dtype=torch.float32,
    )
    logging.info(f"Class counts (train): {dict(sorted(label_counts.items()))}")
    logging.info(f"Class weights: {class_weights.tolist()}")

    # ---- model ----
    model = build_model(config, class_weights=class_weights).to(device)

    # ---- optimizer & scheduler ----
    optimizer   = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    if config.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                     num_training_steps=total_steps)
        logging.info(f"Using cosine LR scheduler with {warmup_steps} warmup steps")
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                     num_training_steps=total_steps)
        logging.info(f"Using linear LR scheduler with {warmup_steps} warmup steps")



    # ---- training log ----
    training_log = {
        "stage": f"stage2_{args.topic}",
        "topic": args.topic,
        "start_time": datetime.now().isoformat(),
        "config": {
            "model_name":    config.model_name,
            "num_labels":    config.num_labels,
            "num_epochs":    config.num_epochs,
            "batch_size":    config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay":  config.weight_decay,
            "max_length":    config.max_length,
            "warmup_ratio":  config.warmup_ratio,
            "lr_scheduler":  config.lr_scheduler_type,
            "dropout_rate":  config.dropout_rate,
            "early_stopping_patience": args.patience,
        },
        "epochs": [],
        "device": str(device),
        "dataset_sizes": {
            "train": len(train_dataset),
            "val":   len(val_dataset),
            "test":  len(test_dataset),
        },
    }

    # ================================================================
    # Training loop
    # ================================================================
    best_val_f1        = 0.0
    epochs_no_improve  = 0
    early_stop_patience = args.patience

    print("=" * 60)
    print(f"  Stage 2 — Route Classifier ({args.topic})")
    print(f"  Labels: central / peripheral / neutral")
    print("=" * 60)
    print(f"  Dataset:    {args.data_path} (filtered to {args.topic})")
    print(f"  Output dir: {config.results_dir}")
    print(f"  Patience:   {early_stop_patience} epochs")
    print("=" * 60)

    logging.info(f"Early stopping patience: {early_stop_patience} epochs")

    for epoch in range(config.num_epochs):
        epoch_start = datetime.now()
        logging.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args.topic)
        logging.info(f"  Train loss: {train_loss:.4f}")

        # Validate
        val_metrics   = evaluate_model(model, val_loader,   device, config, args.topic)
        train_metrics = evaluate_model(model, train_loader, device, config, args.topic)

        epoch_duration = (datetime.now() - epoch_start).total_seconds()

        is_best = val_metrics["macro_f1"] > best_val_f1

        logging.info(f"  Train F1: {train_metrics['macro_f1']:.4f}  |  Train Acc: {train_metrics['accuracy']:.4f}")
        logging.info(f"  Val   F1: {val_metrics['macro_f1']:.4f}  |  Val   Acc: {val_metrics['accuracy']:.4f}")
        logging.info(f"  Val  loss: {val_metrics['loss']:.4f}  |  LR: {optimizer.param_groups[0]['lr']:.2e}  |  {epoch_duration:.1f}s")


        # Save best model
        if is_best:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config":           config,
                "tokenizer":        tokenizer,
                "epoch":            epoch + 1,
                "val_metrics":      val_metrics,
            }, config.model_save_path + ".pt")
            logging.info(f"  *** New best model saved  (Val F1 = {best_val_f1:.4f}) ***")

        # Append epoch log
        epoch_log = {
            "epoch":       epoch + 1,
            "timestamp":   epoch_start.isoformat(),
            "duration_seconds": epoch_duration,
            "train_loss":  train_loss,
            "train_metrics": train_metrics,
            "val_metrics":   val_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "is_best":      is_best,
        }
        training_log["epochs"].append(epoch_log)

        # Incremental log save
        with open(os.path.join(config.results_dir, "training_log.json"), "w") as f:
            json.dump(training_log, f, indent=2, default=str)

        # ---- Early stopping check ----
        if is_best:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logging.info(f"  No improvement for {epochs_no_improve}/{early_stop_patience} epoch(s).")
            if epochs_no_improve >= early_stop_patience:
                logging.info(f"  *** Early stopping triggered after {epoch + 1} epochs (patience={early_stop_patience}) ***")
                logging.info("-" * 60)
                break

        logging.info("-" * 60)

    # ================================================================
    # Final test evaluation
    # ================================================================
    logging.info(f"\nEvaluating best model on test set ({args.topic})...")

    best_ckpt = torch.load(config.model_save_path + ".pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate_model(model, test_loader, device, config, args.topic)

    training_log["end_time"] = datetime.now().isoformat()
    training_log["total_duration_seconds"] = (
        datetime.now() - datetime.fromisoformat(training_log["start_time"])
    ).total_seconds()
    training_log["best_val_f1"]   = best_val_f1
    training_log["test_metrics"]  = test_metrics

    save_metrics(test_metrics, os.path.join(config.results_dir, "test_metrics.json"))

    final_log_path = os.path.join(config.results_dir, "training_log_final.json")
    with open(final_log_path, "w") as f:
        json.dump(training_log, f, indent=2, default=str)

    # Save summary
    actual_epochs_run = len(training_log["epochs"])
    stopped_early     = actual_epochs_run < config.num_epochs
    summary = {
        "training_summary": {
            "stage": f"stage2_{args.topic}",
            "topic": args.topic,
            "total_epochs_configured": config.num_epochs,
            "epochs_actually_run":     actual_epochs_run,
            "stopped_early":           stopped_early,
            "early_stopping_patience": early_stop_patience,
            "best_epoch":              max(training_log["epochs"],
                                          key=lambda e: e["val_metrics"]["macro_f1"])["epoch"],
            "best_val_f1":         best_val_f1,
            "final_test_f1":       test_metrics["macro_f1"],
            "final_test_accuracy": test_metrics["accuracy"],
            "total_training_time": training_log["total_duration_seconds"],
        },
        "per_epoch_summary": [
            {
                "epoch":     ep["epoch"],
                "train_loss": ep["train_loss"],
                "train_f1":  ep["train_metrics"]["macro_f1"],
                "train_acc": ep["train_metrics"]["accuracy"],
                "val_f1":    ep["val_metrics"]["macro_f1"],
                "val_acc":   ep["val_metrics"]["accuracy"],
                "val_loss":  ep["val_metrics"]["loss"],
                "is_best":   ep["is_best"],
                "duration":  ep["duration_seconds"],
            }
            for ep in training_log["epochs"]
        ],
    }
    with open(os.path.join(config.results_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("=" * 60)
    logging.info(f"Stage 2 ({args.topic}) training complete!")
    logging.info(f"  Best Val  F1: {best_val_f1:.4f}")
    logging.info(f"  Test      F1: {test_metrics['macro_f1']:.4f}")
    logging.info(f"  Test     Acc: {test_metrics['accuracy']:.4f}")
    logging.info(f"  ROC AUC:     {test_metrics['roc_auc']:.4f}")
    logging.info(f"  Total time:  {training_log['total_duration_seconds']:.1f}s")
    logging.info(f"  Results dir: {config.results_dir}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
