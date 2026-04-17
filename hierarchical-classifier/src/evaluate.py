"""Hierarchical evaluation: Stage 1 (topic) → Stage 2 (route).

Loads all three trained models (stage1, stage2_health, stage2_technology),
runs the full hierarchical pipeline on a test dataset, and computes both
per-stage and end-to-end metrics.
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import Stage1Config, Stage2Config
from dataset import HeadlineDataset, load_topic_data, load_route_data
from model import build_model
from utils import (
    setup_logging,
    compute_metrics,
    print_classification_report,
    save_metrics,
)


def load_trained_model(model_path: str, device):
    """Load a trained model checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config    = checkpoint["config"]
    tokenizer = checkpoint["tokenizer"]

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, config, tokenizer


def predict_batch(model, dataloader, device):
    """Run inference and return (probabilities, true_labels)."""
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

    return np.array(all_probs), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Hierarchical pipeline evaluation")
    parser.add_argument("--data_path",     type=str, required=True, help="Path to JSON test data")
    parser.add_argument("--stage1_model",  type=str, required=True, help="Path to stage1 best_model.pt")
    parser.add_argument("--stage2_health_model",  type=str, required=True,
                        help="Path to stage2 health best_model.pt")
    parser.add_argument("--stage2_tech_model",    type=str, required=True,
                        help="Path to stage2 technology best_model.pt")
    parser.add_argument("--output_dir",    type=str, default="results/hierarchical_eval",
                        help="Output directory for metrics")
    args = parser.parse_args()

    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # ---- Load all three models ----
    stage1_model, stage1_config, stage1_tokenizer = load_trained_model(args.stage1_model, device)
    stage2_health_model, stage2_health_config, stage2_health_tokenizer = load_trained_model(
        args.stage2_health_model, device)
    stage2_tech_model, stage2_tech_config, stage2_tech_tokenizer = load_trained_model(
        args.stage2_tech_model, device)

    # ---- Load full dataset ----
    with open(args.data_path, "r") as f:
        data_list = json.load(f)

    texts = [item["text"] for item in data_list]
    true_topics = []
    true_routes = []

    stage1_config_obj = Stage1Config()
    stage2_config_obj = Stage2Config()

    for item in data_list:
        true_topics.append(stage1_config_obj.topic_label_map[item["topic"].lower()])

        route_label = None
        for idx, label_name in enumerate(stage2_config_obj.label_names):
            field_name = stage2_config_obj.field_mapping[label_name]
            if item[field_name] == 1:
                route_label = idx
                break
        true_routes.append(route_label)

    true_topics = np.array(true_topics)
    true_routes = np.array(true_routes)

    # ---- Stage 1: Topic prediction ----
    topic_dataset = HeadlineDataset(texts, true_topics.tolist(), stage1_tokenizer,
                                     stage1_config.max_length)
    topic_loader  = DataLoader(topic_dataset, batch_size=16, shuffle=False)

    topic_probs, _ = predict_batch(stage1_model, topic_loader, device)
    pred_topics = topic_probs.argmax(axis=1)

    topic_metrics = compute_metrics(topic_probs, true_topics, stage1_config_obj.label_names)

    print("\n" + "=" * 60)
    print("  STAGE 1 — Topic Classification Results")
    print("=" * 60)
    print(f"  Accuracy:   {topic_metrics['accuracy']:.4f}")
    print(f"  Macro F1:   {topic_metrics['macro_f1']:.4f}")
    print_classification_report(topic_probs, true_topics, stage1_config_obj.label_names)

    save_metrics(topic_metrics, os.path.join(args.output_dir, "stage1_metrics.json"))

    # ---- Stage 2: Route prediction (hierarchical) ----
    pred_routes = np.full(len(texts), -1, dtype=int)
    route_probs_all = np.zeros((len(texts), 3))

    # Health branch
    health_mask = pred_topics == 0
    if health_mask.any():
        health_texts  = [texts[i] for i in range(len(texts)) if health_mask[i]]
        health_labels = [true_routes[i] for i in range(len(texts)) if health_mask[i]]
        health_dataset = HeadlineDataset(health_texts, health_labels,
                                          stage2_health_tokenizer, stage2_health_config.max_length)
        health_loader  = DataLoader(health_dataset, batch_size=16, shuffle=False)

        health_probs, _ = predict_batch(stage2_health_model, health_loader, device)
        health_preds = health_probs.argmax(axis=1)

        j = 0
        for i in range(len(texts)):
            if health_mask[i]:
                pred_routes[i] = health_preds[j]
                route_probs_all[i] = health_probs[j]
                j += 1

    # Technology branch
    tech_mask = pred_topics == 1
    if tech_mask.any():
        tech_texts  = [texts[i] for i in range(len(texts)) if tech_mask[i]]
        tech_labels = [true_routes[i] for i in range(len(texts)) if tech_mask[i]]
        tech_dataset = HeadlineDataset(tech_texts, tech_labels,
                                        stage2_tech_tokenizer, stage2_tech_config.max_length)
        tech_loader  = DataLoader(tech_dataset, batch_size=16, shuffle=False)

        tech_probs, _ = predict_batch(stage2_tech_model, tech_loader, device)
        tech_preds = tech_probs.argmax(axis=1)

        j = 0
        for i in range(len(texts)):
            if tech_mask[i]:
                pred_routes[i] = tech_preds[j]
                route_probs_all[i] = tech_probs[j]
                j += 1

    # ---- End-to-end route metrics ----
    route_metrics = compute_metrics(route_probs_all, true_routes, stage2_config_obj.label_names)

    print("\n" + "=" * 60)
    print("  STAGE 2 — Hierarchical Route Classification Results")
    print("=" * 60)
    print(f"  Accuracy:   {route_metrics['accuracy']:.4f}")
    print(f"  Macro F1:   {route_metrics['macro_f1']:.4f}")
    print_classification_report(route_probs_all, true_routes, stage2_config_obj.label_names)

    print("Confusion matrix (rows=true, cols=pred):")
    for row in route_metrics["confusion_matrix"]:
        print("  ", row)

    save_metrics(route_metrics, os.path.join(args.output_dir, "stage2_hierarchical_metrics.json"))

    # ---- Combined summary ----
    # Topic accuracy impact: how many headlines' routes were affected by misclassified topics
    topic_errors = (pred_topics != true_topics).sum()
    combined = {
        "stage1_topic_accuracy": float(topic_metrics["accuracy"]),
        "stage1_topic_f1":       float(topic_metrics["macro_f1"]),
        "stage2_route_accuracy": float(route_metrics["accuracy"]),
        "stage2_route_f1":       float(route_metrics["macro_f1"]),
        "stage2_route_roc_auc":  float(route_metrics["roc_auc"]),
        "topic_misclassifications": int(topic_errors),
        "total_samples":         len(texts),
    }
    save_metrics(combined, os.path.join(args.output_dir, "combined_metrics.json"))

    print("\n" + "=" * 60)
    print("  COMBINED PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Stage 1 topic Acc:   {combined['stage1_topic_accuracy']:.4f}")
    print(f"  Stage 1 topic F1:    {combined['stage1_topic_f1']:.4f}")
    print(f"  Stage 2 route Acc:   {combined['stage2_route_accuracy']:.4f}")
    print(f"  Stage 2 route F1:    {combined['stage2_route_f1']:.4f}")
    print(f"  Topic misclassified: {combined['topic_misclassifications']}/{combined['total_samples']}")
    print("=" * 60)

    logging.info(f"All metrics saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
