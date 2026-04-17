import json
import torch
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import sys
sys.path.append('src')

from config import Stage1Config, Stage2Config
from dataset import HeadlineDataset
from model import build_model

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

def load_trained_model(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tokenizer = checkpoint["tokenizer"]
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, config, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Generate predictions markdown file")
    parser.add_argument("--data", required=True, help="Path to JSON test data")
    parser.add_argument("--stage1-model", default="results/stage1/best_model.pt", help="Path to stage1 best_model.pt")
    parser.add_argument("--stage2-health-model", default="results/stage2_health/best_model.pt", help="Path to stage2 health best_model.pt")
    parser.add_argument("--stage2-tech-model", default="results/stage2_technology/best_model.pt", help="Path to stage2 technology best_model.pt")
    parser.add_argument("--output", required=True, help="Output markdown file path")
    args = parser.parse_args()

    stage1_model, stage1_config, stage1_tokenizer = load_trained_model(args.stage1_model)
    stage2_health_model, stage2_health_config, stage2_health_tokenizer = load_trained_model(args.stage2_health_model)
    stage2_tech_model, stage2_tech_config, stage2_tech_tokenizer = load_trained_model(args.stage2_tech_model)

    with open(args.data, 'r') as f:
        data_list = json.load(f)

    texts = [item["text"] for item in data_list]

    topic_dataset = HeadlineDataset(texts, [0]*len(texts), stage1_tokenizer, stage1_config.max_length)
    topic_loader = DataLoader(topic_dataset, batch_size=16, shuffle=False)

    topic_preds = []
    with torch.no_grad():
        for batch in topic_loader:
            outputs = stage1_model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            topic_preds.extend(preds)

    route_preds = [-1] * len(texts)
    stage2_config_obj = Stage2Config()
    labels_map = stage2_config_obj.label_names

    for i, text in enumerate(texts):
        topic = topic_preds[i]
        if topic == 0:
            model = stage2_health_model
            tokenizer = stage2_health_tokenizer
            config = stage2_health_config
        else:
            model = stage2_tech_model
            tokenizer = stage2_tech_tokenizer
            config = stage2_tech_config
        
        encoded = tokenizer(text, padding="max_length", truncation=True, max_length=config.max_length, return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device))
            route_preds[i] = out["logits"].argmax(dim=-1).item()

    true_routes = []
    for item in data_list:
        route_label = -1
        for idx, label_name in enumerate(stage2_config_obj.label_names):
            field_name = stage2_config_obj.field_mapping[label_name]
            if item.get(field_name) == 1:
                route_label = idx
                break
        true_routes.append(route_label)

    md_content = "# Headline Predictions\n\n"
    md_content += "| ID | Headline | True Topic | Pred Topic | True Route | Pred Route | Match |\n"
    md_content += "|---|---|---|---|---|---|---|\n"

    for i in range(len(texts)):
        true_l = labels_map[true_routes[i]] if true_routes[i] != -1 else "Unknown"
        pred_l = labels_map[route_preds[i]]
        true_topic = data_list[i].get("topic", "Unknown")
        pred_topic = "health" if topic_preds[i] == 0 else "technology"
        match = "✅" if true_l == pred_l and true_topic == pred_topic else "❌"
        
        row = f"| {i+1} | {texts[i]} | {true_topic} | {pred_topic} | {true_l} | {pred_l} | {match} |\n"
        md_content += row

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(md_content)
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    main()
