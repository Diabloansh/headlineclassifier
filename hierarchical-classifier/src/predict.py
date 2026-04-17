"""End-to-end hierarchical prediction for new headline text.

Usage:
    python src/predict.py \
        --stage1_model results/stage1/best_model.pt \
        --stage2_health_model results/stage2_health/best_model.pt \
        --stage2_tech_model results/stage2_technology/best_model.pt \
        --text "New study links daily exercise to 30% lower cancer risk"
"""

import argparse
import torch
from model import build_model


def load_model(model_path: str, device):
    """Load a trained model checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config    = checkpoint["config"]
    tokenizer = checkpoint["tokenizer"]

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, tokenizer


def predict_single(text: str, model, tokenizer, config, device):
    """Run inference on a single text and return probabilities."""
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=config.max_length,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask  = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs["logits"], dim=-1).squeeze()

    return probs


def main():
    parser = argparse.ArgumentParser(description="Hierarchical headline prediction")
    parser.add_argument("--stage1_model",         type=str, required=True,
                        help="Path to stage1 best_model.pt")
    parser.add_argument("--stage2_health_model",  type=str, required=True,
                        help="Path to stage2 health best_model.pt")
    parser.add_argument("--stage2_tech_model",    type=str, required=True,
                        help="Path to stage2 technology best_model.pt")
    parser.add_argument("--text", type=str, required=True, help="Headline text to classify")
    args = parser.parse_args()

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # ---- Load models ----
    stage1_model, stage1_config, stage1_tokenizer = load_model(args.stage1_model, device)
    stage2_health_model, stage2_health_config, stage2_health_tokenizer = load_model(
        args.stage2_health_model, device)
    stage2_tech_model, stage2_tech_config, stage2_tech_tokenizer = load_model(
        args.stage2_tech_model, device)

    # ---- Stage 1: topic prediction ----
    topic_probs = predict_single(args.text, stage1_model, stage1_tokenizer, stage1_config, device)
    topic_idx   = topic_probs.argmax().item()
    topic_name  = stage1_config.label_names[topic_idx]
    topic_conf  = float(topic_probs[topic_idx])

    # ---- Stage 2: route prediction ----
    if topic_name == "health":
        route_probs = predict_single(args.text, stage2_health_model,
                                      stage2_health_tokenizer, stage2_health_config, device)
        route_config = stage2_health_config
    else:
        route_probs = predict_single(args.text, stage2_tech_model,
                                      stage2_tech_tokenizer, stage2_tech_config, device)
        route_config = stage2_tech_config

    route_idx  = route_probs.argmax().item()
    route_name = route_config.label_names[route_idx]
    route_conf = float(route_probs[route_idx])

    # ---- Display results ----
    print(f"\n{'=' * 60}")
    print(f"  Headline: {args.text}")
    print(f"{'=' * 60}")
    print(f"\n  Stage 1 — Topic:")
    print(f"    Predicted:  {topic_name}  (confidence: {topic_conf:.4f})")
    print(f"    Probabilities:")
    for i, name in enumerate(stage1_config.label_names):
        print(f"      {name:15s}  {float(topic_probs[i]):.4f}")

    print(f"\n  Stage 2 — Persuasion Route ({topic_name}):")
    print(f"    Predicted:  {route_name}  (confidence: {route_conf:.4f})")
    print(f"    Probabilities:")
    for i, name in enumerate(route_config.label_names):
        print(f"      {name:20s}  {float(route_probs[i]):.4f}")

    print(f"\n  Final Classification: {topic_name} / {route_name}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
