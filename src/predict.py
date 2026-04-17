"""Predict persuasion route for new headline text."""

import argparse
import torch
from transformers import AutoTokenizer

from config import Config
from model import build_model


def load_model(model_path: str, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config    = checkpoint["config"]
    tokenizer = checkpoint["tokenizer"]

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, tokenizer


def predict(text: str, model, tokenizer, config, device):
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

    predicted_class = probs.argmax().item()
    return {
        "text": text,
        "predicted_label": config.label_names[predicted_class],
        "confidence": float(probs[predicted_class]),
        "probabilities": {
            name: float(probs[i]) for i, name in enumerate(config.label_names)
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Predict persuasion route for a headline")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--text", type=str, required=True, help="Headline text to classify")
    args = parser.parse_args()

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model, config, tokenizer = load_model(args.model_path, device)
    result = predict(args.text, model, tokenizer, config, device)

    print(f"\nHeadline:  {result['text']}")
    print(f"Predicted: {result['predicted_label']}  (confidence: {result['confidence']:.4f})")
    print("Probabilities:")
    for name, prob in result["probabilities"].items():
        print(f"  {name:20s}  {prob:.4f}")


if __name__ == "__main__":
    main()
