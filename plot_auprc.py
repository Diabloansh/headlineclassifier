import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import Config
from dataset import create_datasets
from model import build_model
from utils import set_seed

def main():
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Paths
    results_dir = "results_20260407_140736"
    model_path = os.path.join(results_dir, "best_model.pt")
    data_path = "Dataset/Curated_3000.json"
    
    # Load config from checkpoint
    print("Loading test dataset and best model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tokenizer = checkpoint["tokenizer"]
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")
    os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), ".matplotlib")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Get test dataloader
    _, _, test_dataset, _ = create_datasets(data_path, config)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Build model & load weights
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    
    # Predict
    print("Running inference...")
    y_true = []
    y_score = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs["logits"], dim=-1)
            
            y_true.extend(labels.cpu().numpy())
            y_score.extend(probs.cpu().numpy())
            
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    n_classes = config.num_labels
    
    # Convert labels to one-hot for PR calculation
    Y_bin = np.zeros((y_true.size, n_classes))
    Y_bin[np.arange(y_true.size), y_true] = 1
    
    # Prepare plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot per-class PR curves
    lines = []
    labels = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay.from_predictions(
            Y_bin[:, i],
            y_score[:, i],
            name=f"Class {config.label_names[i]}",
            color=color,
            ax=ax,
            plot_chance_level=(i == n_classes - 1)
        )
    
    # Calculate Macro AUPRC
    auprc_scores = [average_precision_score(Y_bin[:, i], y_score[:, i]) for i in range(n_classes)]
    macro_auprc = np.mean(auprc_scores)
    
    ax.set_title(f"Precision-Recall Curve (Macro AUPRC: {macro_auprc:.4f})")
    ax.legend(loc="best")
    plt.grid(alpha=0.3)
    
    # Save the figure
    out_img = os.path.join(results_dir, "auprc_curve.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Saved AUPRC curve to: {out_img}")
    print(f"Scores per class: {dict(zip(config.label_names, auprc_scores))}")

if __name__ == "__main__":
    main()
