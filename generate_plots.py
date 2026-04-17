import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    precision_recall_curve, average_precision_score, PrecisionRecallDisplay,
    roc_curve, auc, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
)

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import Config
from dataset import create_datasets
from model import build_model
from utils import set_seed

def plot_learning_curves(log_path, output_dir):
    with open(log_path, 'r') as f:
        log_data = json.load(f)
        
    epochs = []
    train_loss, val_loss = [], []
    train_f1, val_f1 = [], []
    train_acc, val_acc = [], []
    
    for ep in log_data["epochs"]:
        epochs.append(ep["epoch"])
        train_loss.append(ep["train_loss"])
        val_loss.append(ep.get("val_loss", ep["val_metrics"]["loss"]))
        train_f1.append(ep["train_metrics"]["macro_f1"])
        val_f1.append(ep["val_metrics"]["macro_f1"])
        train_acc.append(ep["train_metrics"]["accuracy"])
        val_acc.append(ep["val_metrics"]["accuracy"])
        
    # --- Plot Loss ---
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'fig_01_loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Plot F1 Score Metrics ---
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_f1, 'bo-', label='Train Macro F1')
    plt.plot(epochs, val_f1, 'ro-', label='Val Macro F1')
    plt.plot(epochs, train_acc, 'b--', alpha=0.5, label='Train Accuracy')
    plt.plot(epochs, val_acc, 'r--', alpha=0.5, label='Val Accuracy')
    plt.title('Training & Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'fig_02_metrics_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_evaluation_plots(results_dir, model_name="best_model.pt", data_path="Dataset/Curated_3000.json"):
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = os.path.join(results_dir, model_name)
    
    # Load config from checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    
    # Set HF env vars explicitly inside code as fallback
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Get test dataloader
    _, _, test_dataset, _ = create_datasets(data_path, config)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Build model & load weights
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    
    y_true = []
    y_score = []
    
    print("Running inference on test dataset...")
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
    y_pred = y_score.argmax(axis=-1)
    n_classes = config.num_labels
    class_names = config.label_names
    
    Y_bin = np.zeros((y_true.size, n_classes))
    Y_bin[np.arange(y_true.size), y_true] = 1
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # --- 3. Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title("Test Set Confusion Matrix")
    plt.savefig(os.path.join(results_dir, 'fig_03_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 4. ROC Curves ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(Y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"{class_names[i]}")
        display.plot(ax=ax, color=color)
    
    # Plot chance level
    ax.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.5)')
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'fig_04_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 5. AUPRC Curves ---
    fig, ax = plt.subplots(figsize=(8, 6))
    auprc_scores = []
    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay.from_predictions(
            Y_bin[:, i], y_score[:, i], name=f"{class_names[i]}", color=color, ax=ax
        )
        auprc_scores.append(average_precision_score(Y_bin[:, i], y_score[:, i]))
        
    macro_auprc = np.mean(auprc_scores)
    ax.set_title(f"Precision-Recall Curve (Macro AUPRC: {macro_auprc:.4f})")
    ax.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'fig_05_auprc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All plots generated successfully!")


def main():
    results_dir = "results_20260407_140736"
    log_path = os.path.join(results_dir, "training_log.json")
    
    print("Generating learning curves...")
    plot_learning_curves(log_path, results_dir)
    
    print("Generating evaluation curves...")
    generate_evaluation_plots(results_dir)

if __name__ == "__main__":
    main()
