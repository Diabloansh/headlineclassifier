# Headline Persuasion-Route Classifier

A **BERT-based multi-class classifier** that categorises misinformation headlines into one of three persuasion routes:

| Class | JSON Field | Description |
|---|---|---|
| **Central Route** | `framework1_feature1` | Uses logical/technical-sounding arguments and causal reasoning. |
| **Peripheral Route** | `framework1_feature2` | Relies on authority cues, emotional appeals, fear, and sensationalism. |
| **Neutral Route** | `framework1_feature3` | Bland factual-sounding assertions without strong mechanisms or emotional cues. |

## Project Structure

```
headline-classifier/
├── Dataset/
│   └── 1ansh.json              # 300 labelled headlines
├── src/
│   ├── config.py               # Hyperparameters & label definitions
│   ├── dataset.py              # Data loading & PyTorch Dataset
│   ├── model.py                # BERT classifier architecture
│   ├── train.py                # Full training loop with logging
│   ├── evaluate.py             # Standalone evaluation script
│   ├── predict.py              # Predict on new headlines
│   └── utils.py                # Metrics, seeding, I/O helpers
├── scripts/
│   ├── train_enhanced.py       # Convenience launcher
│   ├── visualize_training.py   # Plot training curves
│   ├── visualize_dataset.py    # Dataset distribution analysis
│   └── compare_baselines.py    # Random & class-prior baselines
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# From the headline-classifier/ directory:
cd src
python train.py --data_path ../Dataset/1ansh.json --output_dir ../results

# Or use the launcher script:
python scripts/train_enhanced.py --data Dataset/1ansh.json
```

## Evaluation

```bash
cd src
python evaluate.py --model_path ../results/best_model.pt --data_path ../Dataset/1ansh.json --output_dir ../results
```

## Prediction

```bash
cd src
python predict.py --model_path ../results/best_model.pt \
    --text "Top cardiologists from AIIMS warn everyone to stop drinking cold water immediately!"
```

## Visualisation

```bash
# Training curves & confusion matrix
python scripts/visualize_training.py --log results/training_log_final.json --output results/visualizations

# Dataset analysis
python scripts/visualize_dataset.py --data Dataset/1ansh.json --output results/dataset_analysis
```

## Baseline Comparison

```bash
python scripts/compare_baselines.py --model results/best_model.pt --data Dataset/1ansh.json --output results
```

## Key Design Decisions

| Aspect | Choice | Rationale |
|---|---|---|
| **Base model** | `bert-base-uncased` | Full BERT (110M params) for stronger representations on a small dataset |
| **Task type** | Multi-class (CrossEntropyLoss) | Labels are mutually exclusive — each headline has exactly one route |
| **Max length** | 128 tokens | Headlines are short; 128 is sufficient and faster than 256/512 |
| **Scheduler** | Cosine decay with warmup | Smooth LR reduction avoids sharp drops |
| **Class weights** | Inverse-frequency weighting | All 3 classes have 100 samples (balanced), but weights are computed dynamically |
| **Gradient clipping** | max_norm = 1.0 | Stabilises training with small datasets |

## Dataset

300 headlines (100 per class) covering **health** and **technology** topics, sourced from `1ansh.json`. Each headline is labelled with exactly one persuasion route based on the Elaboration Likelihood Model (ELM) framework.
