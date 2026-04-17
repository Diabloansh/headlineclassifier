# Hierarchical Headline Classifier

A two-stage BERT-based classification pipeline for headline analysis:

1. **Stage 1 — Topic Classification**: Classifies headlines as **health** or **technology**
2. **Stage 2 — Persuasion Route Classification**: For each topic, classifies into **central route**, **peripheral route**, or **neutral route**

## Architecture

```
Input Headline
    │
    ▼
┌─────────────────────────────────┐
│  Stage 1: Topic Classifier      │
│  (BERT → health / technology)   │
└──────────┬──────────┬───────────┘
           │          │
       health     technology
           │          │
           ▼          ▼
┌──────────────┐  ┌──────────────┐
│  Stage 2a    │  │  Stage 2b    │
│  Route (H)   │  │  Route (T)   │
│  central     │  │  central     │
│  peripheral  │  │  peripheral  │
│  neutral     │  │  neutral     │
└──────────────┘  └──────────────┘
```

## Quick Start

### 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Full Pipeline

```bash
python scripts/train_all.py --data ../Dataset/BERT_training_3000_v2.json
```

This trains all three models sequentially and runs hierarchical evaluation.

### 3. Train Individual Stages

```bash
# Stage 1: Topic classifier
python src/train_stage1.py --data_path ../Dataset/BERT_training_3000_v2.json

# Stage 2: Route classifiers (one per topic)
python src/train_stage2.py --data_path ../Dataset/BERT_training_3000_v2.json --topic health
python src/train_stage2.py --data_path ../Dataset/BERT_training_3000_v2.json --topic technology
```

### 4. Evaluate the Pipeline

```bash
python scripts/evaluate_pipeline.py \
    --data ../Dataset/BERT_training_3000_v2.json \
    --stage1-model results/stage1/best_model.pt \
    --stage2-health-model results/stage2_health/best_model.pt \
    --stage2-tech-model results/stage2_technology/best_model.pt
```

### 5. Predict on a New Headline

```bash
python src/predict.py \
    --stage1_model results/stage1/best_model.pt \
    --stage2_health_model results/stage2_health/best_model.pt \
    --stage2_tech_model results/stage2_technology/best_model.pt \
    --text "New study links daily exercise to 30% lower cancer risk"
```

## Training Options

| Argument       | Default | Description                        |
|---------------|---------|-------------------------------------|
| `--data`      | —       | Path to JSON dataset (required)     |
| `--test-data` | None    | Separate test JSON (optional)       |
| `--epochs`    | 20      | Number of training epochs           |
| `--batch-size`| 16      | Batch size                          |
| `--patience`  | 4       | Early stopping patience (epochs)    |
| `--lr`        | 2e-5    | Learning rate                       |

## Project Structure

```
hierarchical-classifier/
├── requirements.txt
├── README.md
├── src/
│   ├── config.py           # Stage1Config + Stage2Config
│   ├── dataset.py          # Topic + Route data loaders
│   ├── model.py            # HeadlineClassifier (BERT)
│   ├── train_stage1.py     # Train topic classifier
│   ├── train_stage2.py     # Train route classifier (per topic)
│   ├── evaluate.py         # Hierarchical evaluation
│   ├── predict.py          # End-to-end prediction
│   └── utils.py            # Metrics, seeding, logging
├── scripts/
│   ├── train_all.py        # Full pipeline orchestrator
│   └── evaluate_pipeline.py
└── Dataset/                # Symlink or copy of dataset
```

## Dataset Format

The JSON dataset should have this structure per record:

```json
{
    "text": "Headline text here",
    "framework1_feature1": 0,  // central route (1=yes, 0=no)
    "framework1_feature2": 1,  // peripheral route
    "framework1_feature3": 0,  // neutral route
    "topic": "health",         // or "technology"
    "id": 1
}
```
