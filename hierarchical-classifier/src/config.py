"""Configuration dataclasses for the hierarchical headline classifier.

Stage 1: Binary topic classifier  (health vs technology)
Stage 2: 3-class route classifier  (central, peripheral, neutral)  — one per topic
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Stage1Config:
    """Config for Stage 1 — topic classification (health vs technology)."""

    # Model settings
    model_name: str = "bert-base-uncased"
    num_labels: int = 2
    max_length: int = 128

    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 20
    weight_decay: float = 0.01
    dropout_rate: float = 0.1
    warmup_ratio: float = 0.1

    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.1

    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Paths
    results_dir: str = "results/stage1"
    model_save_path: str = "results/stage1/best_model"

    # Labels
    label_names: List[str] = field(default_factory=lambda: [
        "health",
        "technology",
    ])

    # Mapping from label name to JSON field value in the "topic" key
    topic_label_map: dict = field(default_factory=lambda: {
        "health": 0,
        "technology": 1,
    })

    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)


@dataclass
class Stage2Config:
    """Config for Stage 2 — persuasion-route classification (per topic)."""

    # Model settings
    model_name: str = "bert-base-uncased"
    num_labels: int = 3  # central, peripheral, neutral
    max_length: int = 128

    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 20
    weight_decay: float = 0.01
    dropout_rate: float = 0.1
    warmup_ratio: float = 0.1

    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.1

    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Paths (will be overridden per topic)
    results_dir: str = "results/stage2"
    model_save_path: str = "results/stage2/best_model"

    # The topic this stage-2 model is specialised for
    topic: str = ""

    # Labels — order matches the feature columns in the dataset
    label_names: List[str] = field(default_factory=lambda: [
        "central_route",       # framework1_feature1
        "peripheral_route",    # framework1_feature2
        "neutral_route",       # framework1_feature3
    ])

    # Mapping from label names to JSON field names
    field_mapping: dict = field(default_factory=lambda: {
        "central_route":    "framework1_feature1",
        "peripheral_route": "framework1_feature2",
        "neutral_route":    "framework1_feature3",
    })

    def __post_init__(self):
        if self.topic:
            self.results_dir = f"results/stage2_{self.topic}"
            self.model_save_path = f"results/stage2_{self.topic}/best_model"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
