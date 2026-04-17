"""Dataset handling for the hierarchical headline classifier.

Provides separate loaders for:
  - Stage 1: topic classification (health=0, technology=1)
  - Stage 2: persuasion-route classification, filtered to a single topic
"""

import json
import logging
from collections import Counter
from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Core PyTorch Dataset
# ---------------------------------------------------------------------------

class HeadlineDataset(Dataset):
    """Generic PyTorch Dataset for headline classification.

    Each sample has a text string and a single integer class label.
    """

    def __init__(self, texts: List[str], labels: List[int],
                 tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": label,
        }


# ---------------------------------------------------------------------------
# Stage 1 — topic data loading
# ---------------------------------------------------------------------------

def load_topic_data(filepath: str, config: Any) -> Tuple[List[str], List[int]]:
    """Load JSON dataset and return (texts, topic_labels).

    topic_labels: health=0, technology=1
    """
    logging.info(f"Loading topic data from {filepath}")

    with open(filepath, "r") as f:
        data_list = json.load(f)

    texts: List[str] = []
    labels: List[int] = []

    for item in data_list:
        texts.append(item["text"])
        topic = item["topic"].lower()
        labels.append(config.topic_label_map[topic])

    logging.info(f"Loaded {len(texts)} samples  —  "
                 f"topic distribution: {_class_counts(labels, config.label_names)}")
    return texts, labels


def create_topic_datasets(filepath: str, config: Any,
                          test_filepath: Optional[str] = None):
    """Create train / val / test datasets for Stage 1 topic classification."""
    from sklearn.model_selection import train_test_split

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    texts, labels = load_topic_data(filepath, config)

    if test_filepath is None:
        # First split: train+val vs test
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels,
            test_size=config.test_split,
            random_state=42,
            stratify=labels,
        )

        # Second split: train vs val
        val_size = config.val_split / (config.train_split + config.val_split)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels,
            test_size=val_size,
            random_state=42,
            stratify=train_val_labels,
        )
    else:
        test_texts, test_labels = load_topic_data(test_filepath, config)

        val_size = config.val_split / (config.train_split + config.val_split)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            test_size=val_size,
            random_state=42,
            stratify=labels,
        )

    logging.info(f"Topic dataset sizes  —  train: {len(train_texts)}, "
                 f"val: {len(val_texts)}, test: {len(test_texts)}")

    train_dataset = HeadlineDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset   = HeadlineDataset(val_texts, val_labels, tokenizer, config.max_length)
    test_dataset  = HeadlineDataset(test_texts, test_labels, tokenizer, config.max_length)

    return train_dataset, val_dataset, test_dataset, tokenizer


# ---------------------------------------------------------------------------
# Stage 2 — route data loading (filtered by topic)
# ---------------------------------------------------------------------------

def load_route_data(filepath: str, config: Any, topic: str) -> Tuple[List[str], List[int]]:
    """Load JSON dataset, filter to a specific topic, return (texts, route_labels).

    route_labels: central=0, peripheral=1, neutral=2
    """
    logging.info(f"Loading route data from {filepath} (topic={topic})")

    with open(filepath, "r") as f:
        data_list = json.load(f)

    texts: List[str] = []
    labels: List[int] = []

    for item in data_list:
        if item["topic"].lower() != topic.lower():
            continue

        texts.append(item["text"])

        # Determine which route is active (exactly one should be 1)
        label = None
        for idx, label_name in enumerate(config.label_names):
            field_name = config.field_mapping[label_name]
            if item[field_name] == 1:
                label = idx
                break

        if label is None:
            raise ValueError(
                f"No active label found for item id={item.get('id', '?')}"
            )
        labels.append(label)

    logging.info(f"Loaded {len(texts)} {topic} samples  —  "
                 f"route distribution: {_class_counts(labels, config.label_names)}")
    return texts, labels


def create_route_datasets(filepath: str, config: Any, topic: str,
                          test_filepath: Optional[str] = None):
    """Create train / val / test datasets for Stage 2 route classification.

    Only headlines matching the specified topic are included.
    """
    from sklearn.model_selection import train_test_split

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    texts, labels = load_route_data(filepath, config, topic)

    if test_filepath is None:
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels,
            test_size=config.test_split,
            random_state=42,
            stratify=labels,
        )

        val_size = config.val_split / (config.train_split + config.val_split)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels,
            test_size=val_size,
            random_state=42,
            stratify=train_val_labels,
        )
    else:
        test_texts, test_labels = load_route_data(test_filepath, config, topic)

        val_size = config.val_split / (config.train_split + config.val_split)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            test_size=val_size,
            random_state=42,
            stratify=labels,
        )

    logging.info(f"Route dataset ({topic}) sizes  —  train: {len(train_texts)}, "
                 f"val: {len(val_texts)}, test: {len(test_texts)}")

    train_dataset = HeadlineDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset   = HeadlineDataset(val_texts, val_labels, tokenizer, config.max_length)
    test_dataset  = HeadlineDataset(test_texts, test_labels, tokenizer, config.max_length)

    return train_dataset, val_dataset, test_dataset, tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_counts(labels: List[int], label_names: List[str]) -> dict:
    counts = Counter(labels)
    return {label_names[k]: v for k, v in sorted(counts.items())}
