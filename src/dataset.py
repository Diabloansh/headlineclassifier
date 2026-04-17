"""Dataset handling for headline persuasion-route classification (multi-class)."""

import json
import logging
from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class HeadlineDataset(Dataset):
    """PyTorch Dataset for headline classification.

    Each sample is assigned a single integer class label (0, 1, or 2)
    derived from the mutually-exclusive feature columns.
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
# Data loading helpers
# ---------------------------------------------------------------------------

def load_data(filepath: str, config: Any) -> Tuple[List[str], List[int]]:
    """Load JSON dataset and return (texts, integer_labels).

    The integer label is the index of the feature column that equals 1.
    """
    logging.info(f"Loading data from {filepath}")

    with open(filepath, "r") as f:
        data_list = json.load(f)

    texts: List[str] = []
    labels: List[int] = []

    for item in data_list:
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

    logging.info(f"Loaded {len(texts)} samples  —  "
                 f"class distribution: {_class_counts(labels, config.label_names)}")
    return texts, labels


def _class_counts(labels: List[int], label_names: List[str]) -> dict:
    from collections import Counter
    counts = Counter(labels)
    return {label_names[k]: v for k, v in sorted(counts.items())}


def create_datasets(filepath: str, config: Any,
                    test_filepath: Optional[str] = None):
    """Create train / val / test datasets and return them with the tokenizer."""
    from sklearn.model_selection import train_test_split

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    texts, labels = load_data(filepath, config)

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
        # Separate test file provided
        test_texts, test_labels = load_data(test_filepath, config)

        val_size = config.val_split / (config.train_split + config.val_split)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            test_size=val_size,
            random_state=42,
            stratify=labels,
        )

    logging.info(f"Dataset sizes  —  train: {len(train_texts)}, "
                 f"val: {len(val_texts)}, test: {len(test_texts)}")

    train_dataset = HeadlineDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset   = HeadlineDataset(val_texts, val_labels, tokenizer, config.max_length)
    test_dataset  = HeadlineDataset(test_texts, test_labels, tokenizer, config.max_length)

    return train_dataset, val_dataset, test_dataset, tokenizer
