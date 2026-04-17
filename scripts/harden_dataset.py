#!/usr/bin/env python3
"""
Harden the Final_NoDuplicates dataset to remove trivial keyword shortcuts.

The original dataset has near-perfect separability because:
  - 98.9% of central-route headlines contain "because" (0% in other classes)
  - 39.3% of peripheral-route headlines start with ALL-CAPS emotional prefixes

This script diversifies the surface-level cues so the model must learn
actual persuasion semantics rather than keyword matching.

Changes:
  1. Central Route:  Replace "because" with varied causal connectors in ~65%
                     of headlines (keep ~35% with original "because")
  2. Peripheral Route: Remove ALL-CAPS prefixes in ~60% of headlines that
                       have them (keep ~40% so the model still trains on them)
  3. Neutral Route:  Insert "because" into ~15% of headlines to break the
                     exclusive "because" → central-route association

Usage:
    python scripts/harden_dataset.py
    python scripts/harden_dataset.py --input Dataset/Final_NoDuplicates.json \
                                     --output Dataset/Final_NoDuplicates_Hardened.json
"""

import argparse
import copy
import json
import random
import re
import sys
from collections import Counter


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42

# Probability of replacing "because" in central-route headlines
CENTRAL_REPLACE_PROB = 0.65

# Probability of removing an ALL-CAPS prefix in peripheral-route headlines
PERIPHERAL_REMOVE_PROB = 0.60

# Probability of inserting "because" into a neutral-route headline
NEUTRAL_INSERT_PROB = 0.15

# ──────────────────────────────────────────────────────────────────────────────
# Central Route: "because" diversification
# ──────────────────────────────────────────────────────────────────────────────

# Drop-in replacements (grammatically equivalent to "because")
DROPIN_ALTERNATIVES = [
    "since",
    "as",
    "given that",
    "considering that",
    "due to the fact that",
]

# Restructuring connectors (split sentence at "because")
RESTRUCTURE_CONNECTORS = [
    ". This is because",           # keeps "because" but in a different position
    ". Research suggests that",
    ". Studies indicate that",
    ". Evidence shows that",
    ". Experts explain this by noting that",
    ". This occurs as",
]


def harden_central(text: str) -> tuple[str, bool]:
    """Replace 'because' with a varied causal connector."""
    if "because" not in text.lower():
        return text, False

    if random.random() >= CENTRAL_REPLACE_PROB:
        return text, False  # keep original

    # 70% simple drop-in, 30% restructure into two sentences
    if random.random() < 0.70:
        alt = random.choice(DROPIN_ALTERNATIVES)
        new_text = re.sub(r"\bbecause\b", alt, text, count=1, flags=re.IGNORECASE)
    else:
        parts = re.split(r"\bbecause\b", text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            return text, False
        claim = parts[0].rstrip().rstrip(".")
        explanation = parts[1].strip()
        connector = random.choice(RESTRUCTURE_CONNECTORS)
        # Lowercase explanation start if connector ends with "that"/"as"
        if explanation and explanation[0].isupper():
            if connector.rstrip().endswith(("that", "as")):
                explanation = explanation[0].lower() + explanation[1:]
        new_text = claim + connector + " " + explanation

    return new_text, True


# ──────────────────────────────────────────────────────────────────────────────
# Peripheral Route: ALL-CAPS prefix removal (partial)
# ──────────────────────────────────────────────────────────────────────────────

CAPS_PREFIXES = [
    "SHOCKING:",
    "URGENT ALERT:",
    "URGENT:",
    "PLEASE SHARE:",
    "WARNING:",
    "BREAKING:",
    "ALERT:",
    "EMERGENCY:",
    "EXCLUSIVE:",
    "URUGET:",           # typo in original data
]


def harden_peripheral(text: str) -> tuple[str, bool]:
    """Remove ALL-CAPS emotional prefix from some peripheral headlines."""
    matched_prefix = None
    for prefix in CAPS_PREFIXES:
        if text.upper().startswith(prefix):
            matched_prefix = prefix
            break

    if matched_prefix is None:
        return text, False

    if random.random() >= PERIPHERAL_REMOVE_PROB:
        return text, False  # keep original prefix

    # Remove the prefix, clean up whitespace, capitalize
    # Find the actual prefix in the text (case-preserving removal)
    new_text = text[len(matched_prefix):].strip()
    if new_text:
        new_text = new_text[0].upper() + new_text[1:]

    return new_text, True


# ──────────────────────────────────────────────────────────────────────────────
# Neutral Route: inject "because" into some headlines
# ──────────────────────────────────────────────────────────────────────────────

# Patterns where we can safely restructure "[X] leads to / causes / results in [Y]"
# into "[Y] because [X]" — but this is fragile, so we only apply to clean matches.
CAUSAL_VERBS = [
    (r"^(.+?)\s+leads to\s+(.+)$",        "leads to"),
    (r"^(.+?)\s+results in\s+(.+)$",      "results in"),
    (r"^(.+?)\s+causes?\s+(.+)$",         "causes"),
    (r"^(.+?)\s+is linked to\s+(.+)$",    "is linked to"),
]


def harden_neutral(text: str) -> tuple[str, bool]:
    """Insert 'because' into some neutral headlines by restructuring causal verbs."""
    if random.random() >= NEUTRAL_INSERT_PROB:
        return text, False

    # Strip trailing period for matching
    text_clean = text.rstrip(".")

    for pattern, _ in CAUSAL_VERBS:
        match = re.match(pattern, text_clean, re.IGNORECASE)
        if match:
            cause = match.group(1).strip()
            effect = match.group(2).strip()

            # Build: "[Effect] because [cause]."
            # Lowercase the cause start, capitalize the effect start
            if cause and cause[0].isupper():
                cause = cause[0].lower() + cause[1:]
            if effect:
                effect = effect[0].upper() + effect[1:]

            new_text = f"{effect} because {cause}."
            return new_text, True

    return text, False


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(data: list) -> dict:
    """Compute keyword distribution statistics."""
    central   = [d for d in data if d["framework1_feature1"] == 1]
    peripheral = [d for d in data if d["framework1_feature2"] == 1]
    neutral   = [d for d in data if d["framework1_feature3"] == 1]

    # "because" distribution
    because_c = sum(1 for d in central    if "because" in d["text"].lower())
    because_p = sum(1 for d in peripheral if "because" in d["text"].lower())
    because_n = sum(1 for d in neutral    if "because" in d["text"].lower())

    # CAPS prefix distribution
    def has_caps_prefix(text):
        for prefix in CAPS_PREFIXES:
            if text.upper().startswith(prefix):
                return True
        return False

    caps_c = sum(1 for d in central    if has_caps_prefix(d["text"]))
    caps_p = sum(1 for d in peripheral if has_caps_prefix(d["text"]))
    caps_n = sum(1 for d in neutral    if has_caps_prefix(d["text"]))

    return {
        "because": {
            "central":    f"{because_c}/{len(central)} ({100*because_c/len(central):.1f}%)",
            "peripheral": f"{because_p}/{len(peripheral)} ({100*because_p/len(peripheral):.1f}%)",
            "neutral":    f"{because_n}/{len(neutral)} ({100*because_n/len(neutral):.1f}%)",
        },
        "caps_prefix": {
            "central":    f"{caps_c}/{len(central)} ({100*caps_c/len(central):.1f}%)",
            "peripheral": f"{caps_p}/{len(peripheral)} ({100*caps_p/len(peripheral):.1f}%)",
            "neutral":    f"{caps_n}/{len(neutral)} ({100*caps_n/len(neutral):.1f}%)",
        },
    }


def print_stats(label: str, stats: dict):
    """Pretty-print keyword distribution."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  {'Keyword':<20} {'Central':>12} {'Peripheral':>14} {'Neutral':>12}")
    print(f"  {'-'*58}")
    print(f"  {'\"because\"':<20} {stats['because']['central']:>12} "
          f"{stats['because']['peripheral']:>14} {stats['because']['neutral']:>12}")
    print(f"  {'CAPS prefix':<20} {stats['caps_prefix']['central']:>12} "
          f"{stats['caps_prefix']['peripheral']:>14} {stats['caps_prefix']['neutral']:>12}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Harden headline dataset")
    parser.add_argument("--input",  default="Dataset/Final_NoDuplicates.json",
                        help="Input dataset path")
    parser.add_argument("--output", default="Dataset/Final_NoDuplicates_Hardened.json",
                        help="Output hardened dataset path")
    parser.add_argument("--seed",   type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load
    with open(args.input) as f:
        original_data = json.load(f)

    # Deep copy for modification
    data = copy.deepcopy(original_data)

    # Before stats
    before_stats = compute_stats(original_data)
    print_stats("BEFORE HARDENING", before_stats)

    # Apply hardening
    changes = {"central": 0, "peripheral": 0, "neutral": 0}

    for item in data:
        if item["framework1_feature1"] == 1:
            item["text"], changed = harden_central(item["text"])
            if changed:
                changes["central"] += 1

        elif item["framework1_feature2"] == 1:
            item["text"], changed = harden_peripheral(item["text"])
            if changed:
                changes["peripheral"] += 1

        elif item["framework1_feature3"] == 1:
            item["text"], changed = harden_neutral(item["text"])
            if changed:
                changes["neutral"] += 1

    # After stats
    after_stats = compute_stats(data)
    print_stats("AFTER HARDENING", after_stats)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  CHANGES APPLIED")
    print(f"{'=' * 60}")
    print(f"  Central route:    {changes['central']} headlines modified "
          f"('because' diversified)")
    print(f"  Peripheral route: {changes['peripheral']} headlines modified "
          f"(caps prefix removed)")
    print(f"  Neutral route:    {changes['neutral']} headlines modified "
          f"('because' injected)")
    print(f"  Total:            {sum(changes.values())} / {len(data)}")

    # Show some examples of changes
    print(f"\n{'=' * 60}")
    print(f"  SAMPLE CHANGES")
    print(f"{'=' * 60}")

    example_count = 0
    for orig, new in zip(original_data, data):
        if orig["text"] != new["text"] and example_count < 6:
            cls = ("Central" if new["framework1_feature1"] == 1
                   else "Peripheral" if new["framework1_feature2"] == 1
                   else "Neutral")
            print(f"\n  [{cls}]")
            print(f"  BEFORE: {orig['text'][:120]}")
            print(f"  AFTER:  {new['text'][:120]}")
            example_count += 1

    # Save
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  Hardened dataset saved to: {args.output}")
    print(f"  Total samples: {len(data)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
