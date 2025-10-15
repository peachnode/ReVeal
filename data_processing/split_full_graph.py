#!/usr/bin/env python3
"""Split a Devign full_graph JSON file into train/valid/test shards.

The script expects the input file to contain a JSON array whose elements
provide the GGNN inputs (``node_features``, ``graph``, ``targets``) along
with any metadata produced by ``full_data_prep_script.ipynb``.

Example usage::

    python split_full_graph.py \
        --input NOT_HELPFUL_UNREACHED_SATURATED_500-full_graph.json \
        --output-dir prepared_devign \
        --train-ratio 0.8 --valid-ratio 0.1 --test-ratio 0.1

This writes ``train_GGNNinput.json``, ``valid_GGNNinput.json`` and
``test_GGNNinput.json`` into ``prepared_devign``. Ratios are normalized
so they do not need to sum to 1.0 exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the *_full_graph.json produced by full_data_prep_script.ipynb",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the split files will be written",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of examples assigned to the training split (default: 0.8)",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Fraction of examples assigned to the validation split (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of examples assigned to the test split (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for shuffling before splitting (default: 0)",
    )
    return parser.parse_args()


def normalize_ratios(train: float, valid: float, test: float) -> List[float]:
    total = train + valid + test
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    return [train / total, valid / total, test / total]


def split_examples(examples: List[dict], ratios: List[float]) -> List[List[dict]]:
    n_examples = len(examples)
    train_ratio, valid_ratio, test_ratio = ratios

    train_count = math.floor(n_examples * train_ratio)
    valid_count = math.floor(n_examples * valid_ratio)
    # ensure all leftovers go to test split so counts sum to n_examples
    test_count = n_examples - train_count - valid_count

    train_split = examples[:train_count]
    valid_split = examples[train_count : train_count + valid_count]
    test_split = examples[train_count + valid_count : train_count + valid_count + test_count]

    return [train_split, valid_split, test_split]


def write_split(path: Path, examples: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(examples, fh)


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as fh:
        examples = json.load(fh)

    if not isinstance(examples, list):
        raise ValueError("Input JSON must contain a list of examples")

    random.seed(args.seed)
    random.shuffle(examples)

    ratios = normalize_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)
    splits = split_examples(examples, ratios)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_names = ["train_GGNNinput.json", "valid_GGNNinput.json", "test_GGNNinput.json"]

    for filename, split in zip(split_names, splits):
        write_split(args.output_dir / filename, split)

    print("Wrote:")
    for filename, split in zip(split_names, splits):
        print(f"  {filename}: {len(split)} examples")


if __name__ == "__main__":
    main()
