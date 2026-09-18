"""Shared command-line argument types and dataset split options."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

GROUPING_CHOICES = ("pair_stratified", "peptide", "cluster")


def positive_int(value: str) -> int:
    """Parse a positive integer, such as a batch size or epoch count."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be an integer greater than zero")
    return number


def nonnegative_int(value: str) -> int:
    """Parse a nonnegative integer, such as a DataLoader worker count."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Must be a nonnegative integer")
    return number


def positive_float(value: str) -> float:
    """Parse a finite positive number, such as a learning rate."""
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be a finite positive number")
    return number


def probability(value: str) -> float:
    """Parse a classification threshold in [0, 1]."""
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Must be in [0, 1]")
    return number


def fraction(value: str) -> float:
    """Parse a validation fraction strictly between zero and one."""
    number = probability(value)
    if number in (0, 1):
        raise argparse.ArgumentTypeError(
            "Validation fraction must be greater than zero and less than one"
        )
    return number


def add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared split options; explicitly supplied arguments override the JSON file."""
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON file specifying input datasets and split settings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New workspace directory; overrides output in the JSON file",
    )
    parser.add_argument(
        "--validation-grouping",
        choices=GROUPING_CHOICES,
        help="Validation grouping; uses the JSON setting, or pair_stratified if absent",
    )
    parser.add_argument(
        "--test-grouping",
        choices=GROUPING_CHOICES,
        help="Test exclusion grouping; uses the JSON setting or the task default",
    )
    parser.add_argument(
        "--validation-fraction",
        type=fraction,
        help="Target fraction of positive records for validation; JSON default: 0.1",
    )
    parser.add_argument(
        "--split-seed", type=int, help="Split random seed; JSON default: 42"
    )
    parser.add_argument(
        "--validation-negative-seed",
        type=int,
        help="Random seed for fixed validation negatives; JSON default: 1042",
    )


def prepare_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Collect only explicit overrides, leaving all other JSON settings unchanged."""
    names = (
        "output",
        "validation_grouping",
        "test_grouping",
        "validation_fraction",
        "split_seed",
        "validation_negative_seed",
    )
    return {
        name: str(value) if isinstance(value, Path) else value
        for name in names
        if (value := getattr(args, name, None)) is not None
    }


def load_prepare_config(
    path: Path, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Load split settings; resolve relative paths from the current working directory."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    config.update(overrides or {})
    for name in ("validation_grouping", "test_grouping"):
        if name in config and config[name] not in GROUPING_CHOICES:
            raise ValueError(f"{name} must be one of {GROUPING_CHOICES}")
    fraction(str(config.get("validation_fraction", 0.1)))
    return config


def add_inference_arguments(parser: argparse.ArgumentParser, *, native: bool) -> None:
    """Add options for batch prediction from a CSV without labels."""
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV; label is optional; see README for required sequence columns",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint produced by a training entry point in this package",
    )
    if native:
        parser.add_argument(
            "--asset-root",
            type=Path,
            required=True,
            help="Asset directory containing manifest.json and the base model/tokenizer files",
        )
        parser.add_argument(
            "--precision",
            choices=("fp32", "tf32", "bf16"),
            default="bf16",
            help="Inference precision; CPU execution does not enable bf16 autocast",
        )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output CSV; preserves row order and adds score, predicted_label and threshold",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=32 if native else 512,
        help="Number of records per prediction batch, including any supplied positive or negative records",
    )
    parser.add_argument(
        "--device", default="cuda", help="Compute device, such as cuda, cuda:0 or cpu"
    )
    parser.add_argument(
        "--threshold",
        type=probability,
        help="Classification threshold; defaults to the saved validation threshold, never fitted on prediction inputs",
    )
