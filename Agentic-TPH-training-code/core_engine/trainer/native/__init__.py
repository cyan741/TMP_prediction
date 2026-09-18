"""Protein language model training and evaluation.

NativeConfig specifies input encoding and model assets; TrainConfig specifies
optimization settings; EvalConfig specifies the checkpoint and evaluation files.
Use train.py for the command-line interface."""

from .config import (
    LEVEL_COMPONENTS,
    EvalConfig,
    NativeConfig,
    Precision,
    TrainConfig,
)
from .datasets import (
    DistributedEvalSampler,
    NegativeSampler,
    PairedDataset,
    RandomNegativeDataset,
    build_dataloaders,
    load_blacklist,
)
from .distributed import DistributedContext
from .encoding import InputContract, contract_for, encode_batch, pad_record
from .loop import EpochResult, TrainingResult, evaluate, train
from .metrics import Metrics, evaluate_predictions
from .models import HITPH_HIDDEN_SIZES, build_model, load_tokenizer

__all__ = [
    "EpochResult",
    "EvalConfig",
    "DistributedContext",
    "DistributedEvalSampler",
    "HITPH_HIDDEN_SIZES",
    "InputContract",
    "LEVEL_COMPONENTS",
    "Metrics",
    "NativeConfig",
    "NegativeSampler",
    "PairedDataset",
    "Precision",
    "RandomNegativeDataset",
    "TrainConfig",
    "TrainingResult",
    "build_dataloaders",
    "build_model",
    "contract_for",
    "encode_batch",
    "evaluate",
    "evaluate_predictions",
    "load_blacklist",
    "load_tokenizer",
    "pad_record",
    "train",
]
