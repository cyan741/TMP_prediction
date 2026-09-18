"""Configuration objects for input encoding, devices, sampling and optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class Precision(str, Enum):
    FP32 = "fp32"
    TF32 = "tf32"
    BF16 = "bf16"


# Input columns accepted by the dataset loader, indexed by data level.
# Level IV uses the Level III column layout but supplies variable-domain chains.
LEVEL_COMPONENTS: Mapping[str, tuple[str, ...]] = {
    "1": ("pep", "beta"),
    "2": ("pep", "hla", "beta"),
    "3": ("pep", "hla", "ab"),
    "4": ("pep", "hla", "ab"),
}


@dataclass(frozen=True, slots=True)
class NativeConfig:
    """Everything both training and evaluation need."""

    model_id: str
    level: str
    data_dir: Path
    asset_root: Path
    fold: int = 0
    batch_size: int = 32
    pep_max_len: int = 13
    tcr_max_len: int = 19
    hla_max_len: int = 34
    plm_input: str = "cat"
    plm_output: str = "cls"
    head_type: str = "3MLP"
    threshold: float = 0.5
    seed: int = 42
    # Use a separate seed to vary negative sampling independently of model initialization.
    # When omitted, negative sampling uses the main training seed.
    negative_seed: int | None = None
    rand_neg: bool = True
    # Negatives drawn per positive.  1 gives a balanced 1:1 positive/negative ratio;
    # higher values keep the positives fixed and widen the negative side only.
    negatives_per_positive: int = 1
    finetune: bool = True
    num_workers: int = 0
    precision: Precision = Precision.FP32
    device: str = "cuda"
    # ``nproc_per_node`` is the local torchrun world size.  The default keeps
    # direct Python invocation and existing single-card runs unchanged.
    distributed: bool = False
    nproc_per_node: int = 1
    distributed_backend: str | None = None
    components: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.level not in LEVEL_COMPONENTS:
            raise ValueError(f"unsupported level: {self.level!r}")
        if self.plm_input not in {"cat", "sep"}:
            raise ValueError(f"unsupported plm_input: {self.plm_input!r}")
        if self.plm_output not in {"cls", "mean"}:
            raise ValueError(f"unsupported plm_output: {self.plm_output!r}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.negatives_per_positive < 1:
            raise ValueError("negatives_per_positive must be positive")
        if self.nproc_per_node < 1:
            raise ValueError("nproc_per_node must be positive")
        if self.distributed_backend not in {None, "nccl", "gloo"}:
            raise ValueError(
                f"unsupported distributed_backend: {self.distributed_backend!r}"
            )

    @property
    def component_columns(self) -> tuple[str, ...]:
        return self.components or LEVEL_COMPONENTS[self.level]

    @property
    def sampling_seed(self) -> int:
        """Return the random seed used for negative sampling."""

        return self.seed if self.negative_seed is None else self.negative_seed

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sampling_seed"] = self.sampling_seed
        payload["data_dir"] = str(self.data_dir)
        payload["asset_root"] = str(self.asset_root)
        payload["precision"] = self.precision.value
        payload["components"] = list(self.component_columns)
        return payload


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Training-only knobs."""

    base: NativeConfig
    output_dir: Path
    epochs: int = 10
    learning_rate: float = 1e-5
    early_stop: int = 5
    validation_times: int | None = None
    fixed_validation: bool = False
    gradient_accumulation_steps: int = 1
    warm_start_checkpoint: Path | None = None
    keep_checkpoints: int = 1
    negative_blacklist: Path | None = None
    # Reweight the loss so both classes carry equal total weight.  Separates
    # "more negatives" from "a negative-dominated gradient": without it the
    # two effects arrive together and neither can be attributed.
    class_weighted: bool = False

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.early_stop < 1:
            raise ValueError("early_stop must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.keep_checkpoints < 1:
            raise ValueError("keep_checkpoints must be positive")

    @property
    def rounds(self) -> int:
        """Validation repeats per epoch.

        Repeated evaluation can average over independently drawn negative samples.
        Fixed validation labels require only one pass per epoch.
        """

        if self.fixed_validation:
            return 1
        if self.validation_times is not None:
            return self.validation_times
        return 5 if self.base.rand_neg else 1

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "base": self.base.to_dict(),
            "output_dir": str(self.output_dir),
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "early_stop": self.early_stop,
            "validation_times": self.rounds,
            "fixed_validation": self.fixed_validation,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warm_start_checkpoint": str(self.warm_start_checkpoint)
            if self.warm_start_checkpoint
            else None,
            "keep_checkpoints": self.keep_checkpoints,
            "negative_blacklist": str(self.negative_blacklist)
            if self.negative_blacklist
            else None,
            "class_weighted": self.class_weighted,
        }
        return payload


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Evaluation-only knobs."""

    base: NativeConfig
    checkpoint: Path
    splits: tuple[str, ...] = ("test",)
    rounds: int | None = None
    # Evaluation reads the labels in the file by default.  Redrawing is only
    # correct for a split that carries positives alone, and then only with the
    # corpus blacklist, so both are opt-in and travel together.
    redraw_negatives: bool = False
    negative_blacklist: Path | None = None

    @property
    def repeats(self) -> int:
        if self.rounds is not None:
            return self.rounds
        return 5 if self.redraw_negatives else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "checkpoint": str(self.checkpoint),
            "splits": list(self.splits),
            "rounds": self.repeats,
            "redraw_negatives": self.redraw_negatives,
            "negative_blacklist": str(self.negative_blacklist)
            if self.negative_blacklist
            else None,
        }
