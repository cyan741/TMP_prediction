"""Train models, evaluate fixed labels and select validation thresholds.

Validation AUROC selects the best checkpoint. Checkpoints store model settings
and sequence widths; the selected threshold is saved in an adjacent JSON file."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import EvalConfig, NativeConfig, Precision, TrainConfig
from .datasets import RandomNegativeDataset, build_dataloaders
from .distributed import (
    DistributedContext,
    barrier,
    broadcast_stop,
    finalize,
    gather_predictions,
    initialize,
    wrap_model,
)
from .encoding import encode_batch
from .metrics import Metrics, best_threshold, evaluate_predictions, mean_metrics
from .models import build_model, load_tokenizer

CHECKPOINT_SUFFIX = ".pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def apply_precision(precision: Precision) -> None:
    """Allow CUDA TF32 matrix operations for TF32 and BF16 precision modes."""

    allow = precision in {Precision.TF32, Precision.BF16}
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow


@dataclass(frozen=True, slots=True)
class EpochResult:
    epoch: int
    train_loss: float
    train_metrics: Metrics
    valid_loss: float
    valid_rounds: list[Metrics]
    valid_summary: Mapping[str, float]
    seconds: float
    checkpoint: str | None = None

    @property
    def selection_score(self) -> float:
        return float(self.valid_summary["selection_score_avg"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_metrics": self.train_metrics.to_dict(),
            "valid_loss": self.valid_loss,
            "valid_rounds": [item.to_dict() for item in self.valid_rounds],
            "valid_summary": dict(self.valid_summary),
            "seconds": self.seconds,
            "checkpoint": self.checkpoint,
        }


@dataclass(frozen=True, slots=True)
class TrainingResult:
    config: Mapping[str, Any]
    epochs: list[EpochResult]
    best_epoch: int
    best_score: float
    best_checkpoint: str | None
    stopped_early: bool
    seconds: float
    selected_threshold: float = 0.5
    selected_threshold_mcc: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": dict(self.config),
            "epochs": [item.to_dict() for item in self.epochs],
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "best_checkpoint": self.best_checkpoint,
            "stopped_early": self.stopped_early,
            "seconds": self.seconds,
            "selected_threshold": self.selected_threshold,
            "selected_threshold_mcc": self.selected_threshold_mcc,
            "selected_threshold_source": "validation split, cut maximising MCC",
            # Select checkpoints using validation AUROC only.
            # Threshold-dependent metrics are reported separately and do not select the model.
            "checkpoint_selection_metric": "validation_auroc",
            "best_validation_auroc": self.best_score,
        }


def _autocast(config: NativeConfig):
    if config.precision is Precision.BF16 and config.device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", enabled=False)


def _forward(
    model: nn.Module,
    tokenizer: Any,
    batch: Sequence[Any],
    *,
    config: NativeConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One batch to logits and labels, for both sampling regimes."""

    if config.rand_neg:
        positive, negatives, pmhc = batch
        count = len(positive)
        # negatives collates to one list per draw index, each in batch order,
        # so repeating the pMHC side per draw keeps every negative against its
        # own positive's pMHC.
        drawn = [str(item) for chunk in negatives for item in chunk]
        tokens = encode_batch(
            tokenizer,
            list(positive) + drawn,
            list(pmhc) * (1 + len(negatives)),
            model_id=config.model_id,
            plm_input=config.plm_input,
            device=device,
        )
        labels = torch.LongTensor([1] * count + [0] * len(drawn)).to(device)
    else:
        tcr, pmhc, raw_labels = batch
        tokens = encode_batch(
            tokenizer,
            list(tcr),
            list(pmhc),
            model_id=config.model_id,
            plm_input=config.plm_input,
            device=device,
        )
        labels = torch.LongTensor([int(value) for value in raw_labels]).to(device)
    return model(tokens), labels


def _set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    """Keep both sampler order and random-negative draws on the same epoch."""

    sampler = loader.sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    dataset = loader.dataset
    if isinstance(dataset, RandomNegativeDataset):
        dataset.set_epoch(epoch)


def _class_weights(negatives_per_positive: int) -> torch.Tensor:
    """Weights whose total mass is equal for positive and negative classes."""

    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be positive")
    return torch.tensor([1.0 / negatives_per_positive, 1.0])


def _run_epoch(
    model: nn.Module,
    tokenizer: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    *,
    config: NativeConfig,
    device: torch.device,
    accumulation: int = 1,
    distributed: DistributedContext | None = None,
    class_weights: torch.Tensor | None = None,
) -> tuple[float, Metrics]:
    training = optimizer is not None
    model.train() if training else model.eval()
    criterion = nn.CrossEntropyLoss(
        weight=None if class_weights is None else class_weights.to(device)
    )
    loss_sum = 0.0
    sample_count = 0
    true_all: list[int] = []
    prob_all: list[float] = []
    if training:
        optimizer.zero_grad()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, batch in enumerate(loader):
            with _autocast(config):
                logits, labels = _forward(
                    model, tokenizer, batch, config=config, device=device
                )
                loss = criterion(logits, labels)
            if training:
                # Scaling by the accumulation factor keeps the gradient equal to
                # the one a single large batch would have produced.
                (loss / accumulation).backward()
                if (step + 1) % accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            probabilities = nn.Softmax(dim=1)(logits.float())[:, 1]
            true_all.extend(labels.detach().cpu().numpy().tolist())
            prob_all.extend(probabilities.detach().cpu().numpy().tolist())
            batch_count = int(labels.numel())
            loss_sum += float(loss.detach()) * batch_count
            sample_count += batch_count
    if training and len(loader) % accumulation:
        optimizer.step()
        optimizer.zero_grad()
    if not sample_count and distributed is None:
        raise RuntimeError("the loader produced no batches")
    if distributed is not None:
        true_all, prob_all, loss_sum, sample_count = gather_predictions(
            distributed, true_all, prob_all, loss_sum, sample_count
        )
    if not sample_count:
        raise RuntimeError("the loader produced no samples")
    return float(loss_sum / sample_count), evaluate_predictions(
        true_all, prob_all, threshold=config.threshold
    )


def _predict(
    model: nn.Module,
    tokenizer: Any,
    loader: DataLoader,
    *,
    config: NativeConfig,
    device: torch.device,
    distributed: DistributedContext | None = None,
) -> tuple[list[int], list[float]]:
    """Labels and class-1 probabilities for one pass, without thresholding.

    Selecting a cut needs the probabilities themselves; _run_epoch computes
    them and then throws them away behind a Metrics object.
    """

    model.eval()
    true_all: list[int] = []
    prob_all: list[float] = []
    with torch.no_grad():
        for batch in loader:
            with _autocast(config):
                logits, labels = _forward(
                    model, tokenizer, batch, config=config, device=device
                )
            probabilities = nn.Softmax(dim=1)(logits.float())[:, 1]
            true_all.extend(labels.detach().cpu().numpy().tolist())
            prob_all.extend(probabilities.detach().cpu().numpy().tolist())
    if distributed is not None:
        true_all, prob_all, _, _ = gather_predictions(
            distributed, true_all, prob_all, 0.0, len(true_all)
        )
    return true_all, prob_all


THRESHOLD_SUFFIX = ".selected_threshold.json"


def threshold_sidecar(checkpoint: Path) -> Path:
    """Where a checkpoint's frozen decision threshold lives.

    A sidecar rather than a field inside the checkpoint: the checkpoint is
    written during the epoch loop, while the threshold can only be chosen after
    the loop knows which epoch won.
    """

    return Path(str(checkpoint) + THRESHOLD_SUFFIX)


def _validate(
    model: nn.Module,
    tokenizer: Any,
    loader: DataLoader,
    *,
    config: NativeConfig,
    device: torch.device,
    rounds: int,
    epoch: int,
    distributed: DistributedContext | None = None,
    class_weights: torch.Tensor | None = None,
) -> tuple[float, list[Metrics]]:
    """Repeat validation, redrawing negatives each pass when they are random."""

    losses: list[float] = []
    results: list[Metrics] = []
    for index in range(rounds):
        dataset = loader.dataset
        if isinstance(dataset, RandomNegativeDataset):
            dataset.set_epoch(epoch * 1000 + index)
        sampler = loader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch * 1000 + index)
        loss, metrics = _run_epoch(
            model,
            tokenizer,
            loader,
            None,
            config=config,
            device=device,
            distributed=distributed,
            class_weights=class_weights,
        )
        losses.append(loss)
        results.append(metrics)
    return float(sum(losses) / len(losses)), results


def _checkpoint_payload(
    model: nn.Module, config: TrainConfig, *, epoch: int, score: float
) -> dict[str, Any]:
    """Save model weights and input settings needed to restore the same model structure."""

    module = model.module if hasattr(model, "module") else model
    return {
        "format_version": 1,
        "state_dict": module.state_dict(),
        "model_id": config.base.model_id,
        "level": config.base.level,
        "head_type": config.base.head_type,
        "plm_output": config.base.plm_output,
        "plm_input": config.base.plm_input,
        "finetune": config.base.finetune,
        "pep_max_len": config.base.pep_max_len,
        "tcr_max_len": config.base.tcr_max_len,
        "hla_max_len": config.base.hla_max_len,
        "epoch": epoch,
        "selection_score": score,
        "seed": config.base.seed,
    }


def load_checkpoint(path: Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        # Legacy checkpoints may contain only weights; model settings must then
        # be supplied separately by the evaluation configuration.
        return {"format_version": 0, "state_dict": payload}
    return payload


def _prune(directory: Path, keep: int) -> None:
    files = sorted(
        directory.glob(f"*{CHECKPOINT_SUFFIX}"), key=lambda p: p.stat().st_mtime
    )
    for stale in files[: max(0, len(files) - keep)]:
        stale.unlink(missing_ok=True)


def train(config: TrainConfig) -> TrainingResult:
    """Train one model, returning every epoch's metrics and the best checkpoint."""

    base = config.base
    context: DistributedContext | None = None
    try:
        context = initialize(base)
        set_seed(base.seed)
        apply_precision(base.precision)
        device = context.device
        output_dir = Path(config.output_dir)
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
        barrier(context)

        tokenizer = load_tokenizer(base.model_id, base.asset_root)
        model = build_model(
            base.model_id,
            base.asset_root,
            head_type=base.head_type,
            plm_output=base.plm_output,
            finetune=base.finetune,
        ).to(device)

        start_epoch = 0
        if config.warm_start_checkpoint is not None:
            payload = load_checkpoint(config.warm_start_checkpoint, map_location=device)
            model.load_state_dict(payload["state_dict"])
            # A warm start loads model weights but starts a fresh epoch budget.

        model = wrap_model(model, context)
        validation_base = (
            replace(base, rand_neg=False) if config.fixed_validation else base
        )
        loaders = build_dataloaders(
            base,
            splits=("train",),
            blacklist_path=config.negative_blacklist,
            distributed=context.enabled,
        )
        loaders.update(
            build_dataloaders(
                validation_base,
                splits=("valid",),
                blacklist_path=config.negative_blacklist,
                distributed=context.enabled,
            )
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        # Equal total weight per class: with k negatives per positive the negative
        # side carries k times the rows, so it is scaled by 1/k.
        class_weights = (
            _class_weights(base.negatives_per_positive)
            if config.class_weighted
            else None
        )

        epochs: list[EpochResult] = []
        best_score, best_epoch, best_checkpoint = -math.inf, -1, None
        stopped_early = False
        started = time.time()

        for epoch in range(start_epoch + 1, config.epochs + 1):
            _set_loader_epoch(loaders["train"], epoch)
            epoch_start = time.time()
            train_loss, train_metrics = _run_epoch(
                model,
                tokenizer,
                loaders["train"],
                optimizer,
                config=base,
                device=device,
                accumulation=config.gradient_accumulation_steps,
                distributed=context,
                class_weights=class_weights,
            )
            valid_loss, valid_rounds = _validate(
                model,
                tokenizer,
                loaders["valid"],
                config=validation_base,
                device=device,
                rounds=config.rounds,
                epoch=epoch,
                distributed=context,
                class_weights=class_weights,
            )
            summary = mean_metrics(valid_rounds)
            checkpoint_path: str | None = None
            if summary["selection_score_avg"] > best_score:
                best_score = summary["selection_score_avg"]
                best_epoch = epoch
                target = (
                    output_dir
                    / f"{base.model_id}_fold{base.fold}_epoch{epoch}{CHECKPOINT_SUFFIX}"
                )
                if context.is_main:
                    torch.save(
                        _checkpoint_payload(
                            model, config, epoch=epoch, score=best_score
                        ),
                        target,
                    )
                    _prune(output_dir, config.keep_checkpoints)
                checkpoint_path = str(target)
                best_checkpoint = checkpoint_path
            # Rank 0 must finish its checkpoint write before any rank starts the
            # next epoch or exits on the shared stop decision.
            barrier(context)
            epochs.append(
                EpochResult(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_metrics=train_metrics,
                    valid_loss=valid_loss,
                    valid_rounds=valid_rounds,
                    valid_summary=summary,
                    seconds=time.time() - epoch_start,
                    checkpoint=checkpoint_path,
                )
            )
            if context.is_main:
                progress = epochs[-1].to_dict()
                with (output_dir / "epochs.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(progress, ensure_ascii=False) + "\n")
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "valid_auroc": summary["selection_score_avg"],
                            "best_epoch": best_epoch,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if not math.isfinite(train_loss) or not math.isfinite(valid_loss):
                raise FloatingPointError("non-finite training or validation loss")
            should_stop = epoch - best_epoch >= config.early_stop
            if broadcast_stop(context, should_stop):
                stopped_early = True
                break

        # Choose the threshold on validation data and freeze it before evaluating the test set.
        cut, cut_mcc = 0.5, float("nan")
        if best_checkpoint:
            best_payload = load_checkpoint(Path(best_checkpoint), map_location=device)
            model.load_state_dict(best_payload["state_dict"]) if not hasattr(
                model, "module"
            ) else model.module.load_state_dict(best_payload["state_dict"])
            _set_loader_epoch(loaders["valid"], config.epochs + 1)
            valid_true, valid_prob = _predict(
                model,
                tokenizer,
                loaders["valid"],
                config=validation_base,
                device=device,
                distributed=context,
            )
            cut, cut_mcc = best_threshold(valid_true, valid_prob)
            if context.is_main:
                threshold_sidecar(Path(best_checkpoint)).write_text(
                    json.dumps(
                        {
                            "threshold": cut,
                            "validation_mcc": cut_mcc,
                            "chosen_on": "valid",
                            "rule": "argmax MCC",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            barrier(context)

        result = TrainingResult(
            config=config.to_dict(),
            epochs=epochs,
            best_epoch=best_epoch,
            best_score=best_score,
            best_checkpoint=best_checkpoint,
            stopped_early=stopped_early,
            seconds=time.time() - started,
            selected_threshold=cut,
            selected_threshold_mcc=cut_mcc,
        )
        if context.is_main:
            (output_dir / "training_result.json").write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return result
    finally:
        if context is not None:
            finalize(context)


def evaluate(config: EvalConfig) -> dict[str, Any]:
    """Score one checkpoint on the requested splits, returning the numbers."""

    base = config.base
    context: DistributedContext | None = None
    try:
        context = initialize(base)
        set_seed(base.seed)
        apply_precision(base.precision)
        device = context.device
        payload = load_checkpoint(config.checkpoint, map_location=device)
        model_id = payload.get("model_id", base.model_id)
        if model_id != base.model_id:
            raise ValueError(
                f"checkpoint was trained as {model_id!r} but the request asks for {base.model_id!r}"
            )
        tokenizer = load_tokenizer(base.model_id, base.asset_root)
        model = build_model(
            base.model_id,
            base.asset_root,
            head_type=payload.get("head_type", base.head_type),
            plm_output=payload.get("plm_output", base.plm_output),
            finetune=payload.get("finetune", base.finetune),
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model = wrap_model(model, context)

        # Evaluation files carry fixed labels.  Redrawing is opt-in and travels
        # with the corpus blacklist so known binders cannot become negatives.
        scoring = replace(base, rand_neg=config.redraw_negatives)
        loaders = build_dataloaders(
            scoring,
            splits=config.splits,
            blacklist_path=config.negative_blacklist,
            distributed=context.enabled,
        )
        sidecar = threshold_sidecar(Path(config.checkpoint))
        if sidecar.is_file():
            frozen = json.loads(sidecar.read_text(encoding="utf-8"))
            frozen_threshold = float(frozen["threshold"])
            frozen_source = "validation split, cut maximising MCC"
        else:
            frozen_threshold, frozen_source = 0.5, "no sidecar found; fell back to 0.5"
        results: dict[str, Any] = {
            "checkpoint": str(config.checkpoint),
            "selected_threshold": frozen_threshold,
            "selected_threshold_source": frozen_source,
            "splits": {},
        }
        for split, loader in loaders.items():
            loss, rounds = _validate(
                model,
                tokenizer,
                loader,
                config=scoring,
                device=device,
                rounds=config.repeats,
                epoch=0,
                distributed=context,
            )
            results["splits"][split] = {
                "loss": loss,
                "rounds": [item.to_dict() for item in rounds],
                "summary": mean_metrics(rounds),
            }
            # AUROC/AUPRC do not depend on the threshold. Report F1, MCC and recall at both
            # 0.5 and the frozen validation threshold; neither is optimized on the test set.
            true_all, prob_all = _predict(
                model,
                tokenizer,
                loader,
                config=scoring,
                device=device,
                distributed=context,
            )
            results["splits"][split]["at_selected_threshold"] = {
                "threshold": frozen_threshold,
                "source": frozen_source,
                **evaluate_predictions(
                    true_all, prob_all, threshold=frozen_threshold
                ).to_dict(),
            }
            results["splits"][split]["at_half"] = {
                "threshold": 0.5,
                **evaluate_predictions(true_all, prob_all, threshold=0.5).to_dict(),
            }
        return results
    finally:
        if context is not None:
            finalize(context)
