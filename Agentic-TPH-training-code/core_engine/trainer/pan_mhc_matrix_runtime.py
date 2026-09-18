"""Train and evaluate the fixed 11-model pan-MHC matrix."""

from __future__ import annotations

import itertools
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .pan_mhc_matrix_data import DOMAINS, DomainBalancedSampler
from .pan_mhc_matrix_model import (
    MODEL_ID,
    PairDataset,
    PanMHCClassifier,
    PanMHCModelConfig,
    best_f1_threshold,
    collate_pairs,
    device_batch,
    evaluate_by_domain,
    predict_frame,
    save_checkpoint,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_specs() -> list[tuple[str, tuple[str, ...]]]:
    specs = [(domain, (domain,)) for domain in DOMAINS]
    specs.extend(
        (f"{left}+{right}", (left, right))
        for left, right in itertools.combinations(DOMAINS, 2)
    )
    specs.append(("all_four", DOMAINS))
    return specs


def _parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def train_one(
    *,
    name: str,
    domains: tuple[str, ...],
    train_all: pd.DataFrame,
    valid_all: pd.DataFrame,
    unseen: dict[str, pd.DataFrame],
    output_root: Path,
    device: torch.device,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    run_dir = Path(output_root) / "models" / name.replace("+", "__")
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    set_seed(seed)
    train = train_all.loc[train_all["domain"].isin(domains)].reset_index(drop=True)
    valid = valid_all.loc[valid_all["domain"].isin(domains)].reset_index(drop=True)
    config = PanMHCModelConfig(
        model_version=f"pan-mhc-chain-cnn-v1-{name.replace('+', '__')}-seed{seed}"
    )
    model = PanMHCClassifier(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    loss_function = nn.BCEWithLogitsLoss()
    best_score = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        sampler = DomainBalancedSampler(train, domains=domains, seed=seed + epoch)
        loader = DataLoader(
            PairDataset(train),
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_pairs,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )
        model.train()
        losses: list[float] = []
        for batch in loader:
            batch = device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch), batch["label"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss in {name} epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_predictions = predict_frame(
            model, valid, device=device, batch_size=batch_size
        )
        threshold = best_f1_threshold(validation_predictions)
        validation_metrics = evaluate_by_domain(
            validation_predictions, threshold=threshold
        )
        score = validation_metrics["macro"]["auprc"]
        history.append(
            {
                "epoch": epoch,
                "mean_loss": float(np.mean(losses)),
                "optimizer_steps": len(losses),
                "sampler": sampler.report(),
                "threshold": threshold,
                "validation": validation_metrics,
            }
        )
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"no best state for {name}")
    model.load_state_dict(best_state)
    model.to(device)
    best_threshold = float(history[best_epoch - 1]["threshold"])
    all_predictions = []
    for domain in unseen:
        frame = predict_frame(
            model, unseen[domain], device=device, batch_size=batch_size
        )
        frame["trained_domain"] = domain in domains
        all_predictions.append(frame)
    unseen_predictions = pd.concat(all_predictions, ignore_index=True)
    unseen_metrics = evaluate_by_domain(unseen_predictions, threshold=best_threshold)
    unseen_predictions.to_csv(run_dir / "unseen_predictions.csv", index=False)
    checkpoint = run_dir / "checkpoint.pt"
    training = {
        "run_name": name,
        "train_domains": list(domains),
        "seed": seed,
        "max_epochs": max_epochs,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_validation_macro_auprc": best_score,
        "decision_threshold": best_threshold,
        "history": history,
    }
    save_checkpoint(checkpoint, model=model, config=config, training=training)
    result = {
        "status": "succeeded",
        "model_id": MODEL_ID,
        "model_version": config.model_version,
        "run_name": name,
        "train_domains": list(domains),
        "trainable_parameters": _parameter_count(model),
        "train_rows_available": len(train),
        "valid_rows": len(valid),
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "duration_seconds": time.time() - started,
        "training": training,
        "unseen": unseen_metrics,
        "production_champion_updated": False,
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result
