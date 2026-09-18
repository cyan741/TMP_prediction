"""Chain-aware fixed-length sequence model for the pan-MHC run matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .pan_mhc_matrix_data import (
    MHC_ALPHA_MAX_LENGTH,
    MHC_BETA_MAX_LENGTH,
    MHC_PACKED_LENGTH,
    PEPTIDE_MAX_LENGTH,
    TCR_MAX_LENGTH,
)

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: index + 1 for index, aa in enumerate(AA_ORDER)}
SPECIES_TO_ID = {"human": 0, "mouse": 1, "other": 2}
MODEL_ID = "pan-mhc-chain-cnn"


@dataclass(frozen=True, slots=True)
class PanMHCModelConfig:
    embedding_dim: int = 24
    sequence_dim: int = 48
    hidden_dim: int = 192
    dropout: float = 0.15
    model_id: str = MODEL_ID
    model_version: str = "pan-mhc-chain-cnn-v1"


class PairDataset(Dataset[dict[str, Any]]):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.records = frame.to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def encode_strict(values: Sequence[str], *, max_length: int) -> torch.Tensor:
    result = torch.zeros((len(values), max_length), dtype=torch.long)
    for row_index, value in enumerate(values):
        sequence = str(value or "")
        if len(sequence) > max_length:
            raise ValueError(
                f"sequence length {len(sequence)} exceeds fixed limit {max_length}"
            )
        for column_index, residue in enumerate(sequence):
            if residue not in AA_TO_ID:
                raise ValueError(f"unsupported amino acid {residue!r}")
            result[row_index, column_index] = AA_TO_ID[residue]
    return result


def encode_packed_mhc(records: list[dict[str, Any]]) -> torch.Tensor:
    result = torch.zeros((len(records), MHC_PACKED_LENGTH), dtype=torch.long)
    alpha = encode_strict(
        [row["mhc_alpha_seq"] for row in records], max_length=MHC_ALPHA_MAX_LENGTH
    )
    beta = encode_strict(
        [row.get("mhc_beta_seq", "") for row in records], max_length=MHC_BETA_MAX_LENGTH
    )
    result[:, :MHC_ALPHA_MAX_LENGTH] = alpha
    result[:, MHC_ALPHA_MAX_LENGTH:] = beta
    return result


def collate_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "peptide": encode_strict(
            [row["peptide"] for row in records], max_length=PEPTIDE_MAX_LENGTH
        ),
        "mhc_packed": encode_packed_mhc(records),
        "tcr_alpha": encode_strict(
            [row["tcr_alpha_cdr3"] for row in records], max_length=TCR_MAX_LENGTH
        ),
        "tcr_beta": encode_strict(
            [row["tcr_beta_cdr3"] for row in records], max_length=TCR_MAX_LENGTH
        ),
        "species": torch.tensor([SPECIES_TO_ID[row["species"]] for row in records]),
        "mhc_class": torch.tensor(
            [0 if row["mhc_class"] == "I" else 1 for row in records]
        ),
        "label": torch.tensor([float(row["label"]) for row in records]),
        "pair_key": [str(row["pair_key"]) for row in records],
        "domain": [str(row["domain"]) for row in records],
        "species_name": [str(row["species"]) for row in records],
    }


class LocalSequenceEncoder(nn.Module):
    def __init__(self, embedding: nn.Embedding, config: PanMHCModelConfig) -> None:
        super().__init__()
        self.embedding = embedding
        self.conv = nn.Conv1d(
            config.embedding_dim, config.sequence_dim, kernel_size=5, padding=2
        )
        self.norm = nn.LayerNorm(config.sequence_dim * 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(0)
        embedded = self.embedding(tokens).transpose(1, 2)
        features = torch.nn.functional.gelu(self.conv(embedded)).transpose(1, 2)
        mask3 = mask.unsqueeze(-1)
        mean = (features * mask3).sum(dim=1) / mask3.sum(dim=1).clamp_min(1)
        maximum = (
            features.masked_fill(~mask3, torch.finfo(features.dtype).min)
            .max(dim=1)
            .values
        )
        maximum = torch.where(
            mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum)
        )
        return self.norm(torch.cat([mean, maximum], dim=1))


class PanMHCClassifier(nn.Module):
    def __init__(self, config: PanMHCModelConfig) -> None:
        super().__init__()
        self.config = config
        shared_embedding = nn.Embedding(
            len(AA_TO_ID) + 1, config.embedding_dim, padding_idx=0
        )
        self.peptide_encoder = LocalSequenceEncoder(shared_embedding, config)
        self.mhc_alpha_encoder = LocalSequenceEncoder(shared_embedding, config)
        self.mhc_beta_encoder = LocalSequenceEncoder(shared_embedding, config)
        self.tcr_alpha_encoder = LocalSequenceEncoder(shared_embedding, config)
        self.tcr_beta_encoder = LocalSequenceEncoder(shared_embedding, config)
        self.species_embedding = nn.Embedding(3, 8)
        self.class_embedding = nn.Embedding(2, 8)
        component_dim = config.sequence_dim * 2
        input_dim = component_dim * 7 + 16
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        mhc = batch["mhc_packed"]
        peptide = self.peptide_encoder(batch["peptide"])
        mhc_alpha = self.mhc_alpha_encoder(mhc[:, :MHC_ALPHA_MAX_LENGTH])
        mhc_beta = self.mhc_beta_encoder(mhc[:, MHC_ALPHA_MAX_LENGTH:])
        tcr_alpha = self.tcr_alpha_encoder(batch["tcr_alpha"])
        tcr_beta = self.tcr_beta_encoder(batch["tcr_beta"])
        interactions = [peptide * tcr_alpha, peptide * tcr_beta]
        fields = [peptide, mhc_alpha, mhc_beta, tcr_alpha, tcr_beta, *interactions]
        fields.extend(
            [
                self.species_embedding(batch["species"]),
                self.class_embedding(batch["mhc_class"]),
            ]
        )
        return self.classifier(torch.cat(fields, dim=1)).squeeze(1)


def device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def predict_frame(
    model: PanMHCClassifier,
    frame: pd.DataFrame,
    *,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    loader = DataLoader(
        PairDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pairs,
    )
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            scores = torch.sigmoid(model(device_batch(batch, device))).cpu().numpy()
            labels = batch["label"].numpy()
            for index, score in enumerate(scores):
                rows.append(
                    {
                        "pair_key": batch["pair_key"][index],
                        "domain": batch["domain"][index],
                        "species": batch["species_name"][index],
                        "label": int(labels[index]),
                        "score": float(score),
                    }
                )
    return pd.DataFrame(rows)


def best_f1_threshold(predictions: pd.DataFrame) -> float:
    labels = predictions["label"].astype(int).to_numpy()
    scores = predictions["score"].astype(float).to_numpy()
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
    values = [
        (f1_score(labels, scores >= threshold, zero_division=0), float(threshold))
        for threshold in candidates
    ]
    return max(values, key=lambda item: (item[0], -item[1]))[1]


def binary_metrics(predictions: pd.DataFrame, *, threshold: float) -> dict[str, Any]:
    labels = predictions["label"].astype(int).to_numpy()
    scores = predictions["score"].astype(float).to_numpy()
    predicted = (scores >= threshold).astype(int)
    return {
        "n": len(labels),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "threshold": threshold,
    }


def evaluate_by_domain(
    predictions: pd.DataFrame, *, threshold: float
) -> dict[str, Any]:
    domains = {
        domain: binary_metrics(frame, threshold=threshold)
        for domain, frame in predictions.groupby("domain", sort=True)
    }
    macro = {
        metric: float(np.mean([values[metric] for values in domains.values()]))
        for metric in ("auroc", "auprc", "f1", "mcc")
    }
    result: dict[str, Any] = {"domains": domains, "macro": macro}
    if "species" in predictions:
        result["species"] = {
            species: binary_metrics(frame, threshold=threshold)
            for species, frame in predictions.groupby("species", sort=True)
            if frame["label"].nunique() == 2
        }
    return result


def save_checkpoint(
    path: Path,
    *,
    model: PanMHCClassifier,
    config: PanMHCModelConfig,
    training: dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": MODEL_ID,
            "model_config": asdict(config),
            "state_dict": model.state_dict(),
            "training": training,
            "input_contract": {
                "mhc_packed_length": MHC_PACKED_LENGTH,
                "alpha_segment": [0, MHC_ALPHA_MAX_LENGTH],
                "beta_segment": [MHC_ALPHA_MAX_LENGTH, MHC_PACKED_LENGTH],
                "class_I_beta": "padding",
                "species": SPECIES_TO_ID,
            },
        },
        path,
    )
