"""Load base models and classification heads from local assets.

asset_root/manifest.json maps model names to local directories.
Loading does not download model weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from core_engine.trainer.runtime import (
    TrainerPreflightError,
    load_asset_manifest,
    validate_model_asset,
)

HITPH_HIDDEN_SIZES: Mapping[str, int] = {
    "esm2-8M": 320,
    "esm2-35M": 480,
    "esm2-150M": 640,
    "esm2-650M": 1280,
    "tape": 768,
    "protbert": 1024,
    "AMPLIFY-120M-base": 640,
    "AMPLIFY-120M": 640,
    "AMPLIFY-350M": 960,
}

HEAD_LAYERS: Mapping[str, list[int]] = {
    "3MLP": [256, 64, 2],
    "5MLP": [1024, 512, 128, 32, 2],
}


class TrainerMLP(nn.Module):
    """Classification head with Linear/ReLU/BatchNorm hidden layers and a linear output."""

    def __init__(
        self, in_dim: int, layers: list[int], *, batch_norm: bool = True
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for index, out_dim in enumerate(layers):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.ReLU())
            if batch_norm and index != len(layers) - 1:
                modules.append(nn.BatchNorm1d(out_dim))
            in_dim = out_dim
        self.layers = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


MLP = TrainerMLP


class SequenceClassifier(nn.Module):
    """One protein language model plus the classification head."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        *,
        head_type: str = "3MLP",
        plm_output: str = "cls",
        finetune: bool = True,
        amplify: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.plm_output = plm_output
        self.amplify = amplify
        if not finetune:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        try:
            layers = HEAD_LAYERS[head_type]
        except KeyError as exc:
            raise ValueError(f"unsupported head_type: {head_type!r}") from exc
        self.projection = TrainerMLP(hidden_size, list(layers))

    def _hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.amplify:
            return self.encoder(input_ids, output_hidden_states=True).hidden_states[-1]
        outputs = self.encoder(input_ids)
        return outputs[0]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self._hidden(input_ids)
        pooled = hidden.mean(dim=1) if self.plm_output == "mean" else hidden[:, 0]
        logits = self.projection(pooled)
        return logits.view(-1, logits.size(-1))


def _asset_root(asset_root: Path, model_id: str) -> Path:
    manifest = load_asset_manifest(_ConfigShim(asset_root))
    return Path(
        validate_model_asset(_ConfigShim(asset_root), model_id, manifest)["root"]
    )


@dataclass(frozen=True, slots=True)
class _ConfigShim:
    """Only the field the asset validators read."""

    asset_root: Path

    @property
    def manifest_path(self) -> Path:
        return self.asset_root / "manifest.json"


def load_tokenizer(model_id: str, asset_root: Path) -> Any:
    root = _asset_root(Path(asset_root), model_id)
    if model_id == "tape":
        from tape import TAPETokenizer

        return TAPETokenizer(vocab="iupac")
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"local_files_only": True}
    if model_id == "protbert":
        kwargs["do_lower_case"] = False
    return AutoTokenizer.from_pretrained(str(root), **kwargs)


def build_model(
    model_id: str,
    asset_root: Path,
    *,
    head_type: str = "3MLP",
    plm_output: str = "cls",
    finetune: bool = True,
) -> SequenceClassifier:
    """Instantiate one backbone from its declared, size-verified local assets."""

    try:
        hidden_size = HITPH_HIDDEN_SIZES[model_id]
    except KeyError as exc:
        raise TrainerPreflightError(
            "UNSUPPORTED_TRAINER_BACKEND",
            "no native backbone is registered for this model",
            model_id=model_id,
        ) from exc
    root = _asset_root(Path(asset_root), model_id)
    if model_id == "tape":
        from tape import ProteinBertModel

        encoder = ProteinBertModel.from_pretrained(str(root))
        return SequenceClassifier(
            encoder,
            hidden_size,
            head_type=head_type,
            plm_output=plm_output,
            finetune=finetune,
        )
    from transformers import AutoModel

    amplify = "AMPLIFY" in model_id
    encoder = AutoModel.from_pretrained(
        str(root),
        local_files_only=True,
        **({"trust_remote_code": True} if amplify else {}),
    )
    return SequenceClassifier(
        encoder,
        hidden_size,
        head_type=head_type,
        plm_output=plm_output,
        finetune=finetune,
        amplify=amplify,
    )
