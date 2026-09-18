"""Predict unlabeled CSV records while preserving input fields and row order."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cli_utils import probability

PREDICTION_COLUMNS = ("score", "predicted_label", "threshold")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_inputs(path: Path, required: list[str]) -> pd.DataFrame:
    """Read prediction inputs; an optional label column is not used for scoring or threshold selection."""
    frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",", dtype=str, keep_default_na=False)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing inference columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Inference input must contain at least one row")
    collisions = set(PREDICTION_COLUMNS) & set(frame.columns)
    if collisions:
        raise ValueError(f"Output columns already exist in input: {sorted(collisions)}")
    return frame


def write_predictions(
    frame: pd.DataFrame, scores: list[float], threshold: float, output_path: Path
) -> dict[str, Any]:
    """Append positive-class scores and thresholded labels without changing input fields."""
    if output_path.exists():
        raise FileExistsError(output_path)
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError(
            "Prediction scores must be finite and match the input row count"
        )
    result = frame.copy()
    result["score"] = scores
    result["predicted_label"] = (np.asarray(scores) >= threshold).astype(int)
    result["threshold"] = threshold
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return {"output": str(output_path), "rows": len(result), "threshold": threshold}


def predict_native(
    *,
    input_path: Path,
    checkpoint: Path,
    asset_root: Path,
    output_path: Path,
    batch_size: int = 32,
    device: str = "cuda",
    precision: str = "bf16",
    threshold: float | None = None,
) -> dict[str, Any]:
    """Predict pep/hla/ab records using a protein language model checkpoint.

    ab contains alpha/beta CDR3 sequences for Level III or variable domains for
    Level IV. Model type, sequence widths and input concatenation settings are
    restored from the checkpoint. No labels or training workspace are required.
    """
    import torch
    from core_engine.trainer.native.config import Precision
    from core_engine.trainer.native.encoding import (
        contract_for,
        encode_batch,
        pad_record,
    )
    from core_engine.trainer.native.loop import (
        apply_precision,
        load_checkpoint,
        threshold_sidecar,
    )
    from core_engine.trainer.native.models import build_model, load_tokenizer

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if output_path.exists():
        raise FileExistsError(output_path)
    payload = load_checkpoint(checkpoint, map_location="cpu")
    required_metadata = (
        "model_id",
        "level",
        "pep_max_len",
        "tcr_max_len",
        "hla_max_len",
    )
    if any(key not in payload for key in required_metadata):
        raise ValueError(
            "Inference requires a checkpoint with model/input metadata from this package"
        )
    if payload["level"] not in ("3", "4"):
        raise ValueError("This inference interface supports Level III/IV checkpoints")
    if threshold is None:
        sidecar = threshold_sidecar(checkpoint)
        if not sidecar.is_file():
            raise FileNotFoundError(
                f"Missing validation threshold: {sidecar}; provide --threshold explicitly"
            )
        threshold = json.loads(sidecar.read_text(encoding="utf-8"))["threshold"]
    threshold = probability(str(threshold))

    frame = read_inputs(input_path, ["pep", "hla", "ab"])
    records = frame[["pep", "hla", "ab"]].apply(lambda col: col.str.strip().str.upper())
    for index, row in records.iterrows():
        chains = row["ab"].split("/")
        if len(chains) != 2 or not all(chains):
            raise ValueError(
                f"CSV row {index + 2}: ab must contain explicit alpha/beta chains"
            )
        if not all(
            sequence and set(sequence) <= STANDARD_AA
            for sequence in [row["pep"], *chains]
        ):
            raise ValueError(
                f"CSV row {index + 2}: sequences must use standard amino-acid letters"
            )
        # MHC pseudo-sequences allow X for unknown residue slots, as in training.
        if set(row["hla"]) - (STANDARD_AA | {"X"}):
            raise ValueError(
                f"CSV row {index + 2}: invalid MHC pseudo-sequence residue"
            )
        if len(row["hla"]) != payload["hla_max_len"]:
            raise ValueError(
                f"CSV row {index + 2}: MHC pseudo-sequence width must match checkpoint"
            )

    contract = contract_for(
        payload["level"],
        pep_max_len=payload["pep_max_len"],
        tcr_max_len=payload["tcr_max_len"],
        hla_max_len=payload["hla_max_len"],
    )
    # Reuse training padding and tokenization; reject overlength sequences.
    padded = [pad_record(row, contract) for row in records.to_dict("records")]
    apply_precision(Precision(precision))
    tokenizer = load_tokenizer(payload["model_id"], asset_root)
    model = build_model(
        payload["model_id"],
        asset_root,
        head_type=payload.get("head_type", "3MLP"),
        plm_output=payload.get("plm_output", "cls"),
        finetune=payload.get("finetune", True),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(padded), batch_size):
            tcr, pmhc = zip(*padded[start : start + batch_size])
            tokens = encode_batch(
                tokenizer,
                tcr,
                pmhc,
                model_id=payload["model_id"],
                plm_input=payload.get("plm_input", "cat"),
                device=device,
            )
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if precision == "bf16" and device.startswith("cuda")
                else nullcontext()
            )
            with context:
                logits = model(tokens)
            scores.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    return write_predictions(frame, scores, threshold, output_path)


def predict_cnn(
    *,
    input_path: Path,
    checkpoint: Path,
    output_path: Path,
    batch_size: int = 512,
    device: str = "cuda",
    threshold: float | None = None,
) -> dict[str, Any]:
    """Predict with the chain-sequence CNN without labels, allele keys or pretrained assets."""
    import torch
    from core_engine.trainer.pan_mhc_matrix_data import (
        PEPTIDE_MAX_LENGTH,
        TCR_MAX_LENGTH,
    )
    from core_engine.trainer.pan_mhc_matrix_model import (
        MODEL_ID,
        SPECIES_TO_ID,
        PanMHCClassifier,
        PanMHCModelConfig,
        encode_packed_mhc,
        encode_strict,
    )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if output_path.exists():
        raise FileExistsError(output_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_id") != MODEL_ID:
        raise ValueError("Expected a pan-MHC CNN checkpoint")
    if threshold is None:
        threshold = payload.get("training", {}).get("decision_threshold")
        if threshold is None:
            raise ValueError(
                "CNN checkpoint has no validation threshold; provide --threshold"
            )
    threshold = probability(str(threshold))
    fields = [
        "peptide",
        "mhc_alpha_seq",
        "mhc_beta_seq",
        "tcr_alpha_cdr3",
        "tcr_beta_cdr3",
        "species",
        "mhc_class",
    ]
    frame = read_inputs(input_path, fields)
    records = frame[fields].copy()
    for column in fields:
        records[column] = records[column].str.strip()
    for column in fields:
        if column != "species":
            records[column] = records[column].str.upper()
    if (
        not records.species.isin(SPECIES_TO_ID).all()
        or not records.mhc_class.isin(["I", "II"]).all()
    ):
        raise ValueError("species must be human/mouse/other and mhc_class must be I/II")
    for column in ("peptide", "mhc_alpha_seq", "tcr_alpha_cdr3", "tcr_beta_cdr3"):
        if records[column].eq("").any():
            raise ValueError(f"Empty inference sequence: {column}")
    if ((records.mhc_class == "II") & records.mhc_beta_seq.eq("")).any():
        raise ValueError("MHC-II requires both explicit MHC chain sequences")

    if ((records.mhc_class == "I") & records.mhc_beta_seq.ne("")).any():
        raise ValueError("MHC-I requires an empty mhc_beta_seq (padding slot)")

    model = PanMHCClassifier(PanMHCModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            rows = records.iloc[start : start + batch_size].to_dict("records")
            # Build only model inputs; do not invent labels for unlabeled records.
            batch = {
                "peptide": encode_strict(
                    [row["peptide"] for row in rows], max_length=PEPTIDE_MAX_LENGTH
                ),
                "mhc_packed": encode_packed_mhc(rows),
                "tcr_alpha": encode_strict(
                    [row["tcr_alpha_cdr3"] for row in rows], max_length=TCR_MAX_LENGTH
                ),
                "tcr_beta": encode_strict(
                    [row["tcr_beta_cdr3"] for row in rows], max_length=TCR_MAX_LENGTH
                ),
                "species": torch.tensor(
                    [SPECIES_TO_ID[row["species"]] for row in rows]
                ),
                "mhc_class": torch.tensor(
                    [0 if row["mhc_class"] == "I" else 1 for row in rows]
                ),
            }
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            scores.extend(torch.sigmoid(model(batch)).cpu().tolist())
    return write_predictions(frame, scores, threshold, output_path)
