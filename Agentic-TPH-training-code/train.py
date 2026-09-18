#!/usr/bin/env python3
"""Train protein language models, evaluate labeled data and predict unlabeled records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from cli_utils import (
    add_inference_arguments,
    add_prepare_arguments,
    nonnegative_int,
    positive_float,
    positive_int,
    prepare_overrides,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the data preparation, training, evaluation and inference interfaces."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preparation = commands.add_parser(
        "prepare", help="Prepare a training workspace with fixed dataset splits"
    )
    add_prepare_arguments(preparation)

    for command in ("train", "eval"):
        description = (
            "Train a model and save the best checkpoint"
            if command == "train"
            else "Compute metrics on the fixed labeled test set"
        )
        subparser = commands.add_parser(
            command,
            help=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        subparser.add_argument(
            "--workspace",
            type=Path,
            required=True,
            help="Single-dataset directory created by prepare, containing workspace_report.json",
        )
        subparser.add_argument(
            "--asset-root",
            type=Path,
            required=True,
            help="Base model asset directory containing manifest.json",
        )
        subparser.add_argument(
            "--model-id",
            default="esm2-150M",
            help="Model name in the asset manifest; must match the checkpoint for evaluation",
        )
        subparser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Seed for initialization and runtime randomness; does not change data splits",
        )
        subparser.add_argument(
            "--device", default="cuda", help="Compute device: cuda, cuda:0 or cpu"
        )
        subparser.add_argument(
            "--precision",
            default="bf16",
            choices=("bf16", "fp32", "tf32"),
            help="Compute precision; bf16 enables automatic mixed precision on CUDA",
        )
        subparser.add_argument(
            "--num-workers",
            type=nonnegative_int,
            default=0,
            help="DataLoader subprocess count; 0 loads data in the main process",
        )
        subparser.add_argument(
            "--batch-size",
            type=positive_int,
            help="Positive training examples per GPU batch; defaults: III=24, IV=8, pan-ESM2=32",
        )
        if command == "train":
            subparser.add_argument(
                "--negative-seed",
                type=int,
                help="Seed for dynamic negative sampling; defaults to --seed",
            )
            subparser.add_argument(
                "--output",
                type=Path,
                required=True,
                help="Output directory for checkpoints, validation thresholds and training_result.json",
            )
            subparser.add_argument(
                "--epochs",
                type=positive_int,
                default=100,
                help="Maximum number of training epochs",
            )
            subparser.add_argument(
                "--patience",
                type=positive_int,
                help="Epochs without validation AUROC improvement before stopping; defaults: III/IV=4, pan-ESM2=5",
            )
            subparser.add_argument(
                "--learning-rate",
                type=positive_float,
                default=8e-5,
                help="Adam learning rate",
            )
            subparser.add_argument(
                "--gradient-accumulation-steps",
                type=positive_int,
                help="Gradient accumulation steps; defaults: III=2, IV=6, pan-ESM2=1",
            )
        else:
            subparser.add_argument(
                "--checkpoint",
                type=Path,
                required=True,
                help="Best training checkpoint; evaluation uses fixed test labels without drawing negatives",
            )

    inference = commands.add_parser(
        "inference",
        help="Predict a new CSV without labels or a training workspace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_inference_arguments(inference, native=True)
    return parser


def training_defaults(metadata: dict[str, Any]) -> dict[str, int]:
    """Choose workspace-specific training defaults; explicit command-line values take precedence."""
    if metadata.get("backend") == "pan-esm2":
        return {"batch_size": 32, "patience": 5, "gradient_accumulation_steps": 1}
    if metadata["level"] == "III":
        return {"batch_size": 24, "patience": 4, "gradient_accumulation_steps": 2}
    if metadata["level"] == "IV":
        return {"batch_size": 8, "patience": 4, "gradient_accumulation_steps": 6}
    raise ValueError(f"Unsupported workspace level: {metadata['level']}")


def run_workspace(args: argparse.Namespace) -> dict[str, Any]:
    """Train or evaluate using the sequence widths recorded in the prepared workspace."""
    from core_engine.trainer.campaign_protocol import require_current_split
    from core_engine.trainer.native import (
        EvalConfig,
        NativeConfig,
        Precision,
        TrainConfig,
    )
    from core_engine.trainer.native.loop import evaluate, train

    metadata = json.loads(
        (args.workspace / "workspace_report.json").read_text(encoding="utf-8")
    )
    require_current_split(metadata)
    defaults = training_defaults(metadata)
    level = {"III": "3", "IV": "4"}[metadata["level"]]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    batch_size = (
        args.batch_size if args.batch_size is not None else defaults["batch_size"]
    )

    # Sequence widths come from prepare and do not change with the device or batch size.
    base = NativeConfig(
        model_id=args.model_id,
        level=level,
        data_dir=args.workspace,
        asset_root=args.asset_root,
        batch_size=batch_size,
        pep_max_len=metadata.get("pep_max_len", 13),
        tcr_max_len=metadata.get("tcr_max_len", 28 if level == "3" else 127),
        hla_max_len=34,
        seed=args.seed,
        negative_seed=getattr(args, "negative_seed", None),
        precision=Precision(args.precision),
        device=args.device,
        num_workers=args.num_workers,
        distributed=world_size > 1,
        nproc_per_node=world_size,
    )
    blacklist = args.workspace / "negative_blacklist_corpus.csv"
    if not blacklist.is_file():
        raise FileNotFoundError(blacklist)

    if args.command == "eval":
        return evaluate(
            EvalConfig(
                base=base,
                checkpoint=args.checkpoint,
                splits=("test",),
                redraw_negatives=False,
            )
        )

    if (args.output / "training_result.json").exists():
        raise FileExistsError(f"Training result already exists: {args.output}")
    patience = args.patience if args.patience is not None else defaults["patience"]
    accumulation = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else defaults["gradient_accumulation_steps"]
    )
    config = TrainConfig(
        base=base,
        output_dir=args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stop=patience,
        gradient_accumulation_steps=accumulation,
        fixed_validation=True,
        negative_blacklist=blacklist,
    )
    return train(config).to_dict()


def main(argv: list[str] | None = None) -> None:
    """Run the selected command; only rank 0 prints results under torchrun."""
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        from prepare import prepare

        result = prepare(args.config, overrides=prepare_overrides(args))
    elif args.command == "inference":
        from inference import predict_native

        result = predict_native(
            input_path=args.input,
            checkpoint=args.checkpoint,
            asset_root=args.asset_root,
            output_path=args.output,
            batch_size=args.batch_size,
            device=args.device,
            precision=args.precision,
            threshold=args.threshold,
        )
    else:
        result = run_workspace(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
