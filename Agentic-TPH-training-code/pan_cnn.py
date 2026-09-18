#!/usr/bin/env python3
"""Prepare data, train domain combinations and predict with a pan-MHC sequence CNN."""

from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path

import pandas as pd
from cli_utils import (
    add_inference_arguments,
    add_prepare_arguments,
    load_prepare_config,
    positive_float,
    positive_int,
    prepare_overrides,
)
from core_engine.trainer.campaign_protocol import SPLIT_PROTOCOL, require_current_split
from core_engine.trainer.fixed_splits import (
    apply_validation,
    cdr3_group,
    deduplicate_evaluation,
    potential_pair_keys,
    select_validation,
)
from core_engine.trainer.increment_workspaces import cluster_peptides
from core_engine.trainer.workspaces import write_json


def read_frame(path: Path, positive: bool = False) -> pd.DataFrame:
    """Read resolved MHC chains and paired CDR3s; verify domain, species and MHC class agree."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = [
        "peptide",
        "allele_key",
        "tcr_alpha_cdr3",
        "tcr_beta_cdr3",
        "mhc_alpha_seq",
        "mhc_beta_seq",
        "species",
        "mhc_class",
        "domain",
        "label",
    ]
    if any(c not in frame for c in required):
        raise ValueError(f"Missing pan fields: {set(required) - set(frame.columns)}")
    for c in [
        "peptide",
        "allele_key",
        "tcr_alpha_cdr3",
        "tcr_beta_cdr3",
        "mhc_alpha_seq",
        "mhc_beta_seq",
        "mhc_class",
    ]:
        frame[c] = frame[c].str.strip().str.upper()
    if not frame.label.isin(["0", "1"]).all():
        raise ValueError("Explicit binary labels required")
    frame["label"] = frame.label.astype(int)
    if positive and not (frame.label == 1).all():
        raise ValueError("Source must contain positive observations only")
    if (
        not frame.species.isin(["human", "mouse", "other"]).all()
        or not frame.mhc_class.isin(["I", "II"]).all()
    ):
        raise ValueError("Unknown species/class")
    expected = (
        frame.species.map(lambda x: "human" if x == "human" else "nonhuman")
        + "_"
        + frame.mhc_class
    )
    if not (expected == frame.domain).all():
        raise ValueError("domain does not match species and MHC class")
    for col, maximum, allow_empty in [
        ("peptide", 30, False),
        ("tcr_alpha_cdr3", 40, False),
        ("tcr_beta_cdr3", 40, False),
        ("mhc_alpha_seq", 400, False),
        ("mhc_beta_seq", 300, True),
    ]:
        if (
            not frame[col]
            .map(
                lambda s: (
                    len(s) <= maximum
                    and (allow_empty or bool(s))
                    and set(s) <= set("ACDEFGHIKLMNPQRSTVWY")
                )
            )
            .all()
        ):
            raise ValueError(f"Invalid sequence or length in {col}")
    if ((frame.mhc_class == "II") & (frame.mhc_beta_seq == "")).any():
        raise ValueError("MHC-II requires both explicit chain sequences")
    frame["pep"] = frame.peptide
    frame["hla.allele"] = (
        frame.species + "|" + frame.mhc_class + "|" + frame.allele_key
    ).str.upper()
    frame["ab"] = frame.tcr_alpha_cdr3 + "/" + frame.tcr_beta_cdr3
    frame["pair_key"] = frame["hla.allele"] + "|" + frame.pep + "|" + frame.ab
    return frame


def negatives(
    positive: pd.DataFrame, pool: pd.DataFrame, blocked: set, seed: int
) -> pd.DataFrame:
    """Fixed 1:1 same-domain mismatch; peptide/TCR blacklist spans MHC alleles."""
    blocked = {(str(pep).strip().upper(), group) for pep, _, group in blocked}
    rng = random.Random(seed)
    cache = {}
    rows = []
    for domain, group in positive.groupby("domain", sort=True):
        donors = list(
            pool.loc[pool.domain == domain, ["tcr_alpha_cdr3", "tcr_beta_cdr3"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if not donors:
            raise ValueError(f"No training TCR donors in {domain}")
        for row in group.to_dict("records"):
            key = (domain, row["pep"], row["hla.allele"])
            if key not in cache:
                cache[key] = [
                    (a, b)
                    for a, b in donors
                    if (row["pep"].strip().upper(), cdr3_group(a + "/" + b))
                    not in blocked
                ]
            admissible = cache[key]
            if not admissible:
                raise ValueError(f"No admissible negative for {row['pair_key']}")
            a, b = rng.choice(admissible)
            item = dict(row)
            item.update(
                tcr_alpha_cdr3=a,
                tcr_beta_cdr3=b,
                ab=a + "/" + b,
                label=0,
                pair_key=row["hla.allele"] + "|" + row["pep"] + "|" + a + "/" + b,
                negative_type="derived_random_mismatch",
            )
            if "tcr_species" in item:
                item["tcr_species"] = ""  # donor species metadata is not inferred
            rows.append(item)
    return pd.DataFrame(rows)


def prepare(config_path: Path, *, overrides: dict | None = None) -> dict:
    """Write fixed train/validation/test files, retaining training pairs per peptide by default."""
    config = load_prepare_config(config_path, overrides)
    output_root = Path(config["output"])
    if output_root.exists():
        raise FileExistsError(output_root)
    source = read_frame(config["source"], positive=True)
    test, test_report = deduplicate_evaluation(read_frame(config["test"]))
    grouping = config.get("validation_grouping", "pair_stratified")
    test_grouping = config.get("test_grouping", "pair_stratified")
    labels = (
        cluster_peptides(sorted(set(source.pep) | set(test.pep)), 2)
        if "cluster" in (grouping, test_grouping)
        else None
    )
    available, _ = apply_validation(
        source, test, grouping=test_grouping, cluster_labels=labels
    )
    # Each compared training domain is a peer, so a validation draw preserves
    # its available peptide whenever there is more than one admissible pair.
    peers = {d: g for d, g in available.groupby("domain")}
    valid_positive, split_report = select_validation(
        available,
        peers=peers,
        grouping=grouping,
        seed=config.get("split_seed", 42),
        fraction=config.get("validation_fraction", 0.1),
        cluster_labels=labels,
    )
    train, _ = apply_validation(
        available, valid_positive, grouping=grouping, cluster_labels=labels
    )
    blocked = set(potential_pair_keys(pd.concat([source, test], ignore_index=True)))
    val_negative = negatives(
        valid_positive, train, blocked, config.get("validation_negative_seed", 1042)
    )
    valid, valid_report = deduplicate_evaluation(
        pd.concat([valid_positive, val_negative], ignore_index=True)
    )
    blocked.update(potential_pair_keys(valid))
    train_negative = negatives(
        train, train, blocked, config.get("training_negative_seed", 42)
    )
    combined = pd.concat([train, train_negative], ignore_index=True)
    train_keys = set(potential_pair_keys(combined))
    valid_keys = set(potential_pair_keys(valid))
    test_keys = set(potential_pair_keys(test))
    overlaps = {
        "train_valid": len(train_keys & valid_keys),
        "train_test": len(train_keys & test_keys),
        "valid_test": len(valid_keys & test_keys),
    }
    if any(overlaps.values()):
        raise ValueError(f"Split leakage: {overlaps}")
    for name, frame in [("train", combined), ("valid", valid), ("test", test)]:
        if set(frame.domain) != set(source.domain):
            raise ValueError(
                f"{name} missing a source domain; choose a feasible fraction/reference"
            )
        if any(set(g.label) != {0, 1} for _, g in frame.groupby("domain")):
            raise ValueError(f"{name} needs both labels within every domain")
    output_root.mkdir(parents=True)
    combined.to_csv(output_root / "train_all.csv", index=False)
    valid.to_csv(output_root / "valid_all.csv", index=False)
    for domain, frame in test.groupby("domain"):
        frame.to_csv(output_root / f"test_{domain}.csv", index=False)
    report = dict(
        split_protocol=SPLIT_PROTOCOL,
        config=config,
        validation=split_report,
        validation_dedup=valid_report,
        test_dedup=test_report,
        overlaps=overlaps,
        negative_exclusion="peptide/endpoint-TCR across MHC; donors within domain",
        domains=sorted(set(source.domain)),
        train_rows=len(combined),
        valid_rows=len(valid),
        test_rows=len(test),
    )
    write_json(output_root / "workspace_report.json", report)
    return report


DOMAINS = ("human_I", "human_II", "nonhuman_I", "nonhuman_II")
RUN_NAMES = (
    DOMAINS
    + tuple(f"{left}+{right}" for left, right in combinations(DOMAINS, 2))
    + ("all_four",)
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CNN data preparation, training and unlabeled prediction interfaces."""
    parser = argparse.ArgumentParser(
        description="Train and predict with a pan-MHC chain-sequence CNN"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser(
        "prepare",
        help="Prepare a workspace from positive records and a fixed labeled test CSV",
    )
    add_prepare_arguments(preparation)
    preparation.add_argument(
        "--training-negative-seed",
        type=int,
        help="Random seed for fixed training negatives; JSON default: 42",
    )
    training = commands.add_parser(
        "train",
        help="Train selected domain combinations and evaluate all test domains in the workspace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    training.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Prepared directory containing train_all.csv, valid_all.csv and workspace_report.json",
    )
    training.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory for model checkpoints, predictions and summary.json",
    )
    training.add_argument(
        "--runs",
        nargs="+",
        choices=("all", *RUN_NAMES),
        default=["all"],
        help="Training combinations: all runs all 11; all_four trains only the four-domain model",
    )
    training.add_argument(
        "--device", default="cuda:0", help="Compute device, such as cuda:0 or cpu"
    )
    training.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for model initialization and per-epoch balanced sampling; does not change data splits",
    )
    training.add_argument(
        "--epochs",
        type=positive_int,
        default=8,
        help="Maximum number of training epochs",
    )
    training.add_argument(
        "--patience",
        type=positive_int,
        default=2,
        help="Stop after this many epochs without improvement in mean validation AUPRC across domains",
    )
    training.add_argument(
        "--batch-size",
        type=positive_int,
        default=512,
        help="Total records per batch, including positive and negative examples",
    )
    training.add_argument(
        "--learning-rate", type=positive_float, default=8e-4, help="AdamW learning rate"
    )
    inference_parser = commands.add_parser(
        "inference",
        help="Predict records from a CSV without requiring labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_inference_arguments(inference_parser, native=False)
    return parser


def run_training(args: argparse.Namespace) -> dict:
    """Train selected combinations; evaluate each model on all test domains in the workspace."""
    import torch
    from core_engine.trainer.pan_mhc_matrix_runtime import run_specs, train_one

    report = json.loads(
        (args.workspace / "workspace_report.json").read_text(encoding="utf-8")
    )
    require_current_split(report)

    def load_split(filename: str) -> pd.DataFrame:
        """Use saved training/evaluation labels without generating new negative examples."""
        frame = pd.read_csv(args.workspace / filename, keep_default_na=False)
        frame["label"] = frame.label.astype(int)
        return frame

    train = load_split("train_all.csv")
    valid = load_split("valid_all.csv")
    test = {domain: load_split(f"test_{domain}.csv") for domain in report["domains"]}
    specifications = run_specs()
    if "all" in args.runs and args.runs != ["all"]:
        raise ValueError("--runs all must be used alone")
    if args.runs != ["all"]:
        specifications = [
            (name, domains) for name, domains in specifications if name in args.runs
        ]
    if any(set(domains) - set(report["domains"]) for _, domains in specifications):
        raise ValueError(
            "Requested run needs absent domains; use --runs for an available subset"
        )
    if args.output.exists():
        raise FileExistsError(args.output)

    results = []
    for name, domains in specifications:
        result = train_one(
            name=name,
            domains=domains,
            train_all=train,
            valid_all=valid,
            unseen=test,
            output_root=args.output,
            device=torch.device(args.device),
            seed=args.seed,
            max_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        results.append(result)
    summary = {"runs": results, "split_protocol": SPLIT_PROTOCOL}
    write_json(args.output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run the selected command and print a JSON summary."""
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        overrides = prepare_overrides(args)
        if args.training_negative_seed is not None:
            overrides["training_negative_seed"] = args.training_negative_seed
        result = prepare(args.config, overrides=overrides)
    elif args.command == "inference":
        from inference import predict_cnn

        result = predict_cnn(
            input_path=args.input,
            checkpoint=args.checkpoint,
            output_path=args.output,
            batch_size=args.batch_size,
            device=args.device,
            threshold=args.threshold,
        )
    else:
        result = run_training(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
