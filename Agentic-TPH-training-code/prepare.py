"""Prepare training workspaces that share fixed validation and test sets."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
from cli_utils import add_prepare_arguments, load_prepare_config, prepare_overrides
from core_engine.trainer.fixed_splits import (
    apply_validation,
    deduplicate_evaluation,
    select_validation,
)
from core_engine.trainer.increment_workspaces import (
    CapabilityCaps,
    _training_peptide,
    apply_capability_filter,
    cluster_peptides,
)
from core_engine.trainer.workspaces import (
    fixed_negatives,
    identity,
    model_frame,
    save_workspace,
    write_json,
)

GROUPING = {
    "immrep23": "pair_stratified",
    "unseen": "peptide",
    "unseen-cluster": "cluster",
}


def load_input(spec: dict, level: str, positive: bool = False, backend: str = "hla-i"):
    """Read paired records and return supported rows plus exclusion counts.

    Level IV stores variable-domain model inputs separately from original CDR3
    identities used for splitting. Unsupported training rows are excluded;
    unsupported rows in fixed evaluation files raise an error.
    """
    raw = pd.read_csv(spec["path"], dtype=str, keep_default_na=False)
    if spec.get("format", "canonical") == "tph":
        if not positive or spec.get("positive_source") is not True:
            raise ValueError(
                "Unlabelled TPH input requires an explicitly declared positive source"
            )
        frame = pd.DataFrame(
            {
                "pep": _training_peptide(raw),
                "hla": raw["hla.short.seq"],
                "hla.allele": raw["hla.allele"],
            }
        )
        field = "cdr3" if level == "III" else "vseq.reconstructed"
        frame["ab"] = raw["alpha." + field] + "/" + raw["beta." + field]
        frame["cdr3_ab"] = raw["alpha.cdr3"] + "/" + raw["beta.cdr3"]
        frame["label"] = "1"
    else:
        frame = raw.copy()
        if "cdr3_ab" not in frame:
            if level == "III":
                frame["cdr3_ab"] = frame["ab"]
            elif "ab_cdr3" in frame:
                frame["cdr3_ab"] = frame["ab_cdr3"]
            else:
                raise ValueError(
                    "Level IV requires row-local cdr3_ab (or ab_cdr3); no reverse inference"
                )
    required = ["pep", "hla", "hla.allele", "ab", "cdr3_ab", "label"]
    if any(column not in frame for column in required):
        raise ValueError(
            f"Missing columns in {spec['path']}: {set(required) - set(frame.columns)}"
        )
    for column in required:
        frame[column] = frame[column].astype(str).str.strip().str.upper()
    if not frame["label"].isin(["0", "1"]).all():
        raise ValueError("Every input label must be explicitly 0 or 1")
    frame["label"] = frame["label"].astype(int)
    if positive and not (frame.label == 1).all():
        raise ValueError("Training corpus must contain positive observations only")
    if (
        not frame["cdr3_ab"]
        .map(lambda s: len(s.split("/")) == 2 and all(s.split("/")))
        .all()
    ):
        raise ValueError("Explicit paired CDR3 alpha/beta required on every row")
    frame["beta"] = frame["ab"].str.split("/").str[-1]
    if backend == "pan-esm2":
        if level != "III":
            raise ValueError("pan-esm2 requires Level III paired CDR3")
        classes = frame.get("mhc_class", frame.get("mhc.class"))
        if classes is None:
            if spec.get("mhc_class") not in ("I", "II"):
                raise ValueError(
                    "pan-esm2 requires mhc_class I/II per row or input spec"
                )
            classes = pd.Series(spec["mhc_class"], index=frame.index)
        frame["mhc_class"] = classes
        if not classes.isin(["I", "II"]).all():
            raise ValueError("Unknown MHC class")
        pieces = []
        excluded_parts = []
        report = {}
        for mhc_class, group in frame.groupby("mhc_class"):
            kept_part, excluded_part, counts = apply_capability_filter(
                group,
                replace(
                    CapabilityCaps.for_level(level, mhc_class=mhc_class), chain_max=19
                ),
            )
            pieces.append(kept_part)
            excluded_parts.append(excluded_part)
            report[mhc_class] = counts
        kept = pd.concat(pieces, ignore_index=True)
        excluded = pd.concat(excluded_parts, ignore_index=True)
    else:
        kept, excluded, report = apply_capability_filter(
            frame, CapabilityCaps.for_level(level)
        )
    if not positive and len(excluded):
        raise ValueError(f"Fixed evaluation input exceeds model capacity: {report}")
    return kept, report


def prepare(config_path: Path, *, overrides: dict | None = None) -> dict:
    """Prepare datasets with shared validation/test sets; overrides replace JSON settings."""
    config = load_prepare_config(config_path, overrides)
    output_root = Path(config["output"])
    if output_root.exists():
        raise FileExistsError(f"Use a new output directory: {output_root}")
    level = config["level"]
    backend = config.get("backend", "hla-i")
    if backend not in ("hla-i", "pan-esm2"):
        raise ValueError("backend must be hla-i or pan-esm2")
    if level not in ("III", "IV"):
        raise ValueError(
            "This entry supports the documented human HLA-I Level III/IV models"
        )
    grouping = config.get("validation_grouping", "pair_stratified")
    test_grouping = (
        config["test_grouping"]
        if "test_grouping" in config
        else GROUPING[config["task"]]
    )
    corpora, filters = {}, {}
    for name, spec in config["corpora"].items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Corpus names must be simple directory names")
        corpora[name], filters[name] = load_input(
            spec, level, positive=True, backend=backend
        )
    # Exclude the fixed test set from every training source before selecting validation pairs.
    test, _ = load_input(config["test"], level, backend=backend)
    test, test_report = deduplicate_evaluation(identity(test))
    test = model_frame(test)
    universe = sorted(set().union(*(set(f.pep) for f in [*corpora.values(), test])))
    labels = (
        cluster_peptides(universe, 2)
        if "cluster" in (grouping, test_grouping)
        else None
    )
    available = {}
    for name, frame in corpora.items():
        kept, _ = apply_validation(
            identity(frame),
            identity(test),
            grouping=test_grouping,
            cluster_labels=labels,
        )
        available[name] = model_frame(kept)
    reference = config["reference"]
    split_seed = config.get("split_seed", 42)
    negative_seed = config.get("validation_negative_seed", 1042)
    # Group Level IV records by original CDR3s, then restore variable-domain model inputs.
    positive, split_report = select_validation(
        identity(available[reference]),
        peers={name: identity(frame) for name, frame in available.items()},
        grouping=grouping,
        seed=split_seed,
        fraction=config.get("validation_fraction", 0.1),
        cluster_labels=labels,
    )
    positive = model_frame(positive)
    cleaned = {}
    for name, frame in available.items():
        kept, _ = apply_validation(
            identity(frame),
            identity(positive),
            grouping=grouping,
            cluster_labels=labels,
        )
        cleaned[name] = model_frame(kept)
    pool = pd.concat(list(cleaned.values()), ignore_index=True)
    observed = pd.concat([pool, positive, test], ignore_index=True)
    known = {pep: set(group.ab) for pep, group in observed.groupby("pep")}
    cdr_known = {pep: set(group.ab) for pep, group in identity(observed).groupby("pep")}
    negative = fixed_negatives(positive, pool, known, negative_seed, cdr_known)
    valid, valid_report = deduplicate_evaluation(
        identity(pd.concat([positive, negative], ignore_index=True))
    )
    valid = model_frame(valid)
    if set(valid.label) != {0, 1} or set(test.label) != {0, 1}:
        raise ValueError(
            "Validation and fixed test must both contain positive and negative labels"
        )
    for name, frame in cleaned.items():
        save_workspace(
            output_root / name,
            frame,
            valid,
            test,
            dict(
                level=level,
                backend=backend,
                pep_max_len=30 if backend == "pan-esm2" else 13,
                tcr_max_len=19
                if backend == "pan-esm2"
                else (28 if level == "III" else 127),
                task=config["task"],
                grouping=grouping,
                test_grouping=test_grouping,
                reference=reference,
                split_seed=split_seed,
                validation_negative_seed=negative_seed,
                capability_filter=filters[name],
            ),
        )
    write_json(
        output_root / "preparation.json",
        dict(
            config=config,
            validation=split_report,
            validation_dedup=valid_report,
            test_dedup=test_report,
            validation_labels=valid.label.value_counts().to_dict(),
            corpora={name: len(frame) for name, frame in cleaned.items()},
        ),
    )
    if labels is not None:
        write_json(output_root / "cluster_labels.json", labels)
    return {
        "output": str(output_root),
        "corpora": {name: len(frame) for name, frame in cleaned.items()},
        "validation_rows": len(valid),
        "test_rows": len(test),
    }


def main(argv: list[str] | None = None) -> None:
    """Run python prepare.py --config ... as an alternative to train.py prepare."""
    parser = argparse.ArgumentParser(
        description="Prepare fixed-split protein language model workspaces"
    )
    add_prepare_arguments(parser)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare(args.config, overrides=prepare_overrides(args)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
