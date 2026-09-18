from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .campaign_protocol import SPLIT_PROTOCOL
from .fixed_splits import cdr3_group, potential_pair_keys

LEVEL_COLUMNS = ["pep", "hla", "hla.allele", "ab", "label", "beta"]
BLACKLIST_FILENAME = "negative_blacklist_corpus.csv"


def write_json(path, value):
    """Write a JSON-serializable value as a UTF-8 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def identity(frame):
    """Expose source CDR3 identity to split helpers, retaining model input separately."""
    result = frame.copy()
    result["model_ab"] = result["ab"]
    result["ab"] = result.get("split_cdr3_ab", result["cdr3_ab"]).fillna(
        result["cdr3_ab"]
    )
    return result


def model_frame(frame):
    """Restore model input chains while retaining CDR3 identity columns for splitting."""
    result = frame.copy()
    if "cdr3_ab" not in result:
        result["cdr3_ab"] = result["ab"]
    result["ab"] = result["model_ab"]
    result["beta"] = result["ab"].str.split("/").str[-1]
    return result.drop(columns=["model_ab"])


def keys(frame):
    """Return peptide/MHC/paired-TCR group keys derived from the CDR3 identity columns."""
    return set(potential_pair_keys(identity(frame)))


def fixed_negatives(positives, pool, blacklist, seed, cdr3_blacklist):
    """Draw one TCR per positive from the pool after applying peptide/TCR exclusions.

    Draws are uniform and independent. Evaluation deduplication is performed
    by the caller after positives and generated negatives are combined.
    """
    rng = np.random.RandomState(seed)
    frames = []
    source = pool[["ab", "cdr3_ab"]].drop_duplicates().reset_index(drop=True)
    for peptide, group in positives.groupby("pep", sort=True):
        excluded = blacklist.get(peptide, set())
        admissible = source.loc[
            ~source.ab.isin(excluded)
            & ~source.cdr3_ab.map(cdr3_group).isin(
                {cdr3_group(cdr) for cdr in cdr3_blacklist.get(peptide, set())}
            )
        ].reset_index(drop=True)
        if admissible.empty:
            raise ValueError(f"No admissible negative TCR for {peptide}")
        frame = group[LEVEL_COLUMNS].copy()
        draw = rng.randint(len(admissible), size=len(group))
        frame["ab"] = admissible.iloc[draw].ab.to_numpy()
        frame["cdr3_ab"] = admissible.iloc[draw].cdr3_ab.to_numpy()
        frame["beta"] = frame["ab"].str.split("/").str[-1]
        frame["label"] = 0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def negative_exclusion_pairs(train, valid, test):
    """Expand held-out pair groups to actual train-pool model sequences.

    The existing sampler is peptide-only, so this exclusion conservatively
    spans MHC alleles. All held-out labels are excluded without using outcomes.
    """
    observations = pd.concat([train, valid, test], ignore_index=True)
    held = identity(observations)
    blocked = pd.DataFrame(
        {"pep": held.pep, "group": held.ab.map(cdr3_group)}
    ).drop_duplicates()
    pool = pd.DataFrame(
        {"ab": train.ab, "group": identity(train).ab.map(cdr3_group)}
    ).drop_duplicates()
    expanded = blocked.merge(pool, on="group")[["pep", "ab"]]
    exact = observations[["pep", "ab"]]
    return pd.concat([exact, expanded], ignore_index=True).drop_duplicates()


def save_workspace(path, train, valid, test, report):
    """Check split isolation and save data, candidate pools, exclusions and input settings."""
    if not len(train) or not len(valid):
        raise ValueError("empty training or validation data")
    if (
        keys(train) & keys(valid)
        or keys(train) & keys(test)
        or keys(valid) & keys(test)
    ):
        raise ValueError(
            f"potential CDR3 pair overlaps: train/valid={len(keys(train) & keys(valid))}, train/test={len(keys(train) & keys(test))}, valid/test={len(keys(valid) & keys(test))}"
        )
    path.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("valid", valid), ("test", test)):
        frame.to_csv(path / f"{name}_data_fold0.csv", index=False)
    pool = sorted(set(train.ab))
    np.save(path / "tcr2candidates_pools.npy", np.asarray(pool), allow_pickle=False)
    negative_exclusion_pairs(train, valid, test).to_csv(
        path / BLACKLIST_FILENAME, index=False
    )
    report["negative_exclusion"] = (
        "train positives + all validation/test pair groups; expanded to model pool; peptide-only across MHC"
    )
    report.update(
        split_protocol=SPLIT_PROTOCOL,
        train_rows=len(train),
        train_unique_pairs=len(keys(train)),
        valid_rows=len(valid),
        test_rows=len(test),
        train_peptides=int(train.pep.nunique()),
        training_duplicates_retained=len(train) - len(keys(train)),
    )
    write_json(path / "workspace_report.json", report)
