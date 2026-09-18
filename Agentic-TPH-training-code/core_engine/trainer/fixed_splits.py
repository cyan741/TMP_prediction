"""Deterministic validation blocks shared by Trainer corpus comparisons.

The functions in this module operate on positive observation frames only.  They
do not write files and never infer a TCR pairing from row order or co-
occurrence.  Raw CDR3 strings are retained. Split/evaluation groups conservatively ignore
terminal C/F/W notation differences, without asserting biological identity.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Hashable, Iterable, Mapping

import numpy as np
import pandas as pd

from .increment_workspaces import KEY_COLUMNS
from .increment_workspaces import _keys_iter as _exact_keys_iter

GROUPINGS = ("pair_stratified", "peptide", "cluster")


def cdr3_group(value: str) -> str:
    """Conservative endpoint-equivalence group, never a model/source sequence.

    Strip leading C and trailing F/W per chain to closure so the key is
    idempotent, including ambiguous endpoint runs. Internal residues and chain
    order are unchanged. This deliberately groups possible duplicates.
    """
    return "/".join(
        chain.strip().upper().lstrip("C").rstrip("FW")
        for chain in str(value).split("/")
    )


def potential_pair_keys(frame: pd.DataFrame):
    for peptide, mhc, ab in _exact_keys_iter(frame):
        yield peptide, mhc, cdr3_group(ab)


# potential_pair_keys combines peptide, MHC identity and endpoint-grouped TCRs.
# Use these keys for split isolation, never to rewrite model input sequences.
_keys_iter = potential_pair_keys


def _require_columns(frame: pd.DataFrame, *, name: str) -> None:
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalise(frame: pd.DataFrame, *, name: str = "frame") -> pd.DataFrame:
    _require_columns(frame, name=name)
    result = frame.copy()
    for column in KEY_COLUMNS:
        result[column] = result[column].astype(str).str.strip().str.upper()
    return result


def _unique_keys(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    return list(dict.fromkeys(_keys_iter(frame)))


def _keys_by_peptide(frame: pd.DataFrame) -> dict[str, set[tuple[str, str, str]]]:
    result: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for key in _keys_iter(frame):
        result[key[0]].add(key)
    return dict(result)


def _validate_args(grouping: str, fraction: float) -> None:
    if grouping not in GROUPINGS:
        raise ValueError(
            f"unsupported grouping: {grouping!r}; expected one of {GROUPINGS}"
        )
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")


def _check_cluster_labels(
    frames: Iterable[pd.DataFrame], labels: Mapping[str, Hashable] | None
) -> dict[str, Hashable]:
    if labels is None:
        raise ValueError("cluster_labels is required for grouping='cluster'")
    labels = dict(labels)
    peptides = {key[0] for frame in frames for key in _keys_iter(frame)}
    missing = sorted(peptides - labels.keys())
    if missing:
        raise ValueError(f"cluster_labels does not cover peptides: {missing}")
    return labels


def _choose_group_labels(
    sizes: Mapping[Hashable, int], *, fraction: float, seed: int
) -> set[Hashable]:
    """Choose groups near the requested row fraction with a stable seed."""

    if len(sizes) < 2:
        return set()
    total = sum(sizes.values())
    target = max(1, int(math.ceil(total * fraction)))
    rng = np.random.RandomState(seed)
    order = list(sizes)
    rng.shuffle(order)
    # A bitset records reachable row totals; back-pointers reconstruct one
    # seeded subset without allocating a Python set for every possible total.
    reachable = 1
    parents = {}
    limit = min(total - 1, target + max(sizes.values()))
    mask = (1 << (limit + 1)) - 1
    for label in order:
        size = sizes[label]
        new = ((reachable << size) & mask) & ~reachable
        pending = new
        while pending:
            bit = pending & -pending
            value = bit.bit_length() - 1
            parents[value] = (value - size, label)
            pending ^= bit
        reachable |= new
    candidates = [value for value in parents if value > 0]
    if not candidates:
        return set()
    value = min(candidates, key=lambda n: (abs(n - target), n))
    chosen = set()
    while value:
        value, label = parents[value]
        chosen.add(label)
    return chosen


def _pair_validation_keys(
    reference: pd.DataFrame,
    peer_frames: Mapping[str, pd.DataFrame],
    *,
    fraction: float,
    seed: int,
) -> set[tuple[str, str, str]]:
    ref_keys = _unique_keys(reference)
    by_peptide: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for key in ref_keys:
        by_peptide[key[0]].append(key)
    peer_keys = {name: _keys_by_peptide(frame) for name, frame in peer_frames.items()}
    eligible = {pep: keys for pep, keys in by_peptide.items() if len(keys) > 1}
    if not eligible:
        return set()
    requested = max(1, int(math.ceil(len(ref_keys) * fraction)))
    capacities = {pep: len(keys) - 1 for pep, keys in eligible.items()}
    requested = min(requested, sum(capacities.values()))
    raw = {pep: requested * len(keys) / len(ref_keys) for pep, keys in eligible.items()}
    quota = {pep: min(capacities[pep], int(math.floor(raw[pep]))) for pep in eligible}
    remaining = requested - sum(quota.values())
    rng = np.random.RandomState(seed)
    tie = {pep: float(rng.random_sample()) for pep in eligible}
    while remaining:
        choices = [pep for pep in eligible if quota[pep] < capacities[pep]]
        if not choices:
            break
        pep = max(
            choices,
            key=lambda value: (raw[value] - math.floor(raw[value]), tie[value], value),
        )
        quota[pep] += 1
        remaining -= 1

    selected: set[tuple[str, str, str]] = set()
    remaining_by_peer = {
        name: {pep: len(values) for pep, values in groups.items()}
        for name, groups in peer_keys.items()
    }
    for pep in sorted(eligible):
        candidates = list(eligible[pep])
        commonity = {
            key: sum(key in keys.get(pep, set()) for keys in peer_keys.values())
            for key in candidates
        }
        tie = {key: float(rng.random_sample()) for key in candidates}
        candidates.sort(key=lambda key: (-commonity[key], tie[key], key))
        selected_count = 0
        for key in candidates:
            if selected_count >= quota[pep]:
                break
            safe = True
            for name, keys_by_pep in peer_keys.items():
                existing = keys_by_pep.get(pep, set())
                if key in existing and remaining_by_peer[name][pep] <= 1:
                    safe = False
                    break
            if safe:
                selected.add(key)
                selected_count += 1
                for name, groups in peer_keys.items():
                    if key in groups.get(pep, set()):
                        remaining_by_peer[name][pep] -= 1
    return selected


def select_validation(
    reference: pd.DataFrame,
    *,
    peers: dict[str, pd.DataFrame],
    grouping: str,
    seed: int = 42,
    fraction: float = 0.1,
    cluster_labels: dict[str, Hashable] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select one fixed positive validation block from the reference corpus."""

    _validate_args(grouping, fraction)
    ref = _normalise(reference, name="reference")
    # Validation is one observation per potential-duplicate positive pair.  Keep the source
    # frame untouched so apply_validation can later preserve every train row.
    ref_unique = ref.loc[
        ~pd.Series(list(_keys_iter(ref))).duplicated().to_numpy()
    ].reset_index(drop=True)
    peer_frames = {
        name: _normalise(frame, name=f"peers[{name!r}]")
        for name, frame in peers.items()
    }
    labels = (
        _check_cluster_labels([ref, *peer_frames.values()], cluster_labels)
        if grouping == "cluster"
        else None
    )

    if grouping == "pair_stratified":
        valid_keys = _pair_validation_keys(
            ref_unique, {"reference": ref, **peer_frames}, fraction=fraction, seed=seed
        )
        key_set = set(valid_keys)
        valid = ref_unique.loc[
            [key in key_set for key in _keys_iter(ref_unique)]
        ].reset_index(drop=True)
        validation_groups = sorted({key[0] for key in valid_keys})
    else:
        if grouping == "peptide":

            def group_for(key):
                return key[0]
        else:
            assert labels is not None

            def group_for(key):
                return labels[key[0]]

        sizes: dict[Hashable, int] = defaultdict(int)
        for key in _keys_iter(ref_unique):
            sizes[group_for(key)] += 1
        chosen_groups = _choose_group_labels(sizes, fraction=fraction, seed=seed)
        valid = ref_unique.loc[
            [group_for(key) in chosen_groups for key in _keys_iter(ref_unique)]
        ].reset_index(drop=True)
        validation_groups = sorted(chosen_groups, key=str)

    ref_keys_by_pep = _keys_by_peptide(ref)
    peer_coverage_before = {
        name: len(_keys_by_peptide(frame))
        for name, frame in {"reference": ref, **peer_frames}.items()
    }
    report: dict[str, Any] = {
        "grouping": grouping,
        "seed": seed,
        "fraction_requested": fraction,
        "reference_rows": int(len(ref)),
        "reference_unique_pairs": int(len(_unique_keys(ref))),
        "validation_rows": int(len(valid)),
        "validation_unique_pairs": int(len(_unique_keys(valid))),
        "validation_peptides": int(valid["pep"].nunique()),
        "validation_groups": validation_groups,
        "peer_names": sorted(peer_frames),
        "peer_peptide_coverage_before": peer_coverage_before,
        "singleton_peptides_train_only": (
            sorted(pep for pep, keys in ref_keys_by_pep.items() if len(keys) == 1)
            if grouping == "pair_stratified"
            else []
        ),
        "cdr3_normalization": "raw_strip_upper; grouping_only_endpoint_C_FW_closure",
    }
    return valid, report


def apply_validation(
    frame: pd.DataFrame,
    valid_positive: pd.DataFrame,
    *,
    grouping: str,
    cluster_labels: dict[str, Hashable] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove the fixed validation block from a corpus while preserving rows."""

    if grouping not in GROUPINGS:
        raise ValueError(
            f"unsupported grouping: {grouping!r}; expected one of {GROUPINGS}"
        )
    source = _normalise(frame)
    valid = _normalise(valid_positive, name="valid_positive")
    validation_keys = set(_keys_iter(valid))
    before_peptides = set(source["pep"])

    if grouping == "pair_stratified":
        excluded = [key in validation_keys for key in _keys_iter(source)]
        validation_groups = sorted({key[0] for key in validation_keys})
    elif grouping == "peptide":
        validation_peptides = {key[0] for key in validation_keys}
        excluded = [key[0] in validation_peptides for key in _keys_iter(source)]
        validation_groups = sorted(validation_peptides)
    else:
        labels = _check_cluster_labels([source, valid], cluster_labels)
        validation_cluster_labels = {labels[key[0]] for key in validation_keys}
        excluded = [
            labels[key[0]] in validation_cluster_labels for key in _keys_iter(source)
        ]
        validation_groups = sorted(validation_cluster_labels, key=str)

    train = source.loc[~np.asarray(excluded, dtype=bool)].reset_index(drop=True)
    train_keys = set(_keys_iter(train))
    report = {
        "grouping": grouping,
        "input_rows": int(len(source)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "unique_train_keys": int(len(train_keys)),
        "validation_unique_pairs": int(len(validation_keys)),
        "val_pairs_overlap": int(len(train_keys & validation_keys)),
        "validation_pairs_overlap": int(len(train_keys & validation_keys)),
        "peptide_coverage_before": int(len(before_peptides)),
        "peptide_coverage_after": int(train["pep"].nunique()),
        "peptide_coverage_lost": int(len(before_peptides - set(train["pep"]))),
        "validation_groups": validation_groups,
    }
    return train, report


def _label_token(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    text = str(value).strip().upper()
    try:
        numeric = float(text)
    except ValueError:
        return text
    return int(numeric) if numeric.is_integer() else numeric


def deduplicate_evaluation(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deduplicate equal-label potential-duplicate pairs and reject label conflicts."""

    if "label" not in frame.columns:
        raise ValueError("evaluation frame must contain a label column")
    source = _normalise(frame)
    grouped: dict[tuple[str, str, str], list[tuple[int, Any]]] = defaultdict(list)
    for index, key in enumerate(_keys_iter(source)):
        grouped[key].append((index, _label_token(source.iloc[index]["label"])))
    conflicts = {
        key: sorted({label for _, label in entries}, key=str)
        for key, entries in grouped.items()
        if len({label for _, label in entries}) > 1
    }
    if conflicts:
        examples = list(conflicts.items())[:3]
        raise ValueError(
            f"conflicting labels for potential-duplicate evaluation pairs: {examples}"
        )
    keep = [entries[0][0] for entries in grouped.values()]
    result = source.iloc[keep].reset_index(drop=True)
    report = {
        "input_rows": int(len(source)),
        "output_rows": int(len(result)),
        "duplicate_rows_removed": int(len(source) - len(result)),
        "unique_pairs": int(len(grouped)),
        "conflicting_pairs": 0,
        "pair_group_columns": list(KEY_COLUMNS),
        "cdr3_normalization": "raw_strip_upper; grouping_only_endpoint_C_FW_closure",
    }
    return result, report
