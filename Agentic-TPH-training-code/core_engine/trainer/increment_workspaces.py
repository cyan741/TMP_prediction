from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

KEY_COLUMNS = ("pep", "hla.allele", "ab")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
CLASS_PEPTIDE_LIMITS = {"I": (8, 13), "II": (0, 30)}


@dataclass(frozen=True)
class CapabilityCaps:
    """Sequence length limits a workspace accepts.

    ``peptide_max`` is 13 for class I rather than the canonical 8-11 because
    this corpus carries real bulged class-I epitopes at 12-13 aa, including
    DATYQRTRALVR which IMMREP23 also evaluates.  Everything above 13 is either
    an unparsed modification annotation or a screening long peptide, not a
    presented epitope.

    ``peptide_min`` exists because the class I groove cannot present fewer than
    8 residues, so a shorter value is a truncation defect rather than a short
    epitope.  Only an upper bound was enforced before, which let three 7-mer
    rows into training unremarked.
    """

    peptide_max: int = 13
    hla_length: int = 34
    chain_max: int = 28  # level III carries CDR3s; level IV carries variable domains
    peptide_min: int = 8

    @classmethod
    def for_level(
        cls, level: str, *, mhc_class: str = "I", peptide_max: int | None = None
    ) -> "CapabilityCaps":
        """Caps for one level and MHC class.

        ``peptide_max`` still overrides the class default, so a caller that
        deliberately widens or narrows the peptide window keeps saying so
        explicitly.
        """

        try:
            low, high = CLASS_PEPTIDE_LIMITS[str(mhc_class)]
        except KeyError as exc:
            raise ValueError(f"unsupported MHC class: {mhc_class!r}") from exc
        # Levels I-III use CDR3 sequences; Level IV uses reconstructed variable domains.
        # A width of 28 covers the longest CDR3 in the TPH_260630 training sources
        # and the supported IMMREP test inputs without removing terminal residues.
        if level in {"I", "II", "III"}:
            chain_max = 28
        elif level == "IV":
            chain_max = 127
        else:
            raise ValueError(f"unsupported level: {level}")
        return cls(
            peptide_max=high if peptide_max is None else peptide_max,
            chain_max=chain_max,
            peptide_min=low,
        )


def _norm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _training_peptide(raw: pd.DataFrame, column: str = "antigen.epitope") -> pd.Series:
    """The peptide to train on: the minimal epitope when one was resolved.

    A screening long peptide is what the assay used, not what the MHC
    presented.  Where the collection side resolved the presented epitope from
    the source publication, that is the sequence the model should see, because
    it is also the length it will see at inference.  The long peptide stays in
    the snapshot's own column, so this substitution is a training decision that
    can be inspected, not a rewrite of the record.
    """

    peptide = _norm(raw[column])
    if "minimal_epitope" not in raw.columns:
        return peptide
    minimal = _norm(raw["minimal_epitope"])
    return minimal.where(minimal != "", peptide)


def _keys_iter(frame: pd.DataFrame):
    """Formal pair identity using full exact-CDR3 after strip + uppercase."""

    return zip(
        _norm(frame["pep"]),
        _norm(frame["hla.allele"]),
        _norm(frame["ab"]),
    )


def apply_capability_filter(
    frame: pd.DataFrame, caps: CapabilityCaps
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Drop rows the backend cannot represent and report why, per reason."""

    chains = frame["ab"].str.split("/", n=1)
    alpha, beta = chains.str[0], chains.str[-1]
    reasons = {
        "peptide_too_long": frame["pep"].str.len() > caps.peptide_max,
        "peptide_too_short": frame["pep"].str.len() < caps.peptide_min,
        "peptide_not_amino_acid": ~frame["pep"].map(
            lambda s: bool(s) and set(s) <= STANDARD_AA
        ),
        "hla_length_mismatch": frame["hla"].str.len() != caps.hla_length,
        "alpha_too_long": alpha.str.len() > caps.chain_max,
        "beta_too_long": beta.str.len() > caps.chain_max,
        "chain_missing": (alpha.str.len() == 0)
        | (beta.str.len() == 0)
        | (frame["ab"].str.count("/") != 1),
    }
    excluded = np.zeros(len(frame), dtype=bool)
    counts: dict[str, int] = {}
    for name, mask in reasons.items():
        counts[name] = int(mask.sum())
        excluded |= mask.to_numpy()
    excluded_rows = frame.loc[excluded].copy()
    for name, mask in reasons.items():
        excluded_rows[name] = mask.loc[excluded].to_numpy()
    return frame.loc[~excluded].reset_index(drop=True), excluded_rows, counts


def _edit_distance(a: str, b: str, cutoff: int) -> int:
    """Levenshtein distance, abandoned early once it cannot fall below cutoff."""

    if abs(len(a) - len(b)) > cutoff:
        return cutoff + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        if min(current) > cutoff:
            return cutoff + 1
        previous = current
    return previous[-1]


def cluster_peptides(peptides: Sequence[str], threshold: int) -> dict[str, int]:
    """Single-linkage clustering so a mutant family never straddles a split.

    Length bucketing keeps the pairwise sweep tractable: two peptides differing
    by more than ``threshold`` in length cannot be within ``threshold`` edits.
    """

    ordered = sorted(set(peptides))
    parent = {p: p for p in ordered}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_length: dict[int, list[str]] = {}
    for p in ordered:
        by_length.setdefault(len(p), []).append(p)
    for length, group in by_length.items():
        candidates = [
            q
            for delta in range(0, threshold + 1)
            for q in by_length.get(length + delta, [])
        ]
        for i, a in enumerate(group):
            for b in candidates:
                if len(b) == len(a) and b <= a:
                    continue
                if _edit_distance(a, b, threshold) <= threshold:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
    roots = {}
    labels = {}
    for p in ordered:
        root = find(p)
        labels[p] = roots.setdefault(root, len(roots))
    return labels
