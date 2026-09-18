"""Human+mouse pan-MHC Level-III pilot dataset, split and sampler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .omni_sequence_registry import (
    HLA_II_BETA_LOCI,
    OmniSequenceResolver,
    SequenceResolution,
    hla_locus,
    hla_two_field,
    is_standard_sequence,
    normalize_hla_allele,
    normalize_sequence,
)

PILOT_SLICES = ("human_I", "human_II", "mouse_I", "mouse_II")
SOURCE_FILES = {
    "human_I": "TPH_human_hla_i/TPH-level-III.csv",
    "human_II": "TPH_human_hla_ii/TPH-level-III.csv",
    "mouse_I": "TPH_mouse_mhc_i/TPH-level-III.csv",
    "mouse_II": "TPH_mouse_mhc_ii/TPH-level-III.csv",
}
PEPTIDE_MAX_LENGTH = 30
TCR_MAX_LENGTH = 40
MHC_ALPHA_MAX_LENGTH = 400
MHC_BETA_MAX_LENGTH = 300


def exact_cdr3(value: object) -> str:
    """Project exact-CDR3 using only strip and uppercase."""

    return str(value or "").strip().upper()


def _key(*values: object) -> str:
    return "|".join(str(value) for value in values)


def _source_value(row: pd.Series, name: str) -> str:
    return str(row.get(name, "") or "").strip()


@dataclass(slots=True)
class PreparedOmniData:
    positives: pd.DataFrame
    exclusions: pd.DataFrame
    report: dict[str, Any]
    sequence_registry: pd.DataFrame


def prepare_positive_data(
    *, data_root: Path, resolver: OmniSequenceResolver
) -> PreparedOmniData:
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    used_resolutions: list[tuple[str, str, SequenceResolution]] = []
    input_counts: dict[str, int] = {}
    for slice_name in PILOT_SLICES:
        path = Path(data_root) / SOURCE_FILES[slice_name]
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        input_counts[slice_name] = len(frame)
        for offset, row in frame.iterrows():
            source_row = offset + 2
            prepared, reason, resolutions = _prepare_row(
                slice_name=slice_name,
                row=row,
                source_path=path,
                source_row=source_row,
                resolver=resolver,
            )
            if reason:
                exclusions.append(
                    {
                        "slice": slice_name,
                        "source_path": str(path),
                        "source_row_number": source_row,
                        "reason": reason,
                        "peptide": _source_value(row, "antigen.epitope"),
                        "allele_raw": _source_value(
                            row,
                            "hla.allele"
                            if slice_name.startswith("human")
                            else "mhc.allele",
                        ),
                    }
                )
                continue
            assert prepared is not None
            records.append(prepared)
            used_resolutions.extend(resolutions)
    positives = pd.DataFrame(records)
    exclusion_frame = pd.DataFrame(exclusions)
    registry = pd.DataFrame(resolver.used_registry_rows(used_resolutions))
    resolved_counts = positives["slice"].value_counts().to_dict()
    reason_counts = (
        exclusion_frame.groupby(["slice", "reason"])
        .size()
        .rename("rows")
        .reset_index()
        .to_dict("records")
        if not exclusion_frame.empty
        else []
    )
    report = {
        "contract": {
            "scope": "human+mouse/MHC-I+II/paired-alpha-beta/Level-III strict-positive source view",
            "exact_cdr3": "strip+upper only; complete source sequence; no C/W/F endpoint editing",
            "tcr_key": "species|exact_alpha_cdr3|exact_beta_cdr3",
            "allele_key": "species|mhc_class|resolved alpha allele|resolved beta allele",
            "pmhc_key": "species|mhc_class|peptide|allele_key",
            "pair_key": "tcr_key|pmhc_key",
            "mhc_i": "explicit alpha/full sequence or source pseudosequence; beta chain empty",
            "mhc_ii": "explicit alpha and beta protein sequences",
            "human_dr_rule": "DRB-only rows use HLA-DRA*01:01 only; DQ/DP never infer a missing partner",
            "peptide_max_length": PEPTIDE_MAX_LENGTH,
            "measured_unlabeled": "excluded by source contract; no thresholding or binary conversion",
            "strict_negative": "not consumed; label 0 is derived random mismatch only",
        },
        "input_rows": input_counts,
        "resolved_rows": {
            name: int(resolved_counts.get(name, 0)) for name in PILOT_SLICES
        },
        "excluded_rows": {
            name: int(input_counts[name] - resolved_counts.get(name, 0))
            for name in PILOT_SLICES
        },
        "resolution_coverage": {
            name: float(resolved_counts.get(name, 0) / input_counts[name])
            for name in PILOT_SLICES
        },
        "exclusion_reason_counts": reason_counts,
        "peptide_length_distribution": {
            name: {
                str(int(length)): int(count)
                for length, count in positives.loc[
                    positives["slice"] == name, "peptide"
                ]
                .str.len()
                .value_counts()
                .sort_index()
                .items()
            }
            for name in PILOT_SLICES
        },
        "backend_limits": {
            "peptide": PEPTIDE_MAX_LENGTH,
            "tcr_chain": TCR_MAX_LENGTH,
            "mhc_alpha": MHC_ALPHA_MAX_LENGTH,
            "mhc_beta": MHC_BETA_MAX_LENGTH,
        },
        "closure_holds": len(positives) + len(exclusion_frame)
        == sum(input_counts.values()),
        "provenance_columns": ["source_path", "source_row_number", "dataset_tag"],
    }
    if not report["closure_holds"]:
        raise AssertionError("omni input closure failed")
    return PreparedOmniData(positives, exclusion_frame, report, registry)


def _prepare_row(
    *,
    slice_name: str,
    row: pd.Series,
    source_path: Path,
    source_row: int,
    resolver: OmniSequenceResolver,
) -> tuple[
    dict[str, Any] | None, str | None, list[tuple[str, str, SequenceResolution]]
]:
    species, mhc_class = slice_name.split("_")
    if _source_value(row, "mhc.species").lower() != species:
        return None, "source_species_conflict", []
    if _source_value(row, "mhc.class").upper() != mhc_class:
        return None, "source_mhc_class_conflict", []
    peptide = normalize_sequence(row.get("antigen.epitope"))
    alpha_cdr3 = exact_cdr3(row.get("alpha.cdr3"))
    beta_cdr3 = exact_cdr3(row.get("beta.cdr3"))
    if not is_standard_sequence(peptide):
        return None, "peptide_nonstandard_or_annotated", []
    if len(peptide) > PEPTIDE_MAX_LENGTH:
        return None, "peptide_backend_limit", []
    if not is_standard_sequence(alpha_cdr3) or not is_standard_sequence(beta_cdr3):
        return None, "cdr3_nonstandard_or_missing", []
    if len(alpha_cdr3) > TCR_MAX_LENGTH or len(beta_cdr3) > TCR_MAX_LENGTH:
        return None, "cdr3_backend_limit", []

    allele_column = "hla.allele" if species == "human" else "mhc.allele"
    allele_raw = _source_value(row, allele_column)
    mhc_alpha_allele = ""
    mhc_beta_allele = ""
    mhc_alpha_seq = ""
    mhc_beta_seq = ""
    resolutions: list[tuple[str, str, SequenceResolution]] = []
    if slice_name == "human_I":
        mhc_alpha_allele = allele_raw
        mhc_alpha_seq = normalize_sequence(
            row.get("hla.full.seq") or row.get("hla.short.seq")
        )
        if not is_standard_sequence(mhc_alpha_seq):
            return None, "mhc_i_sequence_missing_or_nonstandard", []
    elif slice_name == "mouse_I":
        if allele_raw.upper().startswith(("I-A", "I-E", "H-2-IA", "H-2-IE")):
            return None, "source_class_allele_conflict", []
        resolution = resolver.resolve_mouse_i(allele_raw)
        if resolution is None:
            return None, "mouse_mhc_i_sequence_unresolved", []
        mhc_alpha_allele = allele_raw
        mhc_alpha_seq = resolution.sequence
        resolutions.append((species, "alpha", resolution))
    elif slice_name == "human_II":
        alpha_allele = normalize_hla_allele(row.get("hla.alpha.allele"))
        beta_allele = normalize_hla_allele(row.get("hla.beta.allele"))
        if not beta_allele or hla_locus(beta_allele) not in HLA_II_BETA_LOCI:
            return None, "mhc_ii_beta_allele_missing_or_invalid", []
        if not alpha_allele:
            if hla_locus(beta_allele).startswith("DRB"):
                alpha_allele = "HLA-DRA*01:01"
            else:
                return None, "mhc_ii_alpha_allele_missing", []
        elif hla_locus(beta_allele).startswith("DRB") and alpha_allele == "HLA-DRA*01":
            alpha_allele = "HLA-DRA*01:01"
        alpha_resolution = resolver.resolve_hla_chain(alpha_allele)
        beta_resolution = resolver.resolve_hla_chain(beta_allele)
        if alpha_resolution is None:
            return None, "mhc_ii_alpha_sequence_unresolved", []
        if beta_resolution is None:
            return None, "mhc_ii_beta_sequence_unresolved", []
        mhc_alpha_allele = hla_two_field(alpha_allele)
        mhc_beta_allele = hla_two_field(beta_allele)
        mhc_alpha_seq = alpha_resolution.sequence
        mhc_beta_seq = beta_resolution.sequence
        resolutions.extend(
            [(species, "alpha", alpha_resolution), (species, "beta", beta_resolution)]
        )
    else:
        pair = resolver.resolve_mouse_ii(allele_raw)
        if pair is None:
            return None, "mouse_mhc_ii_chain_pair_unresolved", []
        alpha_resolution, beta_resolution = pair
        mhc_alpha_allele = f"{allele_raw}:alpha"
        mhc_beta_allele = f"{allele_raw}:beta"
        mhc_alpha_seq = alpha_resolution.sequence
        mhc_beta_seq = beta_resolution.sequence
        resolutions.extend(
            [(species, "alpha", alpha_resolution), (species, "beta", beta_resolution)]
        )
    if len(mhc_alpha_seq) > MHC_ALPHA_MAX_LENGTH:
        return None, "mhc_alpha_backend_limit", []
    if mhc_beta_seq and len(mhc_beta_seq) > MHC_BETA_MAX_LENGTH:
        return None, "mhc_beta_backend_limit", []
    if mhc_class == "II" and not mhc_beta_seq:
        return None, "mhc_ii_beta_sequence_missing", []

    tcr_key = _key(species, alpha_cdr3, beta_cdr3)
    allele_key = _key(species, mhc_class, mhc_alpha_allele, mhc_beta_allele)
    pmhc_key = _key(species, mhc_class, peptide, allele_key)
    pair_key = _key(tcr_key, pmhc_key)
    return (
        {
            "record_id": f"{slice_name}:{source_row}",
            "slice": slice_name,
            "species": species,
            "mhc_class": mhc_class,
            "source_path": str(source_path),
            "source_row_number": source_row,
            "dataset_tag": _source_value(row, "dataset.tag"),
            "tcr_species": _source_value(row, "tcr.species").lower(),
            "peptide": peptide,
            "allele_raw": allele_raw,
            "mhc_alpha_allele": mhc_alpha_allele,
            "mhc_beta_allele": mhc_beta_allele,
            "mhc_alpha_seq": mhc_alpha_seq,
            "mhc_beta_seq": mhc_beta_seq,
            "tcr_alpha_cdr3": alpha_cdr3,
            "tcr_beta_cdr3": beta_cdr3,
            "tcr_key": tcr_key,
            "allele_key": allele_key,
            "pmhc_key": pmhc_key,
            "pair_key": pair_key,
            "label": 1,
            "negative_type": "",
        },
        None,
        resolutions,
    )
