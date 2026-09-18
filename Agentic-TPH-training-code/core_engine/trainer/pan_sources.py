"""Four-domain pan-MHC dataset preparation with frozen unseen epitopes."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .omni_data import PreparedOmniData, _prepare_row, exact_cdr3, prepare_positive_data
from .omni_sequence_registry import (
    OmniSequenceResolver,
    SequenceResolution,
    is_standard_sequence,
    normalize_sequence,
    parse_fasta,
)

DOMAINS = ("human_I", "human_II", "nonhuman_I", "nonhuman_II")
BASE_TO_DOMAIN = {
    "human_I": "human_I",
    "human_II": "human_II",
    "mouse_I": "nonhuman_I",
    "mouse_II": "nonhuman_II",
}
OTHER_FILES = {
    "other_I": "TPH_other_mhc_i/TPH-level-III.csv",
    "other_II": "TPH_other_mhc_ii/TPH-level-III.csv",
}
HUMAN_I_UNION_FILE = "TPH_HiTPHplus_human_hla_i/TPH-HiTPHplus-level-III.csv"
PEPTIDE_MAX_LENGTH = 30
TCR_MAX_LENGTH = 40
MHC_ALPHA_MAX_LENGTH = 400
MHC_BETA_MAX_LENGTH = 300
MHC_PACKED_LENGTH = MHC_ALPHA_MAX_LENGTH + MHC_BETA_MAX_LENGTH


def _key(*parts: object) -> str:
    return "|".join(str(part) for part in parts)


def _other_two_field(value: object) -> str:
    text = str(value or "").strip().upper()
    match = re.match(r"^([^*]+\*\d+)(?::(\d+))?", text)
    if not match or not match.group(2):
        return text
    return f"{match.group(1)}:{match.group(2)}"


class PanMHCSequenceResolver(OmniSequenceResolver):
    """Extend the frozen human/mouse resolver with official IPD-MHC proteins."""

    def __init__(
        self, *, hla_fasta: Path, mouse_uniprot_fasta: Path, ipd_mhc_fasta: Path
    ) -> None:
        super().__init__(hla_fasta=hla_fasta, mouse_uniprot_fasta=mouse_uniprot_fasta)
        self.ipd_mhc_source = str(Path(ipd_mhc_fasta).resolve())
        self._other = self._load_other(ipd_mhc_fasta)

    @staticmethod
    def _load_other(path: Path) -> dict[str, SequenceResolution]:
        grouped: dict[str, list[SequenceResolution]] = {}
        for header, record_id, sequence in parse_fasta(path):
            fields = header.split()
            if len(fields) < 2 or not is_standard_sequence(sequence):
                continue
            allele = fields[1]
            key = _other_two_field(allele)
            grouped.setdefault(key, []).append(
                SequenceResolution(
                    sequence=sequence,
                    allele=allele,
                    source=str(Path(path).resolve()),
                    source_record=record_id,
                    method="ipd_mhc_modal_protein_two_field",
                )
            )
        result: dict[str, SequenceResolution] = {}
        for key, values in grouped.items():
            counts = Counter(item.sequence for item in values)
            modal = counts.most_common(1)[0][0]
            first = next(item for item in values if item.sequence == modal)
            result[key] = SequenceResolution(
                sequence=first.sequence,
                allele=key,
                source=first.source,
                source_record=first.source_record,
                method=first.method,
            )
        return result

    def resolve_other_i(self, allele: object) -> SequenceResolution | None:
        return self._other.get(_other_two_field(allele))


@dataclass(slots=True)
class PreparedPanMHCData:
    positives: pd.DataFrame
    exclusions: pd.DataFrame
    sequence_registry: pd.DataFrame
    report: dict[str, Any]


def _prepare_other_i_row(
    row: pd.Series,
    *,
    source_path: Path,
    source_row: int,
    resolver: PanMHCSequenceResolver,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, SequenceResolution | None]:
    peptide = normalize_sequence(row.get("antigen.epitope"))
    alpha_cdr3 = exact_cdr3(row.get("alpha.cdr3"))
    beta_cdr3 = exact_cdr3(row.get("beta.cdr3"))
    allele = str(row.get("mhc.allele", "") or "").strip()
    reason = ""
    resolution: SequenceResolution | None = None
    if str(row.get("mhc.species", "")).strip().lower() != "other":
        reason = "source_species_conflict"
    elif str(row.get("mhc.class", "")).strip().upper() != "I":
        reason = "source_mhc_class_conflict"
    elif not is_standard_sequence(peptide) or len(peptide) > PEPTIDE_MAX_LENGTH:
        reason = "peptide_nonstandard_or_backend_limit"
    elif not is_standard_sequence(alpha_cdr3) or not is_standard_sequence(beta_cdr3):
        reason = "cdr3_nonstandard_or_missing"
    elif len(alpha_cdr3) > TCR_MAX_LENGTH or len(beta_cdr3) > TCR_MAX_LENGTH:
        reason = "cdr3_backend_limit"
    else:
        resolution = resolver.resolve_other_i(allele)
        if resolution is None:
            reason = "other_mhc_i_sequence_unresolved"
        elif len(resolution.sequence) > MHC_ALPHA_MAX_LENGTH:
            reason = "mhc_alpha_backend_limit"
    if reason:
        return (
            None,
            {
                "slice": "other_I",
                "source_path": str(source_path),
                "source_row_number": source_row,
                "reason": reason,
                "peptide": peptide,
                "allele_raw": allele,
            },
            None,
        )
    assert resolution is not None
    species = "other"
    mhc_class = "I"
    tcr_key = _key(
        str(row.get("tcr.species", "") or species).strip().lower(),
        alpha_cdr3,
        beta_cdr3,
    )
    allele_key = _key(species, mhc_class, _other_two_field(allele), "")
    pmhc_key = _key(species, mhc_class, peptide, allele_key)
    return (
        {
            "record_id": f"other_I:{source_row}",
            "slice": "other_I",
            "domain": "nonhuman_I",
            "species": species,
            "mhc_class": mhc_class,
            "source_path": str(source_path),
            "source_row_number": source_row,
            "dataset_tag": str(row.get("dataset.tag", "") or "").strip(),
            "tcr_species": str(row.get("tcr.species", "") or "").strip().lower(),
            "peptide": peptide,
            "allele_raw": allele,
            "mhc_alpha_allele": _other_two_field(allele),
            "mhc_beta_allele": "",
            "mhc_alpha_seq": resolution.sequence,
            "mhc_beta_seq": "",
            "tcr_alpha_cdr3": alpha_cdr3,
            "tcr_beta_cdr3": beta_cdr3,
            "tcr_key": tcr_key,
            "allele_key": allele_key,
            "pmhc_key": pmhc_key,
            "pair_key": _key(tcr_key, pmhc_key),
            "label": 1,
            "negative_type": "",
        },
        None,
        resolution,
    )


def prepare_pan_mhc_data(
    *, data_root: Path, resolver: PanMHCSequenceResolver
) -> PreparedPanMHCData:
    base: PreparedOmniData = prepare_positive_data(
        data_root=data_root, resolver=resolver
    )
    positives = base.positives.loc[base.positives["slice"] != "human_I"].copy()
    exclusions = base.exclusions.loc[base.exclusions["slice"] != "human_I"].copy()
    human_i_path = Path(data_root) / HUMAN_I_UNION_FILE
    human_i = pd.read_csv(human_i_path, dtype=str, keep_default_na=False)
    human_i_records: list[dict[str, Any]] = []
    human_i_exclusions: list[dict[str, Any]] = []
    for offset, row in human_i.iterrows():
        record, reason, _resolutions = _prepare_row(
            slice_name="human_I",
            row=row,
            source_path=human_i_path,
            source_row=offset + 2,
            resolver=resolver,
        )
        if record is not None:
            human_i_records.append(record)
        else:
            human_i_exclusions.append(
                {
                    "slice": "human_I",
                    "source_path": str(human_i_path),
                    "source_row_number": offset + 2,
                    "reason": reason,
                    "peptide": str(row.get("antigen.epitope", "") or "").strip(),
                    "allele_raw": str(row.get("hla.allele", "") or "").strip(),
                }
            )
    positives = pd.concat([positives, pd.DataFrame(human_i_records)], ignore_index=True)
    if human_i_exclusions:
        exclusions = pd.concat(
            [exclusions, pd.DataFrame(human_i_exclusions)], ignore_index=True
        )
    positives["domain"] = positives["slice"].map(BASE_TO_DOMAIN)
    registry_rows = base.sequence_registry.to_dict("records")
    input_rows = dict(base.report["input_rows"])
    resolved_rows = dict(base.report["resolved_rows"])
    excluded_rows = dict(base.report["excluded_rows"])
    input_rows["human_I"] = len(human_i)
    resolved_rows["human_I"] = len(human_i_records)
    excluded_rows["human_I"] = len(human_i_exclusions)

    other_i_path = Path(data_root) / OTHER_FILES["other_I"]
    other_i = pd.read_csv(other_i_path, dtype=str, keep_default_na=False)
    input_rows["other_I"] = len(other_i)
    other_records: list[dict[str, Any]] = []
    other_exclusions: list[dict[str, Any]] = []
    for offset, row in other_i.iterrows():
        record, exclusion, resolution = _prepare_other_i_row(
            row, source_path=other_i_path, source_row=offset + 2, resolver=resolver
        )
        if record is not None:
            other_records.append(record)
            assert resolution is not None
            registry_rows.append(
                {
                    "species": "other",
                    "chain": "alpha",
                    "resolved_allele": resolution.allele,
                    "sequence": resolution.sequence,
                    "source": resolution.source,
                    "source_record": resolution.source_record,
                    "resolution_method": resolution.method,
                }
            )
        else:
            assert exclusion is not None
            other_exclusions.append(exclusion)
    resolved_rows["other_I"] = len(other_records)
    excluded_rows["other_I"] = len(other_exclusions)

    other_ii_path = Path(data_root) / OTHER_FILES["other_II"]
    other_ii = pd.read_csv(other_ii_path, dtype=str, keep_default_na=False)
    input_rows["other_II"] = len(other_ii)
    resolved_rows["other_II"] = 0
    excluded_rows["other_II"] = 0
    if other_records:
        positives = pd.concat(
            [positives, pd.DataFrame(other_records)], ignore_index=True
        )
    if other_exclusions:
        exclusions = pd.concat(
            [exclusions, pd.DataFrame(other_exclusions)], ignore_index=True
        )
    pre_dedup_rows = len(positives)
    positives = positives.drop_duplicates(
        subset=["pair_key"], keep="first"
    ).reset_index(drop=True)
    duplicate_exact_pairs_removed = pre_dedup_rows - len(positives)
    sequence_registry = pd.DataFrame(registry_rows).drop_duplicates(
        subset=["species", "chain", "resolved_allele"], keep="first"
    )
    domain_counts = positives["domain"].value_counts().to_dict()
    report = {
        "contract": {
            "domains": list(DOMAINS),
            "nonhuman_I": "mouse MHC-I plus resolvable other-species MHC-I",
            "nonhuman_II": "mouse MHC-II plus resolvable other-species MHC-II",
            "exact_cdr3": "strip+upper only; no endpoint editing",
            "mhc_input": "fixed 700 tokens: alpha[0:400], beta[400:700]; class-I beta is padding",
            "strict_negative": "not consumed; label 0 is derived random mismatch only",
            "human_I_source": "TPH_260630 TPH+Hi-TpH union",
        },
        "input_rows": input_rows,
        "resolved_rows": resolved_rows,
        "excluded_rows": excluded_rows,
        "resolved_domain_rows": {
            domain: int(domain_counts.get(domain, 0)) for domain in DOMAINS
        },
        "duplicate_exact_pair_rows_removed": duplicate_exact_pairs_removed,
        "other_species_boundary": {
            "other_I_resolved_rows": len(other_records),
            "other_II_input_rows": len(other_ii),
            "claim": "other-species MHC-II is absent when input_rows is zero",
        },
        "pre_dedup_closure_holds": pre_dedup_rows + len(exclusions)
        == sum(input_rows.values()),
    }
    if not report["pre_dedup_closure_holds"]:
        raise AssertionError("pan-MHC input closure failed")
    return PreparedPanMHCData(positives, exclusions, sequence_registry, report)
