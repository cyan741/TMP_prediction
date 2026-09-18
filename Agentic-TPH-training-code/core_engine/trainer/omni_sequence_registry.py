"""Sequence resolution for the human+mouse omni pilot."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
HLA_II_ALPHA_LOCI = frozenset({"DRA", "DQA1", "DPA1"})
HLA_II_BETA_LOCI = frozenset({"DRB1", "DRB3", "DRB4", "DRB5", "DQB1", "DPB1"})

# Stable UniProt accessions only; amino-acid sequences remain external inputs.
MOUSE_MHC_I_ACCESSIONS = {
    "H-2-KB": "P01901",
    "H-2-DB": "P01899",
    "H-2-KD": "P01902",
    "H-2-LD": "P01897",
    "H-2-DD": "P01900",
}
MOUSE_MHC_II_ACCESSIONS = {
    "H-2-IAB": ("P14434", "P14483"),
    "H-2-IAB/H-2-IAB": ("P14434", "P14483"),
    "H-2-IAD": ("P04228", "P01921"),
    "H-2-IAD/H-2-IAD": ("P04228", "P01921"),
    "H-2-IAK": ("P01910", "P06343"),
    "H-2-IAU": ("P14438", "P06344"),
    "H-2-IAS": ("P14437", "P06345"),
}


@dataclass(frozen=True, slots=True)
class SequenceResolution:
    sequence: str
    allele: str
    source: str
    source_record: str
    method: str


def normalize_sequence(value: object) -> str:
    """Uppercase and trim a biological sequence without editing endpoints."""

    return str(value or "").strip().upper()


def is_standard_sequence(value: str) -> bool:
    return bool(value) and set(value) <= STANDARD_AA


def parse_fasta(path: Path) -> list[tuple[str, str, str]]:
    records: list[tuple[str, str, str]] = []
    header = ""
    chunks: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header:
                records.append((header, _record_id(header), "".join(chunks).upper()))
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line.strip())
    if header:
        records.append((header, _record_id(header), "".join(chunks).upper()))
    return records


def _record_id(header: str) -> str:
    first = header.split()[0]
    if "|" in first:
        parts = first.split("|")
        if len(parts) >= 2:
            return parts[1]
    return first


def normalize_hla_allele(value: object) -> str:
    text = str(value or "").strip().upper().replace("HLA/HLA-", "HLA-")
    if text.startswith("HLA/"):
        text = text[4:]
    if text and not text.startswith("HLA-"):
        text = f"HLA-{text}"
    return text


def hla_locus(allele: str) -> str:
    match = re.match(r"^HLA-([A-Z0-9]+)\*", allele)
    return match.group(1) if match else ""


def hla_two_field(allele: str) -> str:
    normalized = normalize_hla_allele(allele)
    match = re.match(r"^(HLA-[A-Z0-9]+\*\d+)(?::(\d+))?", normalized)
    if not match or not match.group(2):
        return normalized
    return f"{match.group(1)}:{match.group(2)}"


class OmniSequenceResolver:
    """Resolve HLA-II chains and selected mouse H-2 molecules from frozen FASTA."""

    def __init__(self, *, hla_fasta: Path, mouse_uniprot_fasta: Path) -> None:
        self.hla_source = str(Path(hla_fasta).resolve())
        self.mouse_source = str(Path(mouse_uniprot_fasta).resolve())
        self._hla_exact, self._hla_two_field = self._load_hla(hla_fasta)
        self._mouse = self._load_mouse(mouse_uniprot_fasta)

    @staticmethod
    def _load_hla(
        path: Path,
    ) -> tuple[dict[str, SequenceResolution], dict[str, SequenceResolution]]:
        exact: dict[str, SequenceResolution] = {}
        grouped: dict[str, list[SequenceResolution]] = {}
        for header, record_id, sequence in parse_fasta(path):
            fields = header.split()
            if len(fields) < 2:
                continue
            allele = normalize_hla_allele(fields[1])
            if hla_locus(allele) not in HLA_II_ALPHA_LOCI | HLA_II_BETA_LOCI:
                continue
            if not is_standard_sequence(sequence):
                continue
            resolution = SequenceResolution(
                sequence=sequence,
                allele=allele,
                source=str(Path(path).resolve()),
                source_record=record_id,
                method="imgt_hla_exact",
            )
            exact[allele] = resolution
            grouped.setdefault(hla_two_field(allele), []).append(resolution)
        two_field: dict[str, SequenceResolution] = {}
        for key, values in grouped.items():
            counts = Counter(value.sequence for value in values)
            ranked = counts.most_common()
            if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
                modal_sequence = ranked[0][0]
                first = next(
                    value for value in values if value.sequence == modal_sequence
                )
                two_field[key] = SequenceResolution(
                    sequence=first.sequence,
                    allele=key,
                    source=first.source,
                    source_record=first.source_record,
                    method="imgt_hla_modal_protein_two_field",
                )
        return exact, two_field

    @staticmethod
    def _load_mouse(path: Path) -> dict[str, SequenceResolution]:
        result: dict[str, SequenceResolution] = {}
        for _header, accession, sequence in parse_fasta(path):
            if accession and is_standard_sequence(sequence):
                result[accession] = SequenceResolution(
                    sequence=sequence,
                    allele=accession,
                    source=str(Path(path).resolve()),
                    source_record=accession,
                    method="uniprot_accession",
                )
        return result

    def resolve_hla_chain(self, allele: object) -> SequenceResolution | None:
        normalized = normalize_hla_allele(allele)
        if not normalized:
            return None
        exact = self._hla_exact.get(normalized)
        if exact is not None:
            return exact
        return self._hla_two_field.get(hla_two_field(normalized))

    def resolve_mouse_i(self, allele: object) -> SequenceResolution | None:
        normalized = str(allele or "").strip().upper()
        accession = MOUSE_MHC_I_ACCESSIONS.get(normalized)
        return self._mouse.get(accession or "")

    def resolve_mouse_ii(
        self, allele: object
    ) -> tuple[SequenceResolution, SequenceResolution] | None:
        normalized = str(allele or "").strip().upper()
        accessions = MOUSE_MHC_II_ACCESSIONS.get(normalized)
        if accessions is None:
            return None
        alpha = self._mouse.get(accessions[0])
        beta = self._mouse.get(accessions[1])
        return (alpha, beta) if alpha is not None and beta is not None else None

    def used_registry_rows(
        self, resolutions: Iterable[tuple[str, str, SequenceResolution]]
    ) -> list[dict[str, str]]:
        unique: dict[tuple[str, str, str], dict[str, str]] = {}
        for species, chain, item in resolutions:
            key = (species, chain, item.allele)
            unique[key] = {
                "species": species,
                "chain": chain,
                "resolved_allele": item.allele,
                "sequence": item.sequence,
                "source": item.source,
                "source_record": item.source_record,
                "resolution_method": item.method,
            }
        return [unique[key] for key in sorted(unique)]
