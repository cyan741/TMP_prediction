"""Pad sequences and tokenize the TCR and peptide-MHC model inputs.

Sequence widths come from the checkpoint or training configuration.
Sequences that exceed those widths raise an error instead of being truncated."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

PAD_RESIDUE = "X"

# Token id inserted between the two halves when plm_input == "sep".  Each family
# uses its own separator token ID, listed here for each supported tokenizer.
SEPARATOR_TOKEN = {
    "tape": 3,
    "protbert": 3,
    "esm2": 2,
    "AMPLIFY": 4,
}


@dataclass(frozen=True, slots=True)
class InputContract:
    """Fixed widths of one level's two halves."""

    level: str
    pep_max_len: int
    hla_max_len: int
    tcr_max_len: int
    tcr_chains: int

    @property
    def pmhc_width(self) -> int:
        return self.pep_max_len + (self.hla_max_len if self.uses_mhc else 0)

    @property
    def tcr_width(self) -> int:
        return self.tcr_max_len * self.tcr_chains

    @property
    def uses_mhc(self) -> bool:
        return self.level in {"2", "3", "4"}

    @property
    def total_residues(self) -> int:
        return self.pmhc_width + self.tcr_width

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "pep_max_len": self.pep_max_len,
            "hla_max_len": self.hla_max_len,
            "tcr_max_len": self.tcr_max_len,
            "tcr_chains": self.tcr_chains,
            "pmhc_width": self.pmhc_width,
            "tcr_width": self.tcr_width,
            "total_residues": self.total_residues,
        }


def contract_for(
    level: str, *, pep_max_len: int, tcr_max_len: int, hla_max_len: int = 34
) -> InputContract:
    """Widths for one level.

    Level I has no MHC and one chain.  Level II has an MHC and one chain, and
    uses a shared peptide-MHC representation in a single
    ``pep_max_len``-wide field rather than getting their own.  Levels III and IV
    give the peptide and MHC separate fields and carry two chains.
    """

    if level == "1":
        return InputContract("1", pep_max_len, 0, tcr_max_len, 1)
    if level == "2":
        # The legacy Level II layout reserves one combined peptide-MHC field.
        return InputContract("2", pep_max_len, 0, tcr_max_len, 1)
    if level in {"3", "4"}:
        return InputContract(level, pep_max_len, hla_max_len, tcr_max_len, 2)
    raise ValueError(f"unsupported level: {level!r}")


def _pad(value: str, width: int) -> str:
    """Right-pad with X.

    ``str.ljust`` does not truncate, so anything wider than the field silently
    widens downstream tensors and produces a ragged batch. Reject overlength
    inputs before padding so every record uses the same field widths.
    """

    text = str(value or "")
    if len(text) > width:
        raise ValueError(
            f"sequence of length {len(text)} exceeds the {width}-residue field"
        )
    return text.ljust(width, PAD_RESIDUE)


def pad_record(row: Mapping[str, Any], contract: InputContract) -> tuple[str, str]:
    """Return (tcr_side, pmhc_side) for one record, already padded."""

    peptide = str(row.get("pep", ""))
    if contract.level == "1":
        pmhc = _pad(peptide, contract.pep_max_len)
    elif contract.level == "2":
        pmhc = _pad(peptide + str(row.get("hla", "")), contract.pep_max_len)
    else:
        pmhc = _pad(peptide, contract.pep_max_len) + _pad(
            str(row.get("hla", "")), contract.hla_max_len
        )

    raw_tcr = str(row.get("ab", "") or row.get("beta", ""))
    chains = raw_tcr.split("/") if contract.tcr_chains > 1 else [raw_tcr]
    if len(chains) != contract.tcr_chains:
        raise ValueError(
            f"level {contract.level} expects {contract.tcr_chains} chain(s), got {len(chains)}"
        )
    tcr = "".join(_pad(chain, contract.tcr_max_len) for chain in chains)
    return tcr, pmhc


def _family(model_id: str) -> str:
    if "esm2" in model_id:
        return "esm2"
    if "AMPLIFY" in model_id:
        return "AMPLIFY"
    if model_id == "tape":
        return "tape"
    return "protbert"


def encode_batch(
    tokenizer: Any,
    tcr_side: Sequence[str],
    pmhc_side: Sequence[str],
    *,
    model_id: str,
    plm_input: str,
    device: Any,
) -> torch.Tensor:
    """Tokenize TCR first, followed by peptide-MHC, using model-specific separators."""

    family = _family(model_id)
    tokens: list[Any] = []
    for tcr, pmhc in zip(tcr_side, pmhc_side):
        joined = tcr + pmhc
        text = " ".join(joined) if family == "protbert" else joined
        encoded = tokenizer.encode(text)
        if plm_input == "sep":
            encoded = np.insert(
                np.asarray(encoded), len(joined) + 1, SEPARATOR_TOKEN[family]
            )
        tokens.append(np.asarray(encoded))
    widths = {len(item) for item in tokens}
    if len(widths) != 1:
        raise ValueError(f"ragged token batch: widths {sorted(widths)}")
    return torch.from_numpy(np.stack(tokens)).to(device)
