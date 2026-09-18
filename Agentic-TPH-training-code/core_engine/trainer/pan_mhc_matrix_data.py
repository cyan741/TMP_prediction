from __future__ import annotations

import random
from typing import Any, Iterator

import pandas as pd
from torch.utils.data import Sampler

DOMAINS = ("human_I", "human_II", "nonhuman_I", "nonhuman_II")
PEPTIDE_MAX_LENGTH = 30
TCR_MAX_LENGTH = 40
MHC_ALPHA_MAX_LENGTH = 400
MHC_BETA_MAX_LENGTH = 300
MHC_PACKED_LENGTH = 700


class DomainBalancedSampler(Sampler[int]):
    """Downsample each included domain/label cell to the smallest cell."""

    def __init__(
        self, frame: pd.DataFrame, *, domains: tuple[str, ...], seed: int
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.domains = domains
        self.seed = seed
        self.groups = {
            f"{domain}:label{label}": list(
                self.frame.index[
                    (self.frame["domain"] == domain) & (self.frame["label"] == label)
                ]
            )
            for domain in domains
            for label in (0, 1)
        }
        missing = [name for name, indices in self.groups.items() if not indices]
        if missing:
            raise ValueError(f"balanced sampler empty cells: {missing}")
        self.target_per_cell = min(map(len, self.groups.values()))

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        indices: list[int] = []
        for name in sorted(self.groups):
            values = self.groups[name].copy()
            rng.shuffle(values)
            indices.extend(values[: self.target_per_cell])
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.target_per_cell * len(self.groups)

    def report(self) -> dict[str, Any]:
        return {
            "domains": list(self.domains),
            "target_per_cell": self.target_per_cell,
            "sampled_rows": len(self),
            "source_counts": {
                name: len(indices) for name, indices in self.groups.items()
            },
        }
