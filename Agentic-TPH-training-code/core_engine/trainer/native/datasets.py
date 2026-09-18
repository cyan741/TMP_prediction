"""Load fixed labels and generate random training negatives.

Candidate TCRs come from the workspace pool. The exclusion table covers known
positive and held-out peptide/TCR pairs so they cannot become training negatives."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .encoding import InputContract, pad_record


def load_blacklist(path: Path) -> dict[str, set[str]]:
    """Read a ``pep,ab`` blacklist into peptide -> known partners."""

    frame = pd.read_csv(Path(path), dtype=str, keep_default_na=False)
    missing = {"pep", "ab"} - set(frame.columns)
    if missing:
        raise ValueError(f"blacklist is missing columns: {sorted(missing)}")
    result: dict[str, set[str]] = {}
    for peptide, partner in frame[["pep", "ab"]].itertuples(index=False, name=None):
        result.setdefault(str(peptide), set()).add(str(partner))
    return result


@dataclass(slots=True)
class NegativeSampler:
    """Draw one mismatched TCR per positive, never one known to bind."""

    pool: Sequence[str]
    blacklist: Mapping[str, set[str]]
    beta_only: bool = False

    def __post_init__(self) -> None:
        if not self.pool:
            raise ValueError("the candidate pool is empty")

    def draw_many(self, peptide: str, rng: random.Random, count: int) -> list[str]:
        """count distinct negatives for one positive.

        Distinct because the same mismatched TCR drawn twice against one peptide
        is the same training example twice, not a wider negative set.
        """

        if count < 1:
            raise ValueError("negative count must be positive")
        known = self.blacklist.get(peptide, set())
        projected = (
            (candidate.split("/")[-1] if self.beta_only else candidate)
            for candidate in self.pool
        )
        admissible = list(
            dict.fromkeys(
                candidate for candidate in projected if candidate not in known
            )
        )
        if len(admissible) < count:
            raise RuntimeError(
                f"only {len(admissible)} of {count} negatives found for {peptide!r}; "
                "the admissible candidate pool is too small"
            )
        return rng.sample(admissible, count)


class PairedDataset(Dataset):
    """Labelled rows, used when the frame carries its own negatives."""

    def __init__(self, frame: pd.DataFrame, contract: InputContract) -> None:
        self.records = frame.to_dict("records")
        self.contract = contract

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[str, str, int]:
        row = self.records[index]
        tcr, pmhc = pad_record(row, self.contract)
        return tcr, pmhc, int(row["label"])


class RandomNegativeDataset(Dataset):
    """Positives only; one fresh negative is drawn per item per epoch."""

    def __init__(
        self,
        frame: pd.DataFrame,
        contract: InputContract,
        sampler: NegativeSampler,
        *,
        seed: int,
        deterministic: bool = False,
        negatives_per_positive: int = 1,
    ) -> None:
        self.records = frame.to_dict("records")
        self.contract = contract
        self.sampler = sampler
        self.seed = seed
        self.deterministic = deterministic
        self.negatives_per_positive = negatives_per_positive
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[str, tuple[str, ...], str]:
        row = self.records[index]
        tcr, pmhc = pad_record(row, self.contract)
        rng = (
            random.Random((self.seed, self.epoch, index).__hash__())
            if self.deterministic
            else random
        )
        raw = self.sampler.draw_many(str(row["pep"]), rng, self.negatives_per_positive)
        key = "ab" if self.contract.tcr_chains > 1 else "beta"
        negatives = []
        for candidate in raw:
            negative_row = dict(row)
            negative_row[key] = candidate
            padded, _ = pad_record(negative_row, self.contract)
            negatives.append(padded)
        return tcr, tuple(negatives), pmhc


class DistributedEvalSampler(DistributedSampler):
    """Shard evaluation rows without the padding used by train samplers.

    ``DistributedSampler(drop_last=False)`` repeats a few rows so every rank
    has the same number of samples.  Repeated rows bias global metrics, so
    evaluation keeps the sampler contract but assigns ``indices[rank::world]``
    and lets ranks have different batch counts.
    """

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=generator).tolist()
        return iter(indices[self.rank :: self.num_replicas])

    def __len__(self) -> int:
        return (
            len(self.dataset) - self.rank + self.num_replicas - 1
        ) // self.num_replicas


def _read(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [name for name in [*columns, "label"] if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame = frame[[*columns, "label"]].copy()
    frame["label"] = frame["label"].astype(int)
    return frame


def _pool(data_dir: Path) -> list[str]:
    path = Path(data_dir) / "tcr2candidates_pools.npy"
    if not path.is_file():
        raise FileNotFoundError(f"candidate pool not found: {path}")
    return [str(value) for value in np.load(path, allow_pickle=False).tolist()]


def _frame_blacklist(frame: pd.DataFrame, key: str) -> dict[str, set[str]]:
    return {
        str(peptide): set(map(str, group))
        for peptide, group in frame[frame["label"] == 1].groupby("pep")[key]
    }


def build_dataloaders(
    config: Any,
    *,
    splits: Iterable[str] = ("train", "valid", "test"),
    blacklist_path: Path | None = None,
    distributed: bool = False,
) -> dict[str, DataLoader]:
    """Build one loader per split, all sharing one candidate pool.

    ``blacklist_path`` supplies the corpus-wide screen. Without it each split
    screens against itself. The training entry point passes the workspace-wide
    exclusion table to prevent held-out pairs from becoming training negatives.
    """

    contract = InputContract(
        config.level,
        config.pep_max_len,
        config.hla_max_len if config.level in {"3", "4"} else 0,
        config.tcr_max_len,
        2 if config.level in {"3", "4"} else 1,
    )
    columns = list(config.component_columns)
    key = "ab" if contract.tcr_chains > 1 else "beta"
    data_dir = Path(config.data_dir)
    pool = _pool(data_dir)
    corpus_blacklist = load_blacklist(blacklist_path) if blacklist_path else None

    filenames = {
        "train": f"train_data_fold{config.fold}.csv",
        "valid": f"valid_data_fold{config.fold}.csv",
        "test": f"test_data_fold{config.fold}.csv",
        "unseen": "unseen_data.csv",
    }
    loaders: dict[str, DataLoader] = {}
    for split in splits:
        path = data_dir / filenames[split]
        if not path.is_file():
            if split in {"unseen"}:
                continue
            raise FileNotFoundError(f"{split} split not found: {path}")
        frame = _read(path, columns)
        if config.rand_neg:
            positives = frame[frame["label"] == 1].reset_index(drop=True)
            sampler = NegativeSampler(
                pool=pool,
                blacklist=corpus_blacklist
                if corpus_blacklist is not None
                else _frame_blacklist(frame, key),
                beta_only=contract.tcr_chains == 1,
            )
            dataset: Dataset = RandomNegativeDataset(
                positives,
                contract,
                sampler,
                seed=getattr(config, "sampling_seed", config.seed),
                deterministic=True,
                negatives_per_positive=getattr(config, "negatives_per_positive", 1),
            )
        else:
            dataset = PairedDataset(frame, contract)
        shuffle = split == "train"
        if distributed:
            torch_sampler = (
                DistributedSampler(dataset, shuffle=True)
                if split == "train"
                else DistributedEvalSampler(dataset, shuffle=False)
            )
        else:
            torch_sampler = None
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle and torch_sampler is None,
            sampler=torch_sampler,
            num_workers=config.num_workers,
            worker_init_fn=_seed_worker,
        )
    return loaders


def _seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
