from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import increment_workspaces as iw


def load_hitph_identity(root: Path, level: str) -> pd.DataFrame:
    """Load Hi-TpH positives in iw.load_hitph_corpus order with row-local identity.

    Level IV keeps ``ab`` as the model V-sequence pair and takes ``cdr3_ab``
    directly from the source row's explicit ``ab_cdr3`` field.  It never
    performs a global V-sequence reverse lookup, so distinct CDR3 variants
    sharing one V-sequence remain distinct observations.
    """
    if level not in {"III", "IV"}:
        raise ValueError(f"unsupported level: {level}")
    level_dir = root / "benchmarks_dataset" / f"level{'3' if level == 'III' else '4'}"
    frames: list[pd.DataFrame] = []
    for name in (
        "train_data_fold0",
        "valid_data_fold0",
        "test_data_fold0",
        "unseen_data",
        "external_data",
    ):
        path = level_dir / f"{name}.csv"
        if not path.is_file():
            continue
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "label" in raw.columns:
            raw = raw.loc[raw["label"].astype(str) == "1"].copy()
        else:
            raw = raw.copy()
        if level == "IV" and "ab_cdr3" not in raw.columns:
            raise KeyError(f"{path} lacks explicit ab_cdr3 for Level IV identity")
        model_ab = iw._norm(raw["ab"])
        cdr3_ab = model_ab if level == "III" else iw._norm(raw["ab_cdr3"])
        frames.append(
            pd.DataFrame(
                {
                    "pep": iw._norm(raw["pep"]),
                    "hla": iw._norm(raw["hla"]),
                    "hla.allele": iw._norm(raw["hla.allele"]),
                    "ab": model_ab,
                    "label": 1,
                    "beta": model_ab.str.split("/").str[-1],
                    "cdr3_ab": cdr3_ab,
                    "observation_id": [f"{path.name}:{int(row)}" for row in raw.index],
                }
            )
        )
    if not frames:
        raise FileNotFoundError(f"no Hi-TpH level {level} files under {level_dir}")
    return pd.concat(frames, ignore_index=True)
