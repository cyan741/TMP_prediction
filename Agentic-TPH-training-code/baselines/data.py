"""Training-only data preparation shared by single-sequence adapters."""
import numpy as np
import pandas as pd
from core_engine.trainer.fixed_splits import cdr3_group


def read_csv(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty:
        raise ValueError(f'Empty input: {path}')
    return frame


def labels(frame):
    if 'label' not in frame or not frame.label.isin(['0','1',0,1]).all():
        raise ValueError('Training/validation require explicit binary labels')
    return frame.label.astype(int)


def projection_conflicts(converted, y):
    return int(converted.assign(label=y.to_numpy()).groupby(['peptide','tcr']).label.nunique().gt(1).sum())


def prepare_training(train, valid, converted_train, converted_valid, exclusion, seed):
    """Materialize fixed train negatives if source is positive-only; preserve val."""
    train_y, valid_y = labels(train), labels(valid)
    if set(valid_y) != {0,1}:
        raise ValueError('Validation requires both labels for AUROC checkpoint selection')
    if set(converted_train.peptide) & set(converted_valid.peptide):
        raise ValueError('Unseen-peptide training/validation overlap')
    result = converted_train.copy()
    result['label'] = train_y.to_numpy()
    if set(train_y) == {1}:
        if exclusion is None:
            raise ValueError('Positive-only train requires negative_exclusion CSV; never infer negatives without exclusions')
        blocked = {}
        observed = pd.concat([converted_train, converted_valid, exclusion], ignore_index=True)
        for p,g in observed.groupby('peptide'):
            blocked[p] = set(g.tcr.map(cdr3_group))
        donors = sorted(set(converted_train.tcr))
        donor_groups = np.array([cdr3_group(t) for t in donors])
        rng = np.random.RandomState(seed)
        negative = converted_train.copy()
        for p,g in converted_train.groupby('peptide',sort=True):
            admissible = np.array(donors)[~np.isin(donor_groups,list(blocked[p]))]
            if not len(admissible):
                raise ValueError(f'No admissible training negative for peptide {p}')
            negative.loc[g.index,'tcr'] = admissible[rng.randint(len(admissible),size=len(g))]
        negative['label'] = 0
        negative['row_id'] = np.arange(len(result),len(result)*2)
        result = pd.concat([result,negative],ignore_index=True)
    elif set(train_y) != {0,1}:
        raise ValueError('Train must contain positives')
    if projection_conflicts(result,result.label):
        raise ValueError('Training has conflicting labels after single-chain projection')
    validation = converted_valid.copy()
    validation['label'] = valid_y.to_numpy()
    return result, validation
