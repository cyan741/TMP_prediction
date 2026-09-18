"""Report labeled prediction metrics by nearest FINAL-training peptide distance."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from core_engine.trainer.workspaces import write_json


def metrics(group):
    labels = group.label.astype(int).to_numpy()
    scores = group.score.astype(float).to_numpy()
    result = dict(rows=len(group), peptides=int(group.pep.nunique()), positives=int((labels == 1).sum()), negatives=int((labels == 0).sum()), auroc=None, auprc=None)
    if len(set(labels)) == 2:
        result.update(auroc=float(roc_auc_score(labels, scores)), auprc=float(average_precision_score(labels, scores)))
    if 'threshold' in group:
        thresholds = group.threshold.astype(float).to_numpy()
        if not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any() or len(np.unique(thresholds)) != 1:
            raise ValueError('Use one frozen validation threshold in [0, 1]')
        predicted = scores >= thresholds
        result.update(f1=float(f1_score(labels, predicted, zero_division=0)), mcc=float(matthews_corrcoef(labels, predicted)))
    per_peptide = [metrics_auc(g) for _, g in group.groupby('pep') if g.label.nunique() == 2]
    result['macro_peptide_auroc'] = float(np.mean([m[0] for m in per_peptide])) if per_peptide else None
    result['macro_peptide_auprc'] = float(np.mean([m[1] for m in per_peptide])) if per_peptide else None
    result['macro_eligible_peptides'] = len(per_peptide)
    return result


def metrics_auc(g):
    return roc_auc_score(g.label, g.score), average_precision_score(g.label, g.score)


def report(predictions, benchmark, output):
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_csv(predictions, keep_default_na=False)
    expected = pd.read_csv(benchmark / 'hitph/test_data_fold0.csv', keep_default_na=False)
    if len(frame) != len(expected) or any(c not in frame for c in ['pep','hla','hla.allele','ab','label','score']):
        raise ValueError('Predictions must retain all labeled fixed-test rows and identity columns')
    for column in ['pep','hla','hla.allele','ab','label']:
        if frame[column].astype(str).tolist() != expected[column].astype(str).tolist():
            raise ValueError(f'Fixed-test row correspondence mismatch: {column}')
    scores = frame.score.astype(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Scores must be finite probabilities in [0, 1]')
    distance = pd.read_csv(benchmark / 'peptide_distances.csv')
    frame = frame.drop(columns=['min_edit_distance','nearest_train_peptide'], errors='ignore').merge(distance.loc[distance.split == 'test', ['pep','min_edit_distance']], on='pep', validate='many_to_one')
    if frame.min_edit_distance.isna().any():
        raise ValueError('Missing peptide distance')
    result = dict(overall=metrics(frame), by_edit_distance={str(int(d)): metrics(g) for d,g in frame.groupby('min_edit_distance')}, threshold_source='prediction CSV; must be frozen on validation data', aggregation='row-pooled and equally weighted per-peptide metrics; single-label groups return null AUC')
    write_json(output, result)
    print(pd.DataFrame(result['by_edit_distance']).T.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, default=Path('workspaces/unseen_tail80_seed42'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report(args.predictions, args.benchmark, args.output)
