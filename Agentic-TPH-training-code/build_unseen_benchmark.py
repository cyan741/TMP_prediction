"""Build a frequency-tail unseen-peptide benchmark using existing split utilities."""
import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from prepare import load_input
from core_engine.trainer.fixed_splits import select_validation, deduplicate_evaluation
from core_engine.trainer.workspaces import identity, model_frame, fixed_negatives, save_workspace, write_json


def build(source, output, seed=42, tail_fraction=0.8, test_peptides=100, validation_fraction=0.1):
    if output.exists():
        raise FileExistsError(output)
    corpus, filters = load_input({'path': str(source)}, 'III', positive=True)
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    raw['pep'] = raw.pep.str.strip().str.upper()
    if set(raw.label) != {'1'}:
        raise ValueError('Source must contain only explicit positives')
    # Frequencies reflect source observations, including duplicates, before filtering.
    frequency = raw.groupby('pep').size().rename('source_positive_rows').reset_index()
    frequency = frequency.sort_values(['source_positive_rows', 'pep']).reset_index(drop=True)
    tail_size = math.floor(len(frequency) * tail_fraction)
    tail = frequency.iloc[:tail_size]
    chosen = np.random.RandomState(seed).choice(tail.pep.to_numpy(), test_peptides, replace=False)
    test_set = set(chosen)
    unsupported = raw.loc[raw.pep.isin(test_set)]
    if len(unsupported) != len(corpus.loc[corpus.pep.isin(test_set)]):
        raise ValueError('Selected test observations exceed Level III capacity; do not silently filter')
    test_positive, test_dedup = deduplicate_evaluation(identity(corpus.loc[corpus.pep.isin(test_set)]))
    test_positive = model_frame(test_positive)
    remaining = corpus.loc[~corpus.pep.isin(test_set)].copy()
    valid_positive, valid_split = select_validation(identity(remaining), peers={'hitph': identity(remaining)}, grouping='peptide', seed=seed, fraction=validation_fraction)
    valid_positive = model_frame(valid_positive)
    train = remaining.loc[~remaining.pep.isin(valid_positive.pep)].copy()
    known = {p: set(g.ab) for p, g in corpus.groupby('pep')}
    cdr_known = {p: set(g.ab) for p, g in identity(corpus).groupby('pep')}
    def evaluation(positive, negative_seed):
        negative = fixed_negatives(positive, train, known, negative_seed, cdr_known)
        negative['negative_type'] = 'derived_random_mismatch'
        frame, report = deduplicate_evaluation(identity(pd.concat([positive, negative], ignore_index=True)))
        return model_frame(frame), report
    valid, valid_dedup = evaluation(valid_positive, seed + 1000)
    test, test_final_dedup = evaluation(test_positive, seed + 2000)
    sets = {'train': set(train.pep), 'valid': set(valid.pep), 'test': set(test.pep)}
    assert not (sets['train'] & sets['valid'] or sets['train'] & sets['test'] or sets['valid'] & sets['test'])
    assert all(set(f.label) == {0, 1} for f in [valid, test])
    output.mkdir(parents=True)
    save_workspace(output / 'hitph', train, valid, test, dict(level='III', backend='hla-i', pep_max_len=13, tcr_max_len=28, task='unseen', grouping='peptide', test_grouping='peptide', reference='hitph', split_seed=seed, validation_negative_seed=seed+1000, test_negative_seed=seed+2000, capability_filter=filters))
    frequency['in_tail'] = frequency.index < tail_size
    frequency['split'] = frequency.pep.map(lambda p: next((k for k, v in sets.items() if p in v), 'excluded'))
    frequency.to_csv(output / 'peptide_frequencies.csv', index=False)
    rows = []
    train_peptides = sorted(sets['train'])
    for split, frame in [('test', test), ('valid', valid)]:
        for peptide, group in frame.groupby('pep', sort=True):
            distances = [(Levenshtein.distance(peptide, p), p) for p in train_peptides]
            distance, nearest = min(distances)
            rows.append(dict(split=split, pep=peptide, nearest_train_peptide=nearest, min_edit_distance=distance, nearest_tie_count=sum(d == distance for d, _ in distances), positives=int((group.label == 1).sum()), negatives=int((group.label == 0).sum()), source_positive_rows=int((raw.pep == peptide).sum())))
    distances = pd.DataFrame(rows)
    distances.to_csv(output / 'peptide_distances.csv', index=False)
    strata = distances.groupby(['split', 'min_edit_distance']).agg(peptides=('pep', 'size'), positives=('positives', 'sum'), negatives=('negatives', 'sum'), source_positive_rows=('source_positive_rows', 'sum')).reset_index()
    strata['total_rows'] = strata.positives + strata.negatives
    strata.to_csv(output / 'distance_strata.csv', index=False)
    for split, frame in [('test', test), ('valid', valid)]:
        frame.merge(distances.loc[distances.split == split, ['pep', 'min_edit_distance', 'nearest_train_peptide']], on='pep', validate='many_to_one').to_csv(output / f'{split}_annotated.csv', index=False)
    report = dict(source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), seed=seed, sampling_rng='numpy.random.RandomState.choice without replacement', frequency_basis='raw positive rows; ascending count, then peptide lexicographic order', tail_fraction=tail_fraction, tail_peptides=tail_size, corpus_peptides=len(frequency), tail_max_frequency=int(tail.source_positive_rows.max()), tail_boundary_tied_peptides=int((frequency.source_positive_rows == tail.source_positive_rows.max()).sum()), selected_test_peptides=sorted(test_set), selected_test_source_rows=int(raw.pep.isin(test_set).sum()), test_positive_dedup=test_dedup, test_final_dedup=test_final_dedup, validation_split=valid_split, validation_dedup=valid_dedup, split_counts={name: dict(rows=len(f), peptides=int(f.pep.nunique()), positives=int((f.label == 1).sum()), negatives=int((f.label == 0).sum())) for name,f in [('train',train),('valid',valid),('test',test)]}, distance_definition='minimum unnormalized Levenshtein distance to FINAL training peptides; unit insertion/deletion/substitution cost; lexical nearest tie break', distance_strata=strata.to_dict('records'), peptide_overlap=0)
    write_json(output / 'benchmark_report.json', report)
    config = dict(output=str(output / 'reprepared'), level='III', backend='hla-i', task='unseen', validation_grouping='peptide', test_grouping='peptide', reference='hitph', validation_fraction=validation_fraction, split_seed=seed, validation_negative_seed=seed+1000, corpora={'hitph': {'path': str(source.resolve()), 'format': 'canonical'}}, test={'path': str((output / 'hitph/test_data_fold0.csv').resolve()), 'format': 'canonical'})
    write_json(output / 'prepare_config.json', config)
    print(pd.DataFrame(report['split_counts']).T.to_string())
    print(strata.to_string(index=False))
    print('Selected raw positive rows:', report['selected_test_source_rows'])
    print('Corpus / tail peptides:', len(frequency), tail_size)
    print('Tail maximum frequency:', report['tail_max_frequency'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('corpora/hitph260630.csv'))
    parser.add_argument('--output', type=Path, default=Path('workspaces/unseen_tail80_seed42'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    build(args.source, args.output, args.seed)
