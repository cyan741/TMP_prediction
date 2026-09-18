"""Build whole-corpus, frequency-stratified, peptide-disjoint train/val/test."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from prepare import load_input
from core_engine.trainer.fixed_splits import deduplicate_evaluation
from core_engine.trainer.workspaces import identity, model_frame, fixed_negatives, save_workspace, write_json

BINS = ['1', '2', '3-5', '6-10', '11-25', '>25']


def frequency_bin(n):
    for bound, name in [(1,'1'), (2,'2'), (5,'3-5'), (10,'6-10'), (25,'11-25')]:
        if n <= bound:
            return name
    return '>25'


def allocate_peptides(frequency, evaluation_peptides=500, seed=42):
    """Equal val/test largest-remainder quotas; sorted inputs, seeded shuffles."""
    if frequency.pep.duplicated().any() or (frequency.unique_positive_pairs < 1).any():
        raise ValueError('Frequency table needs one row per peptide and positive counts')
    if evaluation_peptides < 1 or 2 * evaluation_peptides >= len(frequency):
        raise ValueError('Evaluation peptide count must leave nonempty training')
    result = frequency.sort_values('pep').reset_index(drop=True).copy()
    result['frequency_bin'] = result.unique_positive_pairs.map(frequency_bin)
    sizes = {b: int((result.frequency_bin == b).sum()) for b in BINS}
    targets = {b: evaluation_peptides * sizes[b] / len(result) for b in BINS}
    quota = {b: int(np.floor(targets[b])) for b in BINS}
    remaining = evaluation_peptides - sum(quota.values())
    order = sorted(BINS, key=lambda b: (-(targets[b] - quota[b]), BINS.index(b)))
    for b in order:
        if remaining and 2 * (quota[b] + 1) <= sizes[b]:
            quota[b] += 1
            remaining -= 1
    if remaining or any(2 * quota[b] > sizes[b] for b in BINS):
        raise ValueError('Cannot satisfy equal per-bin validation/test quotas')
    rng = np.random.RandomState(seed)
    result['split'] = 'train'
    for b in BINS:
        indices = result.index[result.frequency_bin == b].to_numpy()
        rng.shuffle(indices)
        q = quota[b]
        result.loc[indices[:q], 'split'] = 'test'
        result.loc[indices[q:2*q], 'split'] = 'valid'
    return result, quota


def build(source, output, seed=42, evaluation_peptides=500):
    if output.exists():
        raise FileExistsError(output)
    corpus, filters = load_input({'path':str(source)}, 'III', positive=True)
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    raw['pep'] = raw.pep.str.strip().str.upper()
    if len(corpus) != len(raw):
        raise ValueError(f'Whole-corpus benchmark requires all rows supported: {filters}')
    unique, source_dedup = deduplicate_evaluation(identity(corpus))
    unique = model_frame(unique)
    frequency = unique.groupby('pep').size().rename('unique_positive_pairs').reset_index()
    frequency['source_positive_rows'] = frequency.pep.map(raw.groupby('pep').size())
    frequency, quotas = allocate_peptides(frequency, evaluation_peptides, seed)
    sets = {s:set(frequency.loc[frequency.split == s,'pep']) for s in ['train','valid','test']}
    train = corpus.loc[corpus.pep.isin(sets['train'])].copy()
    positives = {s:unique.loc[unique.pep.isin(sets[s])].copy() for s in ['valid','test']}
    known = {p:set(g.ab) for p,g in corpus.groupby('pep')}
    cdr_known = {p:set(g.ab) for p,g in identity(corpus).groupby('pep')}
    evaluations, dedup_reports = {}, {}
    for split, negative_seed in [('valid',seed+1000),('test',seed+2000)]:
        print(f'Generating {split}: {len(positives[split])} positive groups, {len(sets[split])} peptides', flush=True)
        negative = fixed_negatives(positives[split], train, known, negative_seed, cdr_known)
        negative['negative_type'] = 'derived_random_mismatch'
        frame, dedup_reports[split] = deduplicate_evaluation(identity(pd.concat([positives[split],negative],ignore_index=True)))
        evaluations[split] = model_frame(frame)
        if set(evaluations[split].label) != {0,1}:
            raise ValueError(f'{split} requires both labels')
    assert not (sets['train'] & sets['valid'] or sets['train'] & sets['test'] or sets['valid'] & sets['test'])
    assert set.union(*sets.values()) == set(corpus.pep)
    output.mkdir(parents=True)
    frames = {'train':train, **evaluations}
    save_workspace(output/'hitph',train,evaluations['valid'],evaluations['test'],dict(level='III',backend='hla-i',pep_max_len=13,tcr_max_len=28,task='unseen',grouping='peptide',test_grouping='peptide',reference='hitph',split_seed=seed,validation_negative_seed=seed+1000,test_negative_seed=seed+2000,capability_filter=filters,validation_selection='fixed peptide count, frequency-stratified',benchmark_report=str((output/'benchmark_report.json').resolve())))
    frequency.to_csv(output/'peptide_frequencies.csv',index=False)
    allocation = frequency.groupby(['frequency_bin','split']).agg(peptides=('pep','size'),unique_positive_pairs=('unique_positive_pairs','sum'),source_positive_rows=('source_positive_rows','sum')).reset_index()
    allocation.to_csv(output/'frequency_strata.csv',index=False)
    for split in sets:
        pd.DataFrame({'pep':sorted(sets[split])}).to_csv(output/f'{split}_peptides.csv',index=False)
    print('Computing distances to FINAL training peptides',flush=True)
    distances=[]
    reference=sorted(sets['train'])
    for split,frame in evaluations.items():
        for p,g in frame.groupby('pep',sort=True):
            pairs=[(Levenshtein.distance(p,q),q) for q in reference]
            d,nearest=min(pairs)
            assert d > 0
            distances.append(dict(split=split,pep=p,nearest_train_peptide=nearest,min_edit_distance=d,nearest_tie_count=sum(v==d for v,_ in pairs),positives=int((g.label==1).sum()),negatives=int((g.label==0).sum()),source_positive_rows=int(frequency.loc[frequency.pep==p,'source_positive_rows'].iloc[0])))
    distance=pd.DataFrame(distances)
    distance.to_csv(output/'peptide_distances.csv',index=False)
    strata=distance.groupby(['split','min_edit_distance']).agg(peptides=('pep','size'),positives=('positives','sum'),negatives=('negatives','sum'),source_positive_rows=('source_positive_rows','sum')).reset_index()
    strata['total_rows']=strata.positives+strata.negatives
    strata.to_csv(output/'distance_strata.csv',index=False)
    for split,frame in evaluations.items():
        frame.merge(distance.loc[distance.split==split,['pep','min_edit_distance','nearest_train_peptide']],on='pep',validate='many_to_one').to_csv(output/f'{split}_annotated.csv',index=False)
    report=dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),seed=seed,source_rows=len(raw),corpus_peptides=len(frequency),source_dedup=source_dedup,frequency_basis='unique peptide/MHC/endpoint-normalized paired-TCR positive groups',frequency_bins=BINS,allocation='largest remainder with equal val/test per-bin peptide quotas; sorted peptide input; RandomState(seed) shuffles; test first, validation second, remainder training',evaluation_peptides=evaluation_peptides,per_evaluation_bin_quotas=quotas,frequency_strata=allocation.to_dict('records'),split_counts={s:dict(rows=len(f),peptides=int(f.pep.nunique()),positives=int((f.label==1).sum()),negatives=int((f.label==0).sum()),source_positive_rows=int(frequency.loc[frequency.split==s,'source_positive_rows'].sum())) for s,f in frames.items()},evaluation_dedup=dedup_reports,negative_seeds={'valid':seed+1000,'test':seed+2000},distance_definition='minimum exact unnormalized Levenshtein distance to FINAL training peptides only; unit edit costs; lexical nearest tie break',distance_strata=strata.to_dict('records'),peptide_overlap=0,runtime_versions={'numpy':np.__version__,'pandas':pd.__version__})
    write_json(output/'benchmark_report.json',report)
    write_json(output/'build_config.json',dict(source=str(source.resolve()),output=str(output.resolve()),seed=seed,evaluation_peptides=evaluation_peptides))
    print(pd.DataFrame(report['split_counts']).T.to_string())
    print(strata.to_string(index=False))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('corpora/hitph260630.csv'))
    p.add_argument('--output',type=Path,default=Path('workspaces/unseen_stratified_seed42'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--evaluation-peptides',type=int,default=500)
    args=p.parse_args()
    build(args.source,args.output,args.seed,args.evaluation_peptides)
