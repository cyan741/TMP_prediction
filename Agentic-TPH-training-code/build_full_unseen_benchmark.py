"""Whole-corpus unseen-peptide split with symmetric evaluation size constraints."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from prepare import load_input
from core_engine.trainer.fixed_splits import cdr3_group, deduplicate_evaluation
from core_engine.trainer.workspaces import identity, model_frame, save_workspace, write_json


def choose_split(counts, seed=42, peptides_per_eval=500, cap=12269, balance_tolerance=.05, max_attempts=100000):
    """Reject whole uniformly shuffled partitions using counts only, never distance/scores."""
    if peptides_per_eval < 1 or 2 * peptides_per_eval >= len(counts):
        raise ValueError('Need nonempty train and two evaluation peptide sets')
    rng = np.random.RandomState(seed)
    values = counts[['source_positive_rows', 'unique_positive_pairs']].to_numpy()
    for attempt in range(1, max_attempts+1):
        order = rng.permutation(len(counts))
        vi, ti = order[:peptides_per_eval], order[peptides_per_eval:2*peptides_per_eval]
        v, t = values[vi].sum(axis=0), values[ti].sum(axis=0)
        if max(v[0], t[0]) <= cap and np.all(np.abs(v-t) <= balance_tolerance * np.maximum(v,t)):
            return set(counts.iloc[vi].pep), set(counts.iloc[ti].pep), attempt
    raise ValueError('No feasible split found; adjust count/size constraints explicitly')


def fixed_evaluation(positive, train, known, seed):
    """Original same-peptide/MHC random TCR mismatch; reject duplicate negative groups.

    Donors are train (ab,cdr3_ab) pairs, as in workspaces.fixed_negatives.
    Conditioning on unique groups preserves exact 1:1 after evaluation dedup.
    """
    rng=np.random.RandomState(seed)
    donors=train[['ab','cdr3_ab']].drop_duplicates().reset_index(drop=True)
    donor_groups=donors.cdr3_ab.map(cdr3_group)
    pieces=[]
    for peptide, group in positive.groupby('pep',sort=True):
        excluded=known.get(peptide,set())
        mask=~donor_groups.isin(excluded)
        admissible=donors.loc[mask].reset_index(drop=True)
        groups=donor_groups.loc[mask].to_numpy()
        if not len(admissible):
            raise ValueError(f'No admissible TCR negatives for {peptide}')
        available_group_count=len(set(groups))
        used={}
        picks=[]
        for allele in group['hla.allele']:
            seen=used.setdefault(allele,set())
            if len(seen) >= available_group_count:
                raise ValueError(f'Not enough distinct negative TCR groups for {peptide}/{allele}')
            while True:
                index=int(rng.randint(len(admissible)))
                if groups[index] not in seen:
                    seen.add(groups[index]);picks.append(index);break
        negative=group.copy()
        negative['ab']=admissible.iloc[picks].ab.to_numpy()
        negative['cdr3_ab']=admissible.iloc[picks].cdr3_ab.to_numpy()
        negative['beta']=negative.ab.str.split('/').str[-1]
        negative['label']=0
        negative['negative_type']='derived_random_mismatch'
        pieces.append(negative)
    frame,report=deduplicate_evaluation(identity(pd.concat([positive,*pieces],ignore_index=True)))
    frame=model_frame(frame)
    if (frame.label==1).sum() != (frame.label==0).sum():
        raise AssertionError('Fixed evaluation must be exactly 1:1 after deduplication')
    return frame,report


def build(source, output, seed=42, peptides_per_eval=500, cap_basis='positive', balance_tolerance=.05):
    if output.exists():
        raise FileExistsError(output)
    corpus,filters=load_input({'path':str(source)},'III',positive=True)
    raw=pd.read_csv(source,dtype=str,keep_default_na=False)
    if len(raw)!=len(corpus):
        raise ValueError('Whole-corpus split requires all source observations to be supported')
    unique,dedup=deduplicate_evaluation(identity(corpus))
    unique=model_frame(unique)
    counts=corpus.groupby('pep').size().rename('source_positive_rows').to_frame().join(unique.groupby('pep').size().rename('unique_positive_pairs')).reset_index().sort_values('pep').reset_index(drop=True)
    overall_cap=math.floor(len(corpus)*.1)
    # In labeled mode this conservative raw-positive cap guarantees 2*unique <= overall_cap.
    cap=overall_cap if cap_basis=='positive' else overall_cap//2
    valid_set,test_set,attempt=choose_split(counts,seed,peptides_per_eval,cap,balance_tolerance)
    train=corpus.loc[~corpus.pep.isin(valid_set|test_set)].copy()
    vp=unique.loc[unique.pep.isin(valid_set)].copy()
    tp=unique.loc[unique.pep.isin(test_set)].copy()
    known={p:set(g.cdr3_ab.map(cdr3_group)) for p,g in corpus.groupby('pep')}
    valid,vd=fixed_evaluation(vp,train,known,seed+1000)
    test,td=fixed_evaluation(tp,train,known,seed+2000)
    sets={'train':set(train.pep),'valid':valid_set,'test':test_set}
    assert not (sets['train']&valid_set or sets['train']&test_set or valid_set&test_set)
    assert set.union(*sets.values())==set(corpus.pep)
    output.mkdir(parents=True)
    save_workspace(output/'hitph',train,valid,test,dict(level='III',backend='hla-i',pep_max_len=13,tcr_max_len=28,task='unseen',grouping='peptide',test_grouping='peptide',reference='hitph',split_seed=seed,validation_negative_seed=seed+1000,test_negative_seed=seed+2000,capability_filter=filters))
    # Requested train+valid positive blacklist. Test positives screen test generation
    # separately; unseen peptide guarantees their keys are irrelevant to train/val draws.
    positives=corpus.loc[~corpus.pep.isin(test_set)].copy()
    exact=positives[['pep','ab']].drop_duplicates().reset_index(drop=True)
    exact.to_csv(output/'hitph/positive_blacklist_train_valid.csv',index=False)
    blocked=positives[['pep','cdr3_ab']].copy()
    blocked['group']=blocked.cdr3_ab.map(cdr3_group)
    pool=train[['ab','cdr3_ab']].drop_duplicates().copy()
    pool['group']=pool.cdr3_ab.map(cdr3_group)
    expanded=blocked[['pep','group']].drop_duplicates().merge(pool[['ab','group']].drop_duplicates(),on='group')[['pep','ab']]
    blacklist=pd.concat([exact,expanded],ignore_index=True).drop_duplicates().reset_index(drop=True)
    blacklist.to_csv(output/'hitph/negative_blacklist_corpus.csv',index=False)
    metadata=json.loads((output/'hitph/workspace_report.json').read_text())
    metadata['negative_exclusion']='train+valid positives across MHC; endpoint-equivalent expansion to training TCR pool; test positives screened separately during fixed test construction'
    write_json(output/'hitph/workspace_report.json',metadata)
    counts['split']=counts.pep.map(lambda p:next(k for k,v in sets.items() if p in v))
    counts.to_csv(output/'peptide_frequencies.csv',index=False)
    rows=[]
    train_peptides=sorted(sets['train'])
    frequency=counts.set_index('pep').source_positive_rows.to_dict()
    for split,frame in [('valid',valid),('test',test)]:
        for peptide,group in frame.groupby('pep',sort=True):
            distances=[(Levenshtein.distance(peptide,p),p) for p in train_peptides]
            distance,nearest=min(distances)
            assert distance>0
            rows.append(dict(split=split,pep=peptide,min_edit_distance=distance,nearest_train_peptide=nearest,nearest_tie_count=sum(d==distance for d,_ in distances),positives=int((group.label==1).sum()),negatives=int((group.label==0).sum()),source_positive_rows=int(frequency[peptide])))
    distances=pd.DataFrame(rows)
    distances.to_csv(output/'peptide_distances.csv',index=False)
    strata=distances.groupby(['split','min_edit_distance']).agg(peptides=('pep','size'),positives=('positives','sum'),negatives=('negatives','sum'),source_positive_rows=('source_positive_rows','sum')).reset_index()
    strata['total_rows']=strata.positives+strata.negatives
    strata.to_csv(output/'distance_strata.csv',index=False)
    for split,frame in [('valid',valid),('test',test)]:
        frame.merge(distances.loc[distances.split==split,['pep','min_edit_distance','nearest_train_peptide']],on='pep',validate='many_to_one').to_csv(output/f'{split}_annotated.csv',index=False)
    split_counts={s:dict(peptides=len(sets[s]),source_positive_rows=int(counts.loc[counts.split==s,'source_positive_rows'].sum()),positive_rows=int((f.label==1).sum()),negative_rows=int((f.label==0).sum()),labeled_rows=len(f)) for s,f in [('train',train),('valid',valid),('test',test)]}
    for s in ['valid','test']:
        assert split_counts[s]['source_positive_rows']<=overall_cap
        if cap_basis=='labeled':assert split_counts[s]['labeled_rows']<=overall_cap
    result=dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_rows=len(corpus),source_peptides=len(counts),seed=seed,rng='numpy.random.RandomState(seed).permutation on lexical peptide order; first feasible partition',sampling='whole-corpus uniform peptide permutation conditioned on evaluation counts and balance; no frequency-tail/distance/model-performance selection',accepted_attempt=attempt,peptides_per_eval=peptides_per_eval,cap_basis=cap_basis,maximum_rows_per_eval=overall_cap,raw_positive_sampling_cap=cap,balance_tolerance=balance_tolerance,forced_train_peptides=counts.loc[counts.source_positive_rows>cap,['pep','source_positive_rows']].to_dict('records'),split_counts=split_counts,corpus_dedup=dedup,valid_dedup=vd,test_dedup=td,training_negatives='dynamic 1:1 per epoch',validation_negatives='fixed unique 1:1, seed+1000',test_negatives='fixed unique 1:1, seed+2000',negative_donor_pool='unique training (ab,cdr3_ab) records; tcr2candidates_pools.npy contains sorted unique training ab',positive_blacklist='positive_blacklist_train_valid.csv contains exact train+valid positive pep/ab',runtime_blacklist='negative_blacklist_corpus.csv additionally expands endpoint-equivalent identities to training model TCR donors',negative_sampling_difference='same original random mismatch donor/exclusion scheme, with rejection of repeated peptide/MHC/endpoint-TCR negatives to preserve exact 1:1 after deduplication',distance_definition='exact minimum unit-cost Levenshtein distance to final training peptides, validation excluded',distance_strata=strata.to_dict('records'),peptide_overlap=0)
    result.update(blacklist_exact_pairs=len(exact),blacklist_runtime_pairs=len(blacklist),training_tcr_pool_size=int(train.ab.nunique()),positive_blacklist_source='all original corpus positives assigned to train+valid, before evaluation deduplication')
    write_json(output/'benchmark_report.json',result)
    print(pd.DataFrame(split_counts).T.to_string())
    print('accepted attempt',attempt,'cap basis',cap_basis)
    print(strata.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('corpora/hitph260630.csv'))
    p.add_argument('--output',type=Path,default=Path('workspaces/unseen_full_seed42'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--peptides-per-eval',type=int,default=500)
    p.add_argument('--cap-basis',choices=['positive','labeled'],default='positive')
    a=p.parse_args();build(a.source,a.output,a.seed,a.peptides_per_eval,a.cap_basis)
