"""Regenerate a fixed unseen-peptide assignment using original repository samplers."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from prepare import load_input
from core_engine.trainer.fixed_splits import deduplicate_evaluation
from core_engine.trainer.workspaces import identity, model_frame, fixed_negatives, save_workspace, write_json


def regenerate(source, assignment, output):
    if output.exists():
        raise FileExistsError(output)
    previous=json.loads((assignment/'benchmark_report.json').read_text())
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=previous['source_sha256']:
        raise ValueError('Source differs from fixed-assignment benchmark')
    corpus,filters=load_input({'path':str(source)},'III',positive=True)
    frequencies=pd.read_csv(assignment/'peptide_frequencies.csv',keep_default_na=False)
    if frequencies.pep.duplicated().any() or set(frequencies.pep)!=set(corpus.pep) or set(frequencies.split)!={'train','valid','test'}:
        raise ValueError('Invalid peptide assignment')
    membership=frequencies.set_index('pep').split
    corpus['benchmark_split']=corpus.pep.map(membership)
    train=corpus.loc[corpus.benchmark_split=='train'].drop(columns='benchmark_split').copy()
    positives={}
    reports={}
    for split in ['valid','test']:
        raw=corpus.loc[corpus.benchmark_split==split].drop(columns='benchmark_split')
        dedup,report=deduplicate_evaluation(identity(raw))
        positives[split]=model_frame(dedup)
        reports[split+'_positive_dedup']=report
    full=corpus.drop(columns='benchmark_split')
    known={p:set(g.ab) for p,g in full.groupby('pep')}
    cdr_known={p:set(g.ab) for p,g in identity(full).groupby('pep')}
    frames={'train':train}
    seed=previous['seed']
    for split,negative_seed in [('valid',seed+1000),('test',seed+2000)]:
        print(f'Generating {split} negatives with original fixed_negatives, seed {negative_seed}',flush=True)
        negative=fixed_negatives(positives[split],train,known,negative_seed,cdr_known)
        dedup,report=deduplicate_evaluation(identity(pd.concat([positives[split],negative],ignore_index=True)))
        frames[split]=model_frame(dedup)
        reports[split+'_combined_dedup']=report
    output.mkdir(parents=True)
    save_workspace(output/'hitph',train,frames['valid'],frames['test'],dict(level='III',backend='hla-i',pep_max_len=13,tcr_max_len=28,task='unseen',grouping='peptide',test_grouping='peptide',reference='hitph',split_seed=seed,validation_negative_seed=seed+1000,test_negative_seed=seed+2000,capability_filter=filters))
    # Audit artifact requested separately; runtime blacklist remains original save_workspace output.
    full.loc[full.pep.isin(frequencies.loc[frequencies.split.isin(['train','valid']),'pep']),['pep','ab']].drop_duplicates().to_csv(output/'hitph/positive_blacklist_train_valid.csv',index=False)
    frequencies.to_csv(output/'peptide_frequencies.csv',index=False)
    distances=pd.read_csv(assignment/'peptide_distances.csv',keep_default_na=False)
    for split in ['valid','test']:
        group=frames[split].groupby('pep').label
        positive_counts=group.apply(lambda s:int((s==1).sum()))
        negative_counts=group.apply(lambda s:int((s==0).sum()))
        mask=distances.split==split
        distances.loc[mask,'positives']=distances.loc[mask,'pep'].map(positive_counts)
        distances.loc[mask,'negatives']=distances.loc[mask,'pep'].map(negative_counts)
        columns=['pep','min_edit_distance','nearest_train_peptide']
        frames[split].merge(distances.loc[mask,columns],on='pep',validate='many_to_one').to_csv(output/f'{split}_annotated.csv',index=False)
    distances.to_csv(output/'peptide_distances.csv',index=False)
    strata=distances.groupby(['split','min_edit_distance']).agg(peptides=('pep','size'),positives=('positives','sum'),negatives=('negatives','sum'),source_positive_rows=('source_positive_rows','sum')).reset_index()
    strata['total_rows']=strata.positives+strata.negatives
    strata.to_csv(output/'distance_strata.csv',index=False)
    counts={s:dict(peptides=int(f.pep.nunique()),source_positive_rows=int(frequencies.loc[frequencies.split==s,'source_positive_rows'].sum()),positive_rows=int((f.label==1).sum()),negative_rows=int((f.label==0).sum()),labeled_rows=len(f)) for s,f in frames.items()}
    for s in ['valid','test']:
        assert counts[s]['source_positive_rows']<=len(corpus)*.1
    report={k:previous[k] for k in ['source_rows','source_peptides','seed','rng','sampling','accepted_attempt','peptides_per_eval','cap_basis','maximum_rows_per_eval','balance_tolerance','forced_train_peptides','distance_definition']}
    report.update(source=str(source.resolve()),source_sha256=digest,assignment_source=str(assignment.resolve()),assignment_sha256=hashlib.sha256((assignment/'peptide_frequencies.csv').read_bytes()).hexdigest(),split_counts=counts,**reports,training_negatives='original dynamic 1:1 per epoch',validation_negatives='original fixed_negatives at 1:1 before endpoint-equivalence deduplication; no replenishment',test_negatives='original fixed_negatives at 1:1 before endpoint-equivalence deduplication; no replenishment',runtime_blacklist='unmodified save_workspace/negative_exclusion_pairs: train positives plus all valid/test labels, with endpoint-equivalent donor expansion',positive_blacklist='separate audit CSV: all original train+valid positive pep/ab',distance_strata=strata.to_dict('records'),peptide_overlap=0,ab_handling='endpoint removal only in grouping keys; model_ab restored before CSV writing; all positive model sequences preserved from retained original rows')
    write_json(output/'benchmark_report.json',report)
    print(pd.DataFrame(counts).T.to_string())
    print(strata.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('corpora/hitph260630.csv'))
    p.add_argument('--assignment',type=Path,default=Path('workspaces/unseen_full_seed42'))
    p.add_argument('--output',type=Path,default=Path('workspaces/unseen_full_seed42_original'))
    a=p.parse_args();regenerate(a.source,a.assignment,a.output)
