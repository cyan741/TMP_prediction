#!/usr/bin/env python3
"""Plan, prepare or explicitly run an external baseline; planning runs no model."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from baselines.adapters import get_adapter
from baselines.provenance import sha256
from baselines.data import read_csv, labels, projection_conflicts
from inference import read_inputs, write_predictions
from core_engine.trainer.workspaces import write_json


PATH_KEYS = ('repo','python','input','workspace','train','valid','negative_exclusion','checkpoint','init_checkpoint','output')


def load_config(path):
    path = path.resolve()
    cfg = json.loads(path.read_text())
    for key in PATH_KEYS:
        if cfg.get(key):
            p = Path(cfg[key]).expanduser()
            cfg[key] = str((path.parent/p).resolve() if not p.is_absolute() else p.resolve())
    cfg.setdefault('python',sys.executable)
    cfg.setdefault('workflow','predict')
    cfg.setdefault('settings',{})
    if cfg['workflow'] not in ('predict','train-predict'):
        raise ValueError('workflow must be predict or train-predict')
    for key in ['model','repo','input','output']:
        if not cfg.get(key): raise ValueError(f'Missing configuration: {key}')
    if cfg.get('workspace'):
        w = Path(cfg['workspace'])
        for key,file in [('train','train_data_fold0.csv'),('valid','valid_data_fold0.csv'),('negative_exclusion','negative_blacklist_corpus.csv')]:
            cfg.setdefault(key,str(w/file))
        metadata = w/'workspace_report.json'
        if metadata.exists() and json.loads(metadata.read_text()).get('level') == 'IV':
            cfg['settings'].setdefault('paired_column','cdr3_ab')
    return cfg


def plan(cfg):
    adapter = get_adapter(cfg['model'])
    settings = cfg['settings']
    for k in ['batch_size','epochs','patience']:
        if k in settings and (not isinstance(settings[k],int) or settings[k] <= 0):
            raise ValueError(f'{k} must be a positive integer')
    if float(settings.get('learning_rate',1e-3)) <= 0:
        raise ValueError('learning_rate must be positive')
    if cfg.get('threshold') is not None and (not np.isfinite(cfg['threshold']) or not 0 <= cfg['threshold'] <= 1):
        raise ValueError('threshold must be in [0,1]')
    repo = Path(cfg['repo'])
    model_info = adapter.validate(repo,settings)
    original = read_inputs(Path(cfg['input']),[])
    converted = adapter.convert(original,settings)
    info = dict(workflow=cfg['workflow'],model=adapter.name,repo=str(repo),python=cfg['python'],output=cfg['output'],input_rows=len(original),input_peptides=int(converted.peptide.nunique()),dependencies=list(adapter.dependencies),predict_uses_labels=False,paths_relative_to='configuration directory',missing_assets=[])
    info.update(model_info)
    if 'label' in original and original.label.isin(['0','1']).all():
        info['test_projection_conflicts']=projection_conflicts(converted,original.label.astype(int))
    stages=[]
    if cfg['workflow']=='train-predict':
        for k in ['train','valid','init_checkpoint']:
            if not cfg.get(k): raise ValueError(f'train-predict requires {k}')
        train,valid=read_csv(cfg['train']),read_csv(cfg['valid'])
        train,ct,excluded_train_rows=adapter.training_input(train,settings)
        cv=adapter.convert(valid,settings)
        labels(train); vy=labels(valid)
        if set(vy)!={0,1}: raise ValueError('Validation needs both labels')
        sets=[set(f.peptide) for f in [ct,cv,converted]]
        if any(sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)]):
            raise ValueError('Unseen-peptide train/validation/test overlap')
        if set(labels(train))=={1} and not cfg.get('negative_exclusion'):
            raise ValueError('Positive-only train requires negative_exclusion')
        if cfg.get('negative_exclusion'):
            exclusion=read_csv(cfg['negative_exclusion'])
            adapter.convert_exclusions(exclusion,settings)
        info.update(training_excluded_csv_rows=excluded_train_rows,training_kept_rows=len(train),train_source_rows=len(train)+len(excluded_train_rows),valid_rows=len(valid),train_peptides=len(sets[0]),valid_peptides=len(sets[1]),validation_projection_conflicts=projection_conflicts(cv,vy),training_negatives='fixed 1:1 if train is positive-only; exclusion projected to selected chain',selection_metric='validation AUROC',threshold_selection='validation MCC; frozen before test')
        stages.append('train')
    else:
        if not cfg.get('checkpoint'): raise ValueError('predict requires a finetuned checkpoint')
        info['threshold_selection']='checkpoint metadata, explicit threshold, or upstream 0.5 for raw state_dict'
    for k in ['checkpoint','init_checkpoint']:
        if cfg.get(k) and not Path(cfg[k]).is_file(): info['missing_assets'].append(cfg[k])
    info['python_exists']=Path(cfg['python']).is_file()
    stages.append('predict')
    info['stages']=[dict(stage=s,argv=[cfg['python'],str(adapter.worker.resolve()),'--request',str(Path(cfg['output'])/f'{s}_request.json')],cwd=str(repo)) for s in stages]
    info['input_sha256']=sha256(Path(cfg['input']))
    try:
        info['repo_commit']=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    except (subprocess.CalledProcessError,FileNotFoundError):
        info['repo_commit']=None
    return info


def prepare(cfg):
    info=plan(cfg)
    output=Path(cfg['output'])
    if output.exists(): raise FileExistsError(output)
    adapter=get_adapter(cfg['model']); settings=cfg['settings']
    source=read_inputs(Path(cfg['input']),[])
    converted=adapter.convert(source,settings)
    output.mkdir(parents=True)
    (output/'data').mkdir()
    # Outcome labels never enter the prediction request or worker input.
    converted.to_csv(output/'data/input.csv',index=False)
    source.to_csv(output/'data/original_input.csv',index=False)
    common=dict(model=adapter.name,repo=cfg['repo'],settings=settings)
    checkpoint=cfg.get('checkpoint')
    if cfg['workflow']=='train-predict':
        train,valid=read_csv(cfg['train']),read_csv(cfg['valid'])
        train,ct,excluded_train_rows=adapter.training_input(train,settings)
        cv=adapter.convert(valid,settings)
        exclusion=adapter.convert_exclusions(read_csv(cfg['negative_exclusion']),settings) if cfg.get('negative_exclusion') else None
        train_data,valid_data=adapter.prepare_training(train,valid,ct,cv,exclusion,int(settings.get('negative_seed',settings.get('seed',42))))
        if excluded_train_rows:
            read_csv(cfg['train']).iloc[np.array(excluded_train_rows)-2].to_csv(output/'data/excluded_train.csv',index=False)
        train_data.to_csv(output/'data/train.csv',index=False)
        valid_data.to_csv(output/'data/valid.csv',index=False)
        checkpoint=str(output/'model/best.pt')
        write_json(output/'train_request.json',dict(**common,stage='train',train=str(output/'data/train.csv'),valid=str(output/'data/valid.csv'),init_checkpoint=cfg['init_checkpoint'],init_kind=cfg.get('init_kind','pretrain'),output=str(output/'model')))
        info['training_prepared_rows']=len(train_data)
        info['train_sha256']=sha256(Path(cfg['train']))
        info['valid_sha256']=sha256(Path(cfg['valid']))
    write_json(output/'predict_request.json',dict(**common,stage='predict',input=str(output/'data/input.csv'),checkpoint=checkpoint,threshold=cfg.get('threshold'),output=str(output/'worker_predictions.csv')))
    write_json(output/'resolved_config.json',cfg)
    write_json(output/'plan.json',info)
    return info


def run(cfg):
    info=plan(cfg)
    if info['missing_assets']: raise FileNotFoundError(f"Missing local weights: {info['missing_assets']}")
    if not info['python_exists']: raise FileNotFoundError(cfg['python'])
    dependency_probe = subprocess.run([cfg['python'],'-c','import importlib.util,json; print(json.dumps([m for m in '+repr(info['dependencies'])+' if importlib.util.find_spec(m) is None]))'],capture_output=True,text=True,check=True)
    missing_dependencies=json.loads(dependency_probe.stdout)
    if missing_dependencies:
        raise RuntimeError(f'External Python missing dependencies: {missing_dependencies}')
    # Only explicit `run` may invoke an external model process.
    output=Path(cfg['output'])
    if output.exists():
        if (output/'predictions.csv').exists():
            raise FileExistsError(output/'predictions.csv')
        if json.loads((output/'resolved_config.json').read_text()) != cfg:
            raise ValueError('Prepared configuration differs; use a new output directory')
        prepared=json.loads((output/'plan.json').read_text())
        if prepared['input_sha256'] != info['input_sha256'] or prepared['repo_commit'] != info['repo_commit']:
            raise ValueError('Input or repo changed since prepare')
        for key in ['train','valid']:
            if cfg.get(key) and prepared.get(key+'_sha256') != sha256(cfg[key]):
                raise ValueError(f'{key} changed since prepare')
    else:
        info=prepare(cfg)
    for stage in info['stages']:
        with (output/f"{stage['stage']}.log").open('w') as log:
            subprocess.run(stage['argv'],cwd=stage['cwd'],stdout=log,stderr=subprocess.STDOUT,check=True)
    predictions=read_csv(output/'worker_predictions.csv')
    ids=predictions.row_id.astype(int)
    original=read_csv(output/'data/original_input.csv')
    if ids.duplicated().any() or set(ids)!=set(range(len(original))):
        raise ValueError('Worker predictions must cover each input row exactly once')
    predictions=predictions.assign(row_id=ids).sort_values('row_id')
    scores=predictions.score.astype(float).to_numpy()
    cuts=predictions.threshold.astype(float).to_numpy()
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any(): raise ValueError('Invalid prediction scores')
    if not np.isfinite(cuts).all() or ((cuts<0)|(cuts>1)).any() or len(np.unique(cuts))!=1: raise ValueError('Expected one frozen probability threshold')
    result=write_predictions(original,scores.tolist(),float(cuts[0]),output/'predictions.csv')
    write_json(output/'result.json',dict(**result,model=cfg['model'],workflow=cfg['workflow'],input_sha256=info['input_sha256'],repo_commit=info['repo_commit']))
    return result


def evaluate(predictions, output, benchmark=None, peptide_column='pep'):
    """Evaluate labeled predictions; optionally verify the full fixed benchmark."""
    if output.exists(): raise FileExistsError(output)
    if benchmark is not None:
        from report_distance_metrics import report
        report(predictions,benchmark,output)
        return json.loads(output.read_text())
    from report_distance_metrics import metrics
    frame=read_csv(predictions)
    if peptide_column not in frame or 'score' not in frame:
        raise ValueError('Evaluation needs peptide identity, label and score columns')
    y=labels(frame)
    scores=frame.score.astype(float)
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError('Evaluation scores must be finite probabilities')
    frame=frame.copy()
    frame['pep']=frame[peptide_column].str.strip().str.upper()
    frame['label']=y
    frame['score']=scores
    result=dict(overall=metrics(frame),distance_reference=None,threshold_source='prediction file; must be selected independently of evaluation labels')
    write_json(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for command in ['plan','prepare','run']:
        q=sub.add_parser(command)
        q.add_argument('--config',type=Path,required=True)
    q=sub.add_parser('evaluate')
    q.add_argument('--predictions',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True)
    q.add_argument('--benchmark',type=Path)
    q.add_argument('--peptide-column',default='pep')
    a=p.parse_args()
    if a.command=='evaluate':
        result=evaluate(a.predictions,a.output,a.benchmark,a.peptide_column)
    else:
        cfg=load_config(a.config)
        result={'plan':plan,'prepare':prepare,'run':run}[a.command](cfg)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
