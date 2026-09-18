"""ATM-TCR worker: upstream Net, fixed workspaces, dynamic screened negatives."""
import argparse
import ast
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from core_engine.trainer.fixed_splits import cdr3_group
from core_engine.trainer.workspaces import write_json

ARCH_DEFAULTS = dict(max_len_pep=22, max_len_tcr=20, heads=5, lin_size=1024,
                     drop_rate=.25, padding='mid', blosum=None)


def upstream_preprocessing(repo):
    """Load only upstream pure preprocessing; avoid legacy torchtext imports.

    Compile the original pad method, tokenizer, vocabulary and load_embedding
    verbatim from its AST. The rest of data_loader.py is not executed.
    """
    tree = ast.parse((Path(repo)/'data_loader.py').read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='AMINO_MAP' for t in node.targets):
            nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in ('tokenizer','load_embedding'):
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name=='Field_modified':
            nodes.extend(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='pad')
    import re
    scope = dict(np=np,re=re)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(Path(repo)/'data_loader.py'),'exec'),scope)
    if not all(k in scope for k in ('AMINO_MAP','pad','tokenizer','load_embedding')):
        raise ValueError('Unsupported upstream preprocessing layout')
    return scope


def encode(frame, repo, arch):
    """Use the exact upstream token IDs and padding with overlength rejection."""
    scope=upstream_preprocessing(repo)
    arrays=[]
    for column,width in [('peptide',arch['max_len_pep']),('tcr',arch['max_len_tcr'])]:
        values=frame[column].astype(str).tolist()
        if any(not s or len(s)>width or set(s)-set('ABCDEFGHIJKLMNOPQRSTUVWXYZ*') for s in values):
            raise ValueError(f'Unsupported {column} sequence; no silent truncation')
        field=SimpleNamespace(sequential=True,fix_length=width,init_token=None,eos_token=None,
                              pad_token='<pad>',pad_type=arch['padding'],truncate_first=False,include_lengths=False)
        padded=scope['pad'](field,[scope['tokenizer'](s) for s in values])
        arrays.append(np.asarray([[scope['AMINO_MAP'][c] for c in row] for row in padded],dtype=np.int64))
    return arrays


def build_model(repo, arch):
    spec=importlib.util.spec_from_file_location('upstream_atm_attention',Path(repo)/'attention.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # Upstream None loads this matrix for dimensions, but uses random embeddings.
    matrix=upstream_preprocessing(repo)['load_embedding'](str(Path(repo)/'data/blosum/BLOSUM45'))
    if arch['blosum'] is not None:
        matrix=upstream_preprocessing(repo)['load_embedding'](arch['blosum'])
    return module.Net(matrix,SimpleNamespace(**arch))


def load_payload(path):
    import torch
    # PyTorch 1.10 has no weights_only argument. Only trusted upstream/local files.
    return torch.load(path,map_location='cpu')


def predict_scores(model, arrays, batch_size, device):
    import torch
    model.eval(); scores=[]
    with torch.no_grad():
        for start in range(0,len(arrays[0]),batch_size):
            batch=[torch.as_tensor(a[start:start+batch_size],device=device) for a in arrays]
            scores.extend(model(*batch).flatten().cpu().tolist())
    return np.asarray(scores)


class AdmissibleDonors:
    """One shared donor array plus sparse blocked indices per peptide."""
    def __init__(self, donors, forbidden):
        self.donors=donors
        self.forbidden=forbidden
        if len(donors)==len(forbidden): raise ValueError('No admissible negative donors')

    def draw(self, count, rng):
        indices=rng.randint(len(self.donors),size=count)
        rejected=np.asarray([i in self.forbidden for i in indices])
        while rejected.any():
            indices[rejected]=rng.randint(len(self.donors),size=int(rejected.sum()))
            rejected=np.asarray([i in self.forbidden for i in indices])
        return self.donors[indices]

    def tolist(self):
        return [t for i,t in enumerate(self.donors) if i not in self.forbidden]


def screened_candidates(train, valid, exclusion):
    """Project known/held-out pairs to beta identity without copying whole pools."""
    import re
    def group(tcr):
        return cdr3_group(re.sub(r'[^ARNDCQEGHILKMFPSTWYVBZX]', '*', tcr))
    blocked={}
    for p,g in pd.concat([train,valid,exclusion],ignore_index=True).groupby('peptide'):
        blocked[p]=set(g.tcr.map(group))
    donors=np.asarray(sorted(set(train.tcr)))
    indices_by_group={}
    for i,tcr in enumerate(donors): indices_by_group.setdefault(group(tcr),[]).append(i)
    result={}
    for p in sorted(set(train.peptide)):
        forbidden={i for g in blocked.get(p,set()) for i in indices_by_group.get(g,())}
        if len(forbidden)==len(donors): raise ValueError(f'No admissible negative for {p}')
        result[p]=AdmissibleDonors(donors,forbidden)
    return result


def dynamic_epoch(train, candidates, seed):
    rng=np.random.RandomState(seed)
    negative=train.copy()
    for p,g in train.groupby('peptide',sort=True):
        negative.loc[g.index,'tcr']=candidates[p].draw(len(g),rng)
    negative['label']=0
    return pd.concat([train,negative],ignore_index=True)


def validation_threshold(y_true, scores):
    """Host protocol: 199 interior thresholds, maximum validation MCC.

    Use >= consistently with the exported predicted labels.
    """
    from sklearn.metrics import matthews_corrcoef
    cuts=np.arange(1,200)/200.
    values=[matthews_corrcoef(y_true,scores>=cut) for cut in cuts]
    best=int(np.argmax(values))
    return float(cuts[best]),float(values[best])


def execute(request):
    import torch
    from sklearn.metrics import roc_auc_score
    settings=request.get('settings',{})
    torch.set_num_threads(int(settings.get('num_threads',4)))
    device=settings.get('device','cuda'); bs=int(settings.get('batch_size',32))
    if bs<2: raise ValueError('ATM-TCR BatchNorm training needs batch_size >= 2')
    seed=int(settings.get('seed',42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    payload=None; saved={}
    source=request.get('checkpoint') if request['stage']=='predict' else request.get('init_checkpoint')
    if source:
        payload=load_payload(source); saved=payload.get('adapter_metadata',{})
    arch=dict(ARCH_DEFAULTS)
    if saved: arch.update(saved['architecture'])
    for k in arch:
        if k in settings:
            if saved and settings[k]!=arch[k]: raise ValueError(f'Checkpoint architecture mismatch: {k}')
            arch[k]=settings[k]
    if saved and (saved.get('model')!='atm-tcr' or saved.get('chain')!=settings.get('chain','beta')):
        raise ValueError('Checkpoint model/chain mismatch')
    model=build_model(request['repo'],arch).to(device)
    if payload is not None:
        raw=payload.get('state_dict',payload)
        model.load_state_dict({(k[7:] if k.startswith('module.') else k):v for k,v in raw.items()},strict=True)
        del payload
    if request['stage']=='predict':
        frame=pd.read_csv(request['input'],keep_default_na=False)
        if 'label' in frame: raise ValueError('Prediction request must be label-free')
        scores=predict_scores(model,encode(frame,request['repo'],arch),bs,device)
        threshold=request.get('threshold')
        if threshold is None: threshold=saved.get('threshold',.5)
        if not np.isfinite(threshold) or not 0<=threshold<=1: raise ValueError('Invalid threshold')
        output=Path(request['output'])
        if output.exists(): raise FileExistsError(output)
        pd.DataFrame(dict(row_id=frame.row_id,score=scores,threshold=threshold)).to_csv(output,index=False)
        return
    if request['stage']!='train': raise ValueError('Unknown worker stage')
    train=pd.read_csv(request['train'],keep_default_na=False)
    valid=pd.read_csv(request['valid'],keep_default_na=False)
    if set(valid.label)!={0,1} or set(train.label) not in ({1},{0,1}): raise ValueError('Invalid train/validation labels')
    candidates=None
    if set(train.label)=={1}:
        exclusion=pd.read_csv(request['negative_exclusion'],keep_default_na=False)
        candidates=screened_candidates(train,valid,exclusion)
    valid_arrays=encode(valid,request['repo'],arch)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(settings.get('learning_rate',.001)))
    criterion=torch.nn.BCELoss()
    output=Path(request['output']); output.mkdir(parents=True,exist_ok=False)
    best=-float('inf'); stale=0; history=[]; best_epoch=0
    print(json.dumps(dict(event='training_started',device=device,torch=torch.__version__,architecture=arch,positive_source_rows=len(train),validation_rows=len(valid),dynamic_negatives=candidates is not None)),flush=True)
    for epoch in range(1,int(settings.get('epochs',200))+1):
        epoch_train=dynamic_epoch(train,candidates,int(settings.get('negative_seed',seed))+epoch) if candidates is not None else train
        arrays=encode(epoch_train,request['repo'],arch)
        labels=epoch_train.label.to_numpy(dtype=np.float32)
        order=np.random.RandomState(seed+epoch).permutation(len(epoch_train))
        # Merge a final singleton into the preceding batch for BatchNorm.
        batches=[order[i:i+bs] for i in range(0,len(order),bs)]
        if len(batches[-1])==1 and len(batches)>1:
            batches[-2]=np.concatenate([batches[-2],batches[-1]]); batches.pop()
        model.train(); loss_sum=0.
        for step,idx in enumerate(batches,1):
            batch=[torch.as_tensor(a[idx],device=device) for a in arrays]
            target=torch.as_tensor(labels[idx],device=device).unsqueeze(1)
            optimizer.zero_grad(); loss=criterion(model(*batch),target)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
            loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(idx)
            if step==1 or step%500==0:
                print(json.dumps(dict(event='train_batch',epoch=epoch,step=step,total_steps=len(batches),loss=float(loss.detach()))),flush=True)
        scores=predict_scores(model,valid_arrays,bs,device)
        auc=float(roc_auc_score(valid.label,scores))
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(epoch_train),validation_auroc=auc))
        print(json.dumps(history[-1]),flush=True)
        if auc>best:
            best=auc;stale=0;best_epoch=epoch
            threshold,mcc=validation_threshold(valid.label.to_numpy(),scores)
            metadata=dict(model='atm-tcr',chain='beta',architecture=arch,threshold=float(threshold),validation_mcc=float(mcc),best_epoch=epoch,validation_auroc=auc,seed=seed,negative_seed=settings.get('negative_seed',seed),selection_metric='validation AUROC',threshold_selection='maximum validation MCC',training_negatives='dynamic 1:1 screened beta mismatches' if candidates is not None else 'supplied fixed labels')
            torch.save(dict(state_dict=model.state_dict(),adapter_metadata=metadata),output/'best.pt')
        else: stale+=1
        write_json(Path(request.get('log_dir',str(output)))/'history.json',history)
        if stale>=int(settings.get('patience',4)): break


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    execute(json.loads(parser.parse_args().request.read_text()))
