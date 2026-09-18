"""Isolated tcrLM model worker. Import upstream model definitions, never scripts."""
import argparse
import json
import random
import sys
from pathlib import Path

# Host package remains available even when cwd is the external repository.
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from core_engine.trainer.workspaces import write_json
from baselines.provenance import sha256


def load_model_class(repo, mask_width):
    sys.path.insert(0,str(Path(repo).resolve()))
    from models import encoder
    if mask_width == 34:
        original = encoder.length_to_mask
        def corrected(length, max_len=34, dtype=None):
            return original(length,max_len=max_len,dtype=dtype)
        encoder.length_to_mask = corrected
    from models.tcrLM import Finetune
    return Finetune


def load_vocab(repo):
    vocab=np.load(Path(repo)/'data/dict.npy',allow_pickle=True).item()
    expected=dict(zip('LAGVSERTIDPKQNFYM HWC'.replace(' ',''),range(20)))
    expected['-']=20
    if vocab != expected:
        raise ValueError('Unexpected upstream tcrLM vocabulary; checkpoint token identities must match')
    return vocab


def encode(frame,vocab):
    """Faithful upstream 34-slot right padding, with explicit overlength rejection."""
    arrays=[]
    for name in ['peptide','tcr']:
        values=frame[name].astype(str).tolist()
        if any(not s or len(s)>34 or set(s)-set(vocab.keys()-{'-'}) for s in values):
            raise ValueError(f'Invalid {name}; sequences must use standard residues and length 1..34')
        arrays.append(np.array([[vocab[c] for c in s.ljust(34,'-')] for s in values],dtype=np.int64))
        arrays.append(np.array([len(s) for s in values],dtype=np.int64))
    # Model signature: peptide, tcr, peptide lengths, tcr lengths.
    return arrays[0],arrays[2],arrays[1],arrays[3]


def state_dict(payload):
    raw=payload.get('state_dict',payload.get('model_state_dict',payload))
    return {k.removeprefix('module.'):v for k,v in raw.items()}


def initialize(model,payload,kind):
    raw=state_dict(payload)
    if kind=='finetuned':
        model.load_state_dict(raw,strict=True)
    elif kind=='pretrain':
        raw={k.removeprefix('protflash.'):v for k,v in raw.items() if not k.startswith('fc_out.')}
        model.encoder_T.load_state_dict(raw,strict=True)
        model.encoder_P.load_state_dict(raw,strict=True)
    else:
        raise ValueError('init_kind must be pretrain or finetuned')


def predict_scores(model,arrays,batch_size,device):
    import torch
    model.eval(); result=[]
    with torch.inference_mode():
        for start in range(0,len(arrays[0]),batch_size):
            batch=[torch.as_tensor(a[start:start+batch_size],device=device) for a in arrays]
            result.extend(torch.softmax(model(*batch).float(),dim=1)[:,1].cpu().tolist())
    return np.array(result)


def validation_threshold(labels,scores):
    from sklearn.metrics import matthews_corrcoef
    cuts=np.arange(1,200)/200
    return float(max(cuts,key=lambda t:(matthews_corrcoef(labels,scores>=t),-abs(t-.5),-t)))


def execute(request):
    import torch
    from sklearn.metrics import roc_auc_score
    settings=request.get('settings',{})
    device=settings.get('device','cuda')
    batch_size=int(settings.get('batch_size',64))
    if batch_size<=0: raise ValueError('batch_size must be positive')
    payload_path=request['checkpoint'] if request['stage']=='predict' else request['init_checkpoint']
    payload=torch.load(payload_path,map_location='cpu',weights_only=True)
    saved=payload.get('adapter_metadata',{})
    mask_width=settings.get('mask_width',saved.get('mask_width',20))
    if mask_width not in (20,34): raise ValueError('mask_width must be 20 or 34')
    if saved and (saved.get('model')!='tcrlm' or saved.get('chain')!=settings.get('chain','beta') or saved.get('mask_width')!=mask_width):
        raise ValueError('Checkpoint adapter metadata disagrees with requested model/chain/mask')
    seed=int(settings.get('seed',42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    Model=load_model_class(request['repo'],mask_width)
    vocab=load_vocab(request['repo'])
    model=Model()
    if request['stage']=='predict':
        initialize(model,payload,'finetuned')
        del payload
        model=model.to(device)
        frame=pd.read_csv(request['input'],keep_default_na=False)
        if 'label' in frame: raise ValueError('Prediction worker input must not contain labels')
        arrays=encode(frame,vocab)
        scores=predict_scores(model,arrays,batch_size,device)
        threshold=request.get('threshold')
        if threshold is None: threshold=saved.get('threshold',.5)
        if not np.isfinite(threshold) or not 0<=threshold<=1: raise ValueError('Invalid threshold')
        output=Path(request['output'])
        if output.exists(): raise FileExistsError(output)
        pd.DataFrame({'row_id':frame.row_id,'score':scores,'threshold':threshold}).to_csv(output,index=False)
        write_json(output.with_suffix('.metadata.json'),dict(checkpoint=str(Path(payload_path).resolve()),checkpoint_sha256=sha256(payload_path),rows=len(frame),threshold=threshold,threshold_source='explicit' if request.get('threshold') is not None else ('checkpoint validation' if 'threshold' in saved else 'upstream default 0.5'),mask_width=mask_width,chain=settings.get('chain','beta')))
        return
    if request['stage']!='train': raise ValueError('Unknown stage')
    initialize(model,payload,request.get('init_kind','pretrain'))
    # Release CPU init payload before moving the large model to its runtime device.
    del payload
    for encoder in [model.encoder_P,model.encoder_T]:
        for parameter in encoder.parameters(): parameter.requires_grad=False
    model=model.to(device)
    train=pd.read_csv(request['train'],keep_default_na=False)
    valid=pd.read_csv(request['valid'],keep_default_na=False)
    if set(train.label)!={0,1} or set(valid.label)!={0,1}: raise ValueError('Train and validation need both labels')
    train_arrays,valid_arrays=encode(train,vocab),encode(valid,vocab)
    y=train.label.to_numpy(dtype=np.int64)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=float(settings.get('learning_rate',1e-3)))
    criterion=torch.nn.CrossEntropyLoss()
    output=Path(request['output'])
    output.mkdir(parents=True,exist_ok=False)
    best=-float('inf'); stale=0; history=[]
    epochs=int(settings.get('epochs',35)); patience=int(settings.get('patience',5))
    for epoch in range(1,epochs+1):
        model.train()
        # Frozen upstream encoders have zero configured attention dropout.
        order=np.random.RandomState(seed+epoch).permutation(len(train))
        loss_sum=0.
        for start in range(0,len(order),batch_size):
            idx=order[start:start+batch_size]
            batch=[torch.as_tensor(a[idx],device=device) for a in train_arrays]
            target=torch.as_tensor(y[idx],device=device)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(*batch).float(),target)
            loss.backward(); optimizer.step()
            loss_sum+=float(loss.detach())*len(idx)
        scores=predict_scores(model,valid_arrays,batch_size,device)
        auc=float(roc_auc_score(valid.label,scores))
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(train),validation_auroc=auc))
        print(json.dumps(history[-1]),flush=True)
        if auc>best:
            best=auc; stale=0
            threshold=validation_threshold(valid.label.to_numpy(),scores)
            metadata=dict(model='tcrlm',chain=settings.get('chain','beta'),mask_width=mask_width,threshold=threshold,best_epoch=epoch,selection_metric='validation AUROC',validation_auroc=auc,threshold_selection='maximum validation MCC; >= classification',seed=seed,init_checkpoint=str(Path(payload_path).resolve()),init_kind=request.get('init_kind','pretrain'))
            torch.save({'state_dict':model.state_dict(),'adapter_metadata':metadata},output/'best.pt')
            write_json(output/'selected_threshold.json',metadata)
        else: stale+=1
        write_json(output/'history.json',history)
        if stale>=patience: break
    write_json(output/'training_result.json',dict(best_checkpoint=str(output/'best.pt'),best_validation_auroc=best,epochs_run=len(history),settings=settings,init_checkpoint_sha256=sha256(payload_path)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True)
    args=p.parse_args()
    execute(json.loads(args.request.read_text()))
