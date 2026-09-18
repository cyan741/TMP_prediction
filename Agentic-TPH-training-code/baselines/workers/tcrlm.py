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
    return {(k[7:] if k.startswith('module.') else k):v for k,v in raw.items()}


def initialize(model,payload,kind):
    raw=state_dict(payload)
    if kind=='finetuned':
        model.load_state_dict(raw,strict=True)
    elif kind=='pretrain':
        raw={(k[10:] if k.startswith('protflash.') else k):v for k,v in raw.items() if not k.startswith('fc_out.')}
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


def frozen_features(model, train, valid, vocab, batch_size, device, standardize=False):
    """Encode each distinct sequence once; upstream fine-tuning freezes encoders."""
    import torch
    tables=[]; indices=[]; means=[]; scales=[]
    model.eval()
    for column,encoder in [('peptide',model.encoder_P),('tcr',model.encoder_T)]:
        values=list(dict.fromkeys(train[column].tolist()+valid[column].tolist()))
        lookup={value:i for i,value in enumerate(values)}
        tokens=np.array([[vocab[c] for c in value.ljust(34,'-')] for value in values],dtype=np.int64)
        lengths=np.array([len(value) for value in values],dtype=np.int64)
        table=torch.empty((len(values),34*512),dtype=torch.float32)
        with torch.no_grad():
            for start in range(0,len(values),batch_size):
                x=torch.as_tensor(tokens[start:start+batch_size],device=device)
                n=torch.as_tensor(lengths[start:start+batch_size],device=device)
                table[start:start+len(x)]=encoder(x,n).reshape(len(x),-1).cpu()
                if start % (batch_size*100)==0:
                    print(json.dumps(dict(encoding=column,completed=start+len(x),total=len(values))),flush=True)
        train_ids=np.array([lookup[v] for v in train[column]],dtype=np.int64)
        if standardize:
            # Weighted by TRAIN occurrences; validation-only sequences have zero weight.
            weights=torch.as_tensor(np.bincount(train_ids,minlength=len(values)),dtype=torch.float64)
            mean=torch.empty(table.shape[1]);scale=torch.empty_like(mean)
            for start in range(0,table.shape[1],512):
                block=table[:,start:start+512].double()
                mu=block.T.mv(weights)/len(train)
                variance=(block-mu).square().T.mv(weights)/len(train)
                sigma=variance.sqrt()
                sigma=torch.where(sigma>1e-3,sigma,torch.ones_like(sigma))
                mean[start:start+512]=mu.float();scale[start:start+512]=sigma.float()
            table.sub_(mean).div_(scale)
            means.append(mean);scales.append(scale)
        tables.append(table)
        indices.append([train_ids,np.array([lookup[v] for v in valid[column]],dtype=np.int64)])
    def features(rows, split):
        return torch.cat([table[index[split][rows]] for table,index in zip(tables,indices)],dim=1).to(device)
    if standardize:
        features.mean=torch.cat(means).to(device)
        features.scale=torch.cat(scales).to(device)
    return features


def export_state(model, features):
    """Fold training standardization into the original linear head for inference."""
    state=model.state_dict()
    if hasattr(features,'mean'):
        weight=model.classifier.weight.detach()/features.scale
        state['classifier.weight']=weight
        state['classifier.bias']=model.classifier.bias.detach()-weight @ features.mean
    return state


def classifier_scores(model, features, size, batch_size):
    import torch
    result=[]
    with torch.no_grad():
        for start in range(0,size,batch_size):
            rows=np.arange(start,min(size,start+batch_size))
            result.extend(torch.softmax(model.classifier(features(rows,1)),dim=1)[:,1].cpu().tolist())
    return np.array(result)


def validation_threshold(labels,scores):
    from sklearn.metrics import matthews_corrcoef
    cuts=np.arange(1,200)/200
    def quality(cut):
        predicted=scores>=cut
        mcc=0. if predicted.all() or not predicted.any() else matthews_corrcoef(labels,predicted)
        return mcc,-abs(cut-.5),-cut
    return float(max(cuts,key=quality))


def execute(request):
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    from sklearn.metrics import roc_auc_score
    settings=request.get('settings',{})
    device=settings.get('device','cuda')
    batch_size=int(settings.get('batch_size',64))
    if batch_size<=0: raise ValueError('batch_size must be positive')
    payload_path=request['checkpoint'] if request['stage']=='predict' else request['init_checkpoint']
    payload=torch.load(payload_path,map_location='cpu')
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
        print(json.dumps(dict(rows=len(frame),checkpoint=str(payload_path),threshold=float(threshold))),flush=True)
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
    encode(train,vocab); encode(valid,vocab)
    output=Path(request['output'])
    output.mkdir(parents=True,exist_ok=True)
    if (output/'best.pt').exists(): raise FileExistsError(output/'best.pt')
    standardize=bool(settings.get('standardize_features',True))
    features=frozen_features(model,train,valid,vocab,batch_size,device,standardize)
    if standardize:
        # No pretrained classifier exists; begin at balanced, unconfident logits.
        if request.get('init_kind','pretrain')=='pretrain':
            torch.nn.init.zeros_(model.classifier.weight);torch.nn.init.zeros_(model.classifier.bias)
        else:
            with torch.no_grad():
                model.classifier.bias.add_(model.classifier.weight @ features.mean)
                model.classifier.weight.mul_(features.scale)
    train_batch_size=int(settings.get('train_batch_size',2048))
    if train_batch_size<=0: raise ValueError('train_batch_size must be positive')
    y=train.label.to_numpy(dtype=np.int64)
    print(json.dumps(dict(training_rows=len(train),validation_rows=len(valid),train_batch_size=train_batch_size,encoder_batch_size=batch_size,learning_rate=settings.get('learning_rate',1e-4),standardize_features=standardize,initialization='zero linear head' if standardize and request.get('init_kind','pretrain')=='pretrain' else 'checkpoint/random head',allow_tf32=False)),flush=True)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=float(settings.get('learning_rate',1e-4)))
    criterion=torch.nn.CrossEntropyLoss()
    best=-float('inf'); stale=0; history=[]
    epochs=int(settings.get('epochs',35)); patience=int(settings.get('patience',5))
    for epoch in range(1,epochs+1):
        model.train()
        # Frozen upstream encoders have zero configured attention dropout.
        order=np.random.RandomState(seed+epoch).permutation(len(train))
        loss_sum=0.
        for start in range(0,len(order),train_batch_size):
            idx=order[start:start+train_batch_size]
            batch=features(idx,0)
            target=torch.as_tensor(y[idx],device=device)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model.classifier(batch).float(),target)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite training loss")
            loss.backward(); optimizer.step()
            loss_sum+=float(loss.detach())*len(idx)
        scores=classifier_scores(model,features,len(valid),train_batch_size)
        validation_loss=0.
        with torch.no_grad():
            for start in range(0,len(valid),train_batch_size):
                idx=np.arange(start,min(len(valid),start+train_batch_size))
                target=torch.as_tensor(valid.label.to_numpy(dtype=np.int64)[idx],device=device)
                validation_loss+=float(criterion(model.classifier(features(idx,1)),target))*len(idx)
        auc=float(roc_auc_score(valid.label,scores))
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(train),validation_loss=validation_loss/len(valid),validation_auroc=auc))
        print(json.dumps(history[-1]),flush=True)
        if auc>best:
            best=auc; stale=0
            threshold=validation_threshold(valid.label.to_numpy(),scores)
            metadata=dict(model='tcrlm',chain=settings.get('chain','beta'),mask_width=mask_width,threshold=threshold,best_epoch=epoch,selection_metric='validation AUROC',validation_auroc=auc,threshold_selection='maximum validation MCC; >= classification',seed=seed,init_checkpoint=str(Path(payload_path).resolve()),init_kind=request.get('init_kind','pretrain'),learning_rate=settings.get('learning_rate',1e-4),train_batch_size=train_batch_size,training_feature_standardization=standardize,allow_tf32=False)
            torch.save({'state_dict':export_state(model,features),'adapter_metadata':metadata},output/'best.pt')
        else: stale+=1
        write_json(Path(request.get('log_dir',str(output)))/'history.json',history)
        if stale>=patience: break


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True)
    args=p.parse_args()
    execute(json.loads(args.request.read_text()))
