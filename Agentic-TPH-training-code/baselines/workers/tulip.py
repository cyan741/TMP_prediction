"""TULIP: original conditional generation, positive-only CUDA training."""
import argparse
import json
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from core_engine.trainer.workspaces import write_json


def load_model(repo):
    import torch
    from transformers import AutoTokenizer, BertConfig, BertModel
    from tokenizers.processors import TemplateProcessing
    sys.path.insert(0,str(Path(repo).resolve()))
    from src.multiTrans import Tulip, TulipPetal, BertLastPooler
    repo=Path(repo)
    tokenizer=AutoTokenizer.from_pretrained(str(repo/'aatok'),local_files_only=True)
    tokenizer.add_special_tokens(dict(pad_token='<PAD>',sep_token='<MIS>',cls_token='<CLS>',eos_token='<EOS>',mask_token='<MASK>'))
    tokenizer._tokenizer.post_processor=TemplateProcessing(single='<CLS> $A <EOS>',pair='<CLS> $A <MIS> $B:1 <EOS>:1',special_tokens=[('<EOS>',2),('<CLS>',3),('<MIS>',4)])
    mhctok=AutoTokenizer.from_pretrained(str(repo/'mhctok'),local_files_only=True)
    config=json.loads((repo/'model_weights/config.json').read_text())
    enc=BertConfig.from_dict(config['encoder']);dec=BertConfig.from_dict(config['decoder'])
    encoders=[BertModel(enc) for _ in range(3)]
    decoders=[TulipPetal(dec) for _ in range(3)]
    for decoder in decoders:decoder.pooler=BertLastPooler(dec)
    model=Tulip(encoderA=encoders[0],encoderB=encoders[1],encoderE=encoders[2],decoderA=decoders[0],decoderB=decoders[1],decoderE=decoders[2])
    return model,tokenizer,mhctok


def encode(frame,tokenizer,mhctok,device):
    arrays=[]
    for name in ['alpha','beta','peptide']:
        values=tokenizer(frame[name].tolist(),padding=True,return_tensors='pt')
        if values['input_ids'].shape[1]>50:raise ValueError('TULIP token length exceeds model positions')
        if name=='peptide' and (values['input_ids']==0).any():raise ValueError('Unknown peptide token')
        arrays.append({k:v.to(device) for k,v in values.items()})
    canonical={k.upper():k for k in mhctok.get_vocab()}
    mhc=[canonical.get(value.upper(),value) for value in frame.mhc]
    arrays.append({k:v.to(device) for k,v in mhctok(mhc,return_tensors='pt').items()})
    return arrays


def batch(arrays,rows):
    return [{k:v[rows] for k,v in item.items()} for item in arrays]


def forward(model,items,peptide_only=False):
    return model(input_ids=tuple(x['input_ids'] for x in items[:3]),attention_mask=tuple(x['attention_mask'] for x in items[:3]),mhc=items[3],labels=None,togenerate='E' if peptide_only else None,return_dict=True)


def masked_tokens(ids,special_ids,vocab_size,mask_id):
    import torch
    special=torch.zeros_like(ids,dtype=torch.bool)
    for token in special_ids:special|=ids==token
    masked=(torch.rand(ids.shape,device=ids.device)<.15)&~special
    labels=ids.masked_fill(~masked,-100)
    corrupted=ids.clone()
    replace=(torch.rand(ids.shape,device=ids.device)<.8)&masked
    corrupted[replace]=mask_id
    randomize=(torch.rand(ids.shape,device=ids.device)<.5)&masked&~replace
    words=torch.randint(vocab_size,ids.shape,device=ids.device)
    corrupted[randomize]=words[randomize]
    return corrupted,labels


def generative_loss(model,items,tokenizer):
    import torch
    from torch.nn import functional as F
    output=forward(model,items)
    total=output.decoder_outputsE.lm_logits.sum()*0.;count=0
    for name,item in zip(['A','B','E'],items[:3]):
        ids=item['input_ids'];observed=ids[:,1]!=4
        if not observed.any():continue
        target=ids[observed,1:]
        logits=getattr(output,'decoder_outputs'+name).lm_logits[observed,:-1].float()
        total=total+F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),ignore_index=tokenizer.pad_token_id,reduction='sum')
        count=count+(target!=tokenizer.pad_token_id).sum()
        corrupted,labels=masked_tokens(ids[observed],tokenizer.all_special_ids,len(tokenizer),tokenizer.mask_token_id)
        encoded=getattr(model,'encoder'+name)(corrupted,attention_mask=item['attention_mask'][observed],return_dict=True).last_hidden_state
        mlm=getattr(model,'MLMHead'+name)(encoded).float()
        total=total+F.cross_entropy(mlm.reshape(-1,mlm.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='sum')
        count=count+(labels!=-100).sum()
    # Original summed LM + MLM objective expressed per predicted token.
    return total/count,count


def log_scores(model,arrays,size,batch_size):
    import torch
    from torch.nn import functional as F
    model.eval();scores=[]
    with torch.no_grad():
        for start in range(0,size,batch_size):
            items=batch(arrays,slice(start,start+batch_size))
            logits=forward(model,items,peptide_only=True).logits.float()
            ids=items[2]['input_ids'][:,1:]
            losses=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),ids.reshape(-1),ignore_index=1,reduction='none').reshape(len(ids),-1)
            scores.extend((-losses.sum(dim=1)).double().cpu().tolist())
    return np.array(scores,dtype=np.float64)


def choose_threshold(y,scores):
    from sklearn.metrics import matthews_corrcoef
    cuts=np.unique(np.quantile(scores,np.linspace(0,1,201)))
    def quality(cut):
        pred=scores>=cut
        mcc=0. if pred.all() or not pred.any() else matthews_corrcoef(y,pred)
        return mcc,-abs(int(pred.sum())-len(pred)/2)
    return float(max(cuts,key=quality))


@contextmanager
def gpu_monitor(directory,stage):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/(stage+'_gpu.csv')).open('w') as log:
        process=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw','--format=csv,nounits','-l','5'],stdout=log,stderr=subprocess.STDOUT)
        try:yield
        finally:
            process.terminate();process.wait()


def execute(request):
    import torch
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(4)
    settings=request.get('settings',{})
    if settings.get('device','cuda')!='cuda' or not torch.cuda.is_available():raise RuntimeError('TULIP requires a working CUDA GPU; CPU fallback is disabled')
    seed=int(settings.get('seed',42));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model,tokenizer,mhctok=load_model(request['repo'])
    model.skipMiss=True
    checkpoint=request.get('checkpoint') if request['stage']=='predict' else request.get('init_checkpoint')
    saved={}
    if checkpoint:
        payload=torch.load(checkpoint,map_location='cpu');saved=payload.get('adapter_metadata',{})
        model.load_state_dict(payload.get('state_dict',payload),strict=True);del payload
        if saved and saved.get('model')!='tulip-tcr':raise ValueError('Wrong checkpoint model')
    else:
        for parameter in model.parameters():
            if parameter.dim()>1:torch.nn.init.xavier_normal_(parameter)
    model=model.cuda()
    batch_size=int(settings.get('batch_size',512))
    log_dir=Path(request['log_dir'])
    with gpu_monitor(log_dir,request['stage']):
        print(json.dumps(dict(device=str(next(model.parameters()).device),gpu=torch.cuda.get_device_name(0),parameters=sum(p.numel() for p in model.parameters()),batch_size=batch_size,training_objective='positive-only LM + MLM per token',score='exp(official peptide log-likelihood)')),flush=True)
        if request['stage']=='predict':
            frame=pd.read_csv(request['input'],keep_default_na=False)
            if 'label' in frame or 'binder' in frame:raise ValueError('Prediction input must not contain binding labels')
            arrays=encode(frame,tokenizer,mhctok,'cuda')
            scores=np.exp(log_scores(model,arrays,len(frame),batch_size))
            threshold=request.get('threshold')
            if threshold is None:threshold=saved.get('threshold',.5)
            pd.DataFrame(dict(row_id=frame.row_id,score=scores,threshold=threshold)).to_csv(request['output'],index=False)
            print(json.dumps(dict(rows=len(frame),threshold=float(threshold))),flush=True)
            return
        if request['stage']!='train':raise ValueError('Unknown stage')
        output=Path(request['output']);output.mkdir(parents=True,exist_ok=True)
        if (output/'best.pt').exists():raise FileExistsError(output/'best.pt')
        train=pd.read_csv(request['train'],keep_default_na=False);valid=pd.read_csv(request['valid'],keep_default_na=False)
        if not (train.label==1).all():raise ValueError('TULIP generative training must contain positives only')
        train_arrays=encode(train,tokenizer,mhctok,'cuda');valid_arrays=encode(valid,tokenizer,mhctok,'cuda')
        optimizer=torch.optim.AdamW(model.parameters(),lr=float(settings.get('learning_rate',1e-4)),weight_decay=float(settings.get('weight_decay',0.)))
        scaler=torch.cuda.amp.GradScaler()
        epochs=int(settings.get('epochs',30));patience=int(settings.get('patience',4));best=-float('inf');stale=0;history=[]
        for epoch in range(1,epochs+1):
            started=time.time();model.train();total_loss=0.;total_tokens=0
            order=torch.randperm(len(train),device='cuda')
            for start in range(0,len(train),batch_size):
                optimizer.zero_grad(set_to_none=True)
                items=batch(train_arrays,order[start:start+batch_size])
                with torch.cuda.amp.autocast():loss,count=generative_loss(model,items,tokenizer)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite TULIP training loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update()
                tokens=int(count);total_loss+=float(loss.detach())*tokens;total_tokens+=tokens
            raw=log_scores(model,valid_arrays,len(valid),batch_size)
            frame=valid.assign(raw=raw)
            per_peptide=[roc_auc_score(g.label,g.raw) for _,g in frame.groupby('peptide') if g.label.nunique()==2]
            auc=float(np.mean(per_peptide));pooled=float(roc_auc_score(valid.label,raw))
            history.append(dict(epoch=epoch,train_loss=total_loss/total_tokens,validation_macro_peptide_auroc=auc,validation_pooled_auroc=pooled,seconds=time.time()-started,peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/1024**2))
            print(json.dumps(history[-1]),flush=True)
            if auc>best:
                best=auc;stale=0
                threshold=choose_threshold(valid.label.to_numpy(),np.exp(raw))
                metadata=dict(model='tulip-tcr',threshold=threshold,best_epoch=epoch,validation_macro_peptide_auroc=auc,selection_metric='validation macro per-peptide AUROC',score='conditional peptide sequence probability; comparable primarily within peptide',initialization='pretrained' if checkpoint else 'scratch',seed=seed)
                torch.save(dict(state_dict=model.state_dict(),adapter_metadata=metadata),output/'best.pt')
            else:stale+=1
            write_json(log_dir/'history.json',history)
            if stale>=patience:break


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True)
    execute(json.loads(parser.parse_args().request.read_text()))
