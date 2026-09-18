"""Model-specific input contracts; imports no external model code."""
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pandas as pd

AA = set('ACDEFGHIKLMNPQRSTVWY')


class Adapter(ABC):
    name: str
    dependencies: tuple[str, ...]
    worker: Path
    uses_negative_exclusion = True

    def validate(self, repo, settings):
        return {}

    def training_input(self, frame, settings):
        return frame, self.convert(frame,settings), []

    def convert_exclusions(self, frame, settings):
        return self.convert(frame,settings)

    def prepare_training(self, *args):
        from baselines.data import prepare_training
        return prepare_training(*args)

    @abstractmethod
    def convert(self, frame, settings):
        """Return ordered row_id/peptide/tcr inputs; never use outcome labels."""


class TcrLMAdapter(Adapter):
    name = 'tcrlm'
    dependencies = ('torch', 'einops', 'numpy', 'pandas', 'sklearn')
    worker = Path(__file__).parent / 'workers' / 'tcrlm.py'

    def validate(self, repo, settings):
        for relative in ['models/tcrLM.py','models/encoder.py','data/dict.npy']:
            if not (repo/relative).is_file():
                raise FileNotFoundError(repo/relative)
        if settings.get('mask_width',20) not in (20,34):
            raise ValueError('tcrLM mask_width must be 20 (upstream) or 34 (corrected)')
        return dict(model_inputs=['peptide','single CDR3'],chain=settings.get('chain','beta'),ignored_model_inputs=['MHC','other TCR chain'],mask_width=settings.get('mask_width',20))

    def _project(self, frame, settings):
        if settings.get('chain', 'beta') not in ('alpha', 'beta'):
            raise ValueError('tcrLM chain must be alpha or beta; paired chains are unsupported')
        peptide_column = settings.get('peptide_column', 'pep')
        if peptide_column not in frame:
            raise ValueError(f'Missing peptide column: {peptide_column}')
        explicit = settings.get('tcr_column')
        if explicit:
            if explicit not in frame:
                raise ValueError(f'Missing TCR column: {explicit}')
            tcr = frame[explicit].astype(str).str.strip().str.upper()
        else:
            paired = settings.get('paired_column') or next((c for c in ['cdr3_ab','ab_cdr3','ab'] if c in frame), None)
            if paired is None:
                raise ValueError('Need paired CDR3 column or explicit tcr_column')
            pairs = frame[paired].astype(str).str.strip().str.upper().str.split('/')
            if not pairs.map(lambda p: len(p) == 2 and all(p)).all():
                raise ValueError('Expected explicit alpha/beta CDR3 pairs')
            tcr = pairs.str[1 if settings.get('chain','beta') == 'beta' else 0].str.strip()
        peptide = frame[peptide_column].astype(str).str.strip().str.upper()
        return pd.DataFrame({'row_id':np.arange(len(frame)), 'peptide':peptide.to_numpy(), 'tcr':tcr.to_numpy()})

    def valid_rows(self, converted):
        return converted[['peptide','tcr']].apply(lambda values: values.map(lambda s: 0 < len(s) <= 34 and set(s) <= AA)).all(axis=1)

    def convert(self, frame, settings):
        converted = self._project(frame,settings)
        good = self.valid_rows(converted)
        if not good.all():
            bad = np.flatnonzero(~good.to_numpy())[:5] + 2
            raise ValueError(f'tcrLM: standard residues and length 1..34 required; CSV rows {bad.tolist()}')
        return converted

    def convert_exclusions(self, frame, settings):
        # Exclusion records are not model inputs; retain all known identities.
        return self._project(frame,settings)

    def training_input(self, frame, settings):
        converted = self._project(frame,settings)
        good = self.valid_rows(converted)
        policy = settings.get('unsupported_train','error')
        if policy not in ('error','exclude'):
            raise ValueError('unsupported_train must be error or exclude')
        if not good.all() and policy == 'error':
            self.convert(frame,settings)
        excluded = (np.flatnonzero(~good.to_numpy()) + 2).tolist()
        kept = frame.loc[good.to_numpy()].reset_index(drop=True)
        if kept.empty:
            raise ValueError('No supported training rows')
        return kept,self.convert(kept,settings),excluded


class ATMTCRAdapter(TcrLMAdapter):
    name = 'atm-tcr'
    dependencies = ('torch', 'numpy', 'pandas', 'sklearn')
    worker = Path(__file__).parent / 'workers' / 'atm_tcr.py'
    requires_init_checkpoint = False
    dynamic_negatives = True

    def validate(self, repo, settings):
        for relative in ['attention.py', 'data_loader.py', 'data/blosum/BLOSUM45']:
            if not (repo/relative).is_file():
                raise FileNotFoundError(repo/relative)
        if settings.get('chain', 'beta') != 'beta':
            raise ValueError('ATM-TCR adapter uses beta CDR3 only')
        for key, default in [('max_len_pep',22),('max_len_tcr',20),('lin_size',1024),('heads',5)]:
            if not isinstance(settings.get(key,default),int) or settings.get(key,default)<1:
                raise ValueError(f'{key} must be a positive integer')
        if 25 % settings.get('heads',5):
            raise ValueError('heads must divide embedding dimension')
        if settings.get('padding','mid') not in ('mid','front','end'):
            raise ValueError('ATM-TCR padding must be mid/front/end')
        if not 0 <= settings.get('drop_rate',.25) < .5:
            raise ValueError('drop_rate must be in [0,.5)')
        return dict(model_inputs=['peptide','beta CDR3'], chain='beta',
                    ignored_model_inputs=['MHC','alpha CDR3'], training_initialization='from scratch unless init_checkpoint supplied',
                    max_len_pep=settings.get('max_len_pep',22),max_len_tcr=settings.get('max_len_tcr',20),
                    overlength_policy='reject evaluation; explicit training exclusion only', residue_policy='original tokenizer: B/Z/X accepted, other residue letters mapped to *')

    def convert(self, frame, settings):
        converted = self._project(frame,settings)
        good = self._valid(converted,settings)
        if not good.all():
            raise ValueError(f'ATM-TCR unsupported input rows {(np.flatnonzero(~good.to_numpy())[:5]+2).tolist()}')
        return converted

    def _valid(self, frame, settings):
        return (frame.peptide.map(lambda s: 1 <= len(s) <= settings.get('max_len_pep',22) and set(s)<=set('ABCDEFGHIJKLMNOPQRSTUVWXYZ*'))
                & frame.tcr.map(lambda s: 2 <= len(s) <= settings.get('max_len_tcr',20) and set(s)<=set('ABCDEFGHIJKLMNOPQRSTUVWXYZ*')))

    def training_input(self, frame, settings):
        converted=self._project(frame,settings)
        good=self._valid(converted,settings)
        policy=settings.get('unsupported_train','error')
        if policy not in ('error','exclude'): raise ValueError('unsupported_train must be error or exclude')
        if not good.all() and policy=='error': self.convert(frame,settings)
        kept=frame.loc[good.to_numpy()].reset_index(drop=True)
        if kept.empty: raise ValueError('No supported training rows')
        return kept,self.convert(kept,settings),(np.flatnonzero(~good.to_numpy())+2).tolist()

    def prepare_training(self, train, valid, ct, cv, exclusion, seed):
        from baselines.data import labels, projection_conflicts
        result=ct.assign(label=labels(train).to_numpy())
        if projection_conflicts(result,result.label): raise ValueError('Conflicting projected training labels')
        if set(result.label)=={1} and exclusion is None: raise ValueError('Dynamic negatives require exclusion table')
        return result,cv.assign(label=labels(valid).to_numpy())



class TULIPAdapter(Adapter):
    name='tulip-tcr'
    dependencies=('torch','transformers','tokenizers','numpy','pandas','sklearn')
    worker=Path(__file__).parent/'workers/tulip.py'
    requires_init_checkpoint=False
    uses_negative_exclusion=False

    def validate(self,repo,settings):
        for name in ['src/multiTrans.py','model_weights/config.json','aatok/tokenizer.json','mhctok/tokenizer.json']:
            if not (repo/name).is_file():raise FileNotFoundError(repo/name)
        if settings.get('device','cuda')!='cuda':raise ValueError('TULIP runs require CUDA')
        return dict(model_inputs=['peptide','alpha CDR3','beta CDR3','MHC allele'],training_negatives='none; positive-only generative learning',score='exp(official conditional peptide log-likelihood)',primary_metrics='macro per-peptide AUROC/AUPRC')

    def convert(self,frame,settings):
        peptide_column=settings.get('peptide_column','pep')
        if peptide_column not in frame:raise ValueError('Missing peptide column '+peptide_column)
        peptide=frame[peptide_column].astype(str).str.strip().str.upper()
        paired=settings.get('paired_column') or next((x for x in ['cdr3_ab','ab_cdr3','ab'] if x in frame),None)
        parts=frame[paired].astype(str).str.strip().str.split('/') if paired else None
        if parts is not None and not parts.map(lambda x:len(x)==2).all():raise ValueError('Expected alpha/beta CDR3 pairs')
        chains=[]
        for name,index in [('alpha',0),('beta',1)]:
            column=settings.get(name+'_column')
            if column:
                if column not in frame:raise ValueError('Missing chain column '+column)
                values=frame[column].astype(str).str.strip().str.upper()
            elif parts is not None:values=parts.str[index].str.strip().str.upper()
            else:values=pd.Series('<MIS>',index=frame.index)
            chains.append(values.replace({'':'<MIS>','NA':'<MIS>','NAN':'<MIS>'}))
        mhc_column=settings.get('mhc_column','hla.allele')
        mhc=frame[mhc_column].astype(str).str.strip().str.upper() if mhc_column in frame else pd.Series('<MIS>',index=frame.index)
        mhc=mhc.map(lambda x:'HLA-'+x if x.startswith(('A*','B*','C*','DR','DQ','DP')) else x).replace({'':'<MIS>'})
        for values,allow_missing in [(peptide,False),(chains[0],True),(chains[1],True)]:
            good=values.map(lambda x:(allow_missing and x=='<MIS>') or (0<len(x)<=48 and set(x)<=set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'+('#?*' if allow_missing else ''))))
            if not good.all():raise ValueError('TULIP unsupported input rows '+str((np.flatnonzero(~good.to_numpy())[:5]+2).tolist()))
        if ((chains[0]=='<MIS>') & (chains[1]=='<MIS>')).any():raise ValueError('At least one TCR chain is required')
        return pd.DataFrame(dict(row_id=np.arange(len(frame)),peptide=peptide.to_numpy(),alpha=chains[0].to_numpy(),beta=chains[1].to_numpy(),mhc=mhc.to_numpy(),tcr=(chains[0]+'/'+chains[1]+'|'+mhc).to_numpy()))

    def prepare_training(self,train,valid,ct,cv,exclusion,seed):
        from baselines.data import labels
        y=labels(train)
        positives=ct.loc[y.to_numpy()==1].reset_index(drop=True)
        if positives.empty:raise ValueError('Generative training requires positive records')
        return positives.assign(label=1),cv.assign(label=labels(valid).to_numpy())



ADAPTERS = {'tcrlm': TcrLMAdapter, 'atm-tcr': ATMTCRAdapter, 'tulip-tcr': TULIPAdapter}


def get_adapter(name):
    try:
        return ADAPTERS[name]()
    except KeyError:
        raise ValueError(f'Unknown adapter {name!r}; available: {sorted(ADAPTERS)}') from None
