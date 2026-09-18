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



ADAPTERS = {'tcrlm': TcrLMAdapter}


def get_adapter(name):
    try:
        return ADAPTERS[name]()
    except KeyError:
        raise ValueError(f'Unknown adapter {name!r}; available: {sorted(ADAPTERS)}') from None
