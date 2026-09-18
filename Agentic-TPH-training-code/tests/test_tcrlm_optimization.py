"""Training-only statistics and compatibility of exported tcrLM classifier weights."""
import unittest
import warnings
import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    torch=None
from baselines.workers.tcrlm import frozen_features, export_state, encode, validation_threshold

@unittest.skipIf(torch is None,'Run with the shared model environment')
class OptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        class Encoder(torch.nn.Module):
            def forward(self,tokens,lengths):
                return tokens.float().unsqueeze(-1).expand(-1,-1,512)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_P=Encoder();self.encoder_T=Encoder()
                self.classifier=torch.nn.Linear(34*512*2,2)
        self.model=Model()
        self.vocab=dict(zip('LAGVSERTIDPKQNFYMHWC',range(20)));self.vocab['-']=20
        self.train=pd.DataFrame({'peptide':['A','A','C'],'tcr':['D','D','C']})
        self.valid=pd.DataFrame({'peptide':['W'],'tcr':['W']})
        self.features=frozen_features(self.model,self.train,self.valid,self.vocab,2,'cpu',standardize=True)

    def test_statistics_weight_training_occurrences_and_exclude_validation(self):
        self.assertAlmostEqual(self.features.mean[0].item(),7.)
        self.assertAlmostEqual(self.features.scale[0].item(),np.sqrt(72),places=5)
        train=self.features(np.arange(3),0)
        self.assertLess(abs(train[:,0].mean().item()),1e-6)
        self.assertAlmostEqual(self.features.mean[512].item(),20.)
        self.assertEqual(self.features.scale[512].item(),1.)
        self.assertTrue(torch.isfinite(self.features(np.arange(1),1)).all())

    def test_constant_threshold_predictions_return_zero_mcc_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            cut=validation_threshold(np.array([0,1]),np.array([.5,.5]))
        self.assertEqual(cut,.5)

    def test_export_preserves_raw_model_logits_without_mutating_training_head(self):
        before=self.model.classifier.weight.detach().clone()
        expected=[self.model.classifier(self.features(np.arange(len(frame)),split)).detach() for split,frame in enumerate([self.train,self.valid])]
        state=export_state(self.model,self.features)
        self.assertTrue(torch.equal(before,self.model.classifier.weight))
        self.model.load_state_dict(state)
        for frame,target in zip([self.train,self.valid],expected):
            p,t,pl,tl=[torch.as_tensor(a) for a in encode(frame,self.vocab)]
            raw=torch.cat([self.model.encoder_P(p,pl),self.model.encoder_T(t,tl)],dim=1).reshape(len(frame),-1)
            actual=self.model.classifier(raw)
            self.assertTrue(torch.allclose(actual,target,atol=1e-4,rtol=1e-4))

if __name__=='__main__':unittest.main()
