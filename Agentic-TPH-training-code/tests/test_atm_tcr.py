"""ATM integration: beta projection, TSV, donor exclusions and scratch requests."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import baseline
from baselines.adapters import get_adapter
from baselines.workers.atm_tcr import screened_candidates, dynamic_epoch, encode, ARCH_DEFAULTS

REPO=Path(__file__).resolve().parents[2]/'baseline_runs/repos/ATM-TCR'

class ATMTests(unittest.TestCase):
    def test_tsv_preserves_order_and_prediction_ignores_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'input.tsv'
            original=pd.DataFrame({'ID':['b','a'],'peptide':['AAAAAAAC','CCCCCCCA'],'tcrb_cdr3':['CASSF','CATF'],'label':['0','1']})
            original.to_csv(source,index=False,sep='\t')
            cfg=dict(model='atm-tcr',workflow='predict',repo=str(REPO),python=baseline.sys.executable,input=str(source),checkpoint=str(root/'absent.ckpt'),output=str(root/'out'),settings=dict(peptide_column='peptide',tcr_column='tcrb_cdr3'))
            baseline.prepare(cfg)
            data=pd.read_csv(root/'out/data/input.csv')
            self.assertNotIn('label',data)
            self.assertEqual(data.row_id.tolist(),[0,1])
            adapter=get_adapter('atm-tcr');expected=adapter.convert(original,cfg['settings'])
            original['label']=['1','0']
            pd.testing.assert_frame_equal(adapter.convert(original,cfg['settings']),expected)

    def test_scratch_training_request_keeps_positive_rows_and_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            train=pd.DataFrame(dict(pep=['AAAAAAAC','CCCCCCCA'],ab=['CAVF/CASSF','CAVF/CATF'],label=['1','1']))
            valid=pd.DataFrame(dict(pep=['DDDDDDDA']*2,ab=['CAVF/CASSF','CAVF/CATF'],label=['1','0']))
            test=valid.assign(pep='EEEEEEEA')
            for name,frame in [('train',train),('valid',valid),('test',test),('excluded',train)]:frame.to_csv(root/(name+'.csv'),index=False)
            cfg=dict(model='atm-tcr',workflow='train-predict',repo=str(REPO),python=baseline.sys.executable,input=str(root/'test.csv'),train=str(root/'train.csv'),valid=str(root/'valid.csv'),negative_exclusion=str(root/'excluded.csv'),output=str(root/'out'),settings={})
            result=baseline.prepare(cfg)
            self.assertEqual(result['training_prepared_rows'],2)
            request=json.loads((root/'out/train_request.json').read_text())
            self.assertIsNone(request['init_checkpoint'])
            self.assertTrue(Path(request['negative_exclusion']).is_file())
            self.assertEqual(set(pd.read_csv(root/'out/data/train.csv').label),{1})
            test.loc[0,'pep']='AAAAAAAC';test.to_csv(root/'test.csv',index=False)
            with self.assertRaises(ValueError):baseline.plan(cfg)

    def test_dynamic_negatives_exclude_endpoint_equivalent_held_pairs(self):
        train=pd.DataFrame(dict(peptide=['AAAAAAAC','CCCCCCCA','GGGGGGGA'],tcr=['CASSF','CATF','CAWF'],label=[1,1,1]))
        valid=pd.DataFrame(dict(peptide=['DDDDDDDA']*2,tcr=['CASSF','CATF'],label=[1,0]))
        exclusion=pd.DataFrame(dict(peptide=['AAAAAAAC'],tcr=['AT']))
        candidates=screened_candidates(train,valid,exclusion)
        self.assertEqual(candidates['AAAAAAAC'].tolist(),['CAWF'])
        epoch=dynamic_epoch(train,candidates,43)
        self.assertEqual(epoch.loc[epoch.label==0].iloc[0].tcr,'CAWF')
        self.assertEqual(epoch.label.value_counts().to_dict(),{1:3,0:3})

    def test_upstream_mid_padding_and_no_silent_truncation(self):
        frame=pd.DataFrame(dict(peptide=['ACDEF'],tcr=['CASSF']))
        arch=dict(ARCH_DEFAULTS,max_len_pep=8,max_len_tcr=8)
        pep,tcr=encode(frame,REPO,arch)
        self.assertEqual(pep.tolist(),[[0,4,3,24,24,24,6,13]])
        self.assertEqual(tcr.tolist(),[[4,0,15,24,24,24,15,13]])
        frame.loc[0,'tcr']='CAOSF'
        _,mapped=encode(frame,REPO,arch)
        self.assertEqual(mapped[0,2],23)  # Original O -> * token
        frame.loc[0,'tcr']='A'*21
        with self.assertRaises(ValueError):encode(frame,REPO,ARCH_DEFAULTS)

if __name__=='__main__':unittest.main()
