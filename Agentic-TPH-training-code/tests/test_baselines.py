"""External baseline contracts verified without launching a model experiment."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import baseline
from baselines.adapters import get_adapter
from baselines.data import prepare_training
from baselines.workers.tcrlm import encode, load_vocab, initialize

class BaselineTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo=Path(temp.name)
        (self.repo/'models').mkdir(); (self.repo/'data').mkdir()
        (self.repo/'models/tcrLM.py').write_text('')
        (self.repo/'models/encoder.py').write_text('')
        vocab=dict(zip('LAGVSERTIDPKQNFYMHWC',range(20))); vocab['-']=20
        np.save(self.repo/'data/dict.npy',vocab)

    def frame(self):
        return pd.DataFrame({'pep':['AAAAAAAC','CCCCCCCA'],'ab':['CAVF/CASSF','CAVF/CAGGF'],'label':['1','1']})

    def config(self,root,workflow='predict'):
        source=root/'input.csv'; self.frame().to_csv(source,index=False)
        checkpoint=root/'weights.bin'; checkpoint.touch()
        return dict(model='tcrlm',workflow=workflow,repo=str(self.repo),python=sys.executable,input=str(source),checkpoint=str(checkpoint),output=str(root/'out'),log_dir=str(root/'logs'),checkpoint_dir=str(root/'checkpoints'),settings={'chain':'beta'})

    def test_cdr3_identity_preferred_and_labels_not_converted(self):
        f=self.frame(); f['cdr3_ab']=['CAGF/CATF','CAGF/CAWF']
        converted=get_adapter('tcrlm').convert(f,{})
        self.assertEqual(converted.tcr.tolist(),['CATF','CAWF'])
        self.assertNotIn('label',converted)
        self.assertEqual(converted.row_id.tolist(),[0,1])

    def test_unsupported_train_explicit_exclusion_and_eval_rejection(self):
        f=self.frame(); f.loc[0,'ab']='CAVF/CASOF'
        adapter=get_adapter('tcrlm')
        with self.assertRaises(ValueError): adapter.convert(f,{})
        with self.assertRaises(ValueError): adapter.training_input(f,{})
        kept,converted,rows=adapter.training_input(f,{'unsupported_train':'exclude'})
        self.assertEqual(rows,[2]); self.assertEqual(len(kept),1)
        self.assertEqual(converted.row_id.tolist(),[0])
        self.assertIn('CASOF',adapter.convert_exclusions(f,{}).tcr.tolist())

    def test_no_negative_projected_as_known_positive(self):
        adapter=get_adapter('tcrlm'); train=self.frame()
        valid=pd.DataFrame({'pep':['DDDDDDDA']*2,'ab':['CAVF/CATF','CAVF/CAWF'],'label':['1','0']})
        ct,cv=adapter.convert(train,{}),adapter.convert(valid,{})
        exclusion=pd.DataFrame({'peptide':['AAAAAAAC'],'tcr':['ASS']})
        result,validation=prepare_training(train,valid,ct,cv,exclusion,42)
        self.assertEqual(result.label.value_counts().to_dict(),{1:2,0:2})
        self.assertEqual(result.loc[result.label==0,'tcr'].tolist(),['CAGGF','CASSF'])
        self.assertEqual(validation.label.tolist(),[1,0])
        with self.assertRaises(ValueError): prepare_training(train,valid,ct,cv,None,42)

    def test_vocabulary_padding_lengths_and_overlength(self):
        vocab=load_vocab(self.repo)
        f=pd.DataFrame({'peptide':['AAAAAAAC'],'tcr':['CASSF']})
        p,t,pl,tl=encode(f,vocab)
        self.assertEqual(p.shape,(1,34)); self.assertEqual(t[0,-1],20)
        self.assertEqual(pl.tolist(),[8]); self.assertEqual(tl.tolist(),[5])
        f.loc[0,'tcr']='A'*35
        with self.assertRaises(ValueError): encode(f,vocab)

    def test_pretrain_and_finetuned_state_key_normalization(self):
        loaded=[]
        encoder=SimpleNamespace(load_state_dict=lambda state,strict:loaded.append((state,strict)))
        model=SimpleNamespace(encoder_T=encoder,encoder_P=encoder,load_state_dict=lambda state,strict:loaded.append((state,strict)))
        initialize(model,{'state_dict':{'module.protflash.token_emb.weight':1,'module.fc_out.weight':2}},'pretrain')
        self.assertEqual(loaded,[({'token_emb.weight':1},True)]*2)
        initialize(model,{'state_dict':{'module.classifier.weight':3}},'finetuned')
        self.assertEqual(loaded[-1],({'classifier.weight':3},True))

    def test_prepare_predict_is_label_free_and_launches_no_worker(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=self.config(Path(d))
            info=baseline.prepare(cfg)
            converted=pd.read_csv(Path(cfg['output'])/'data/input.csv')
            self.assertNotIn('label',converted)
            self.assertEqual([s['stage'] for s in info['stages']],['predict'])
            self.assertFalse((Path(cfg['output'])/'predictions.csv').exists())
            with self.assertRaises(FileExistsError): baseline.prepare(cfg)

    def test_train_prepare_preserves_splits_and_prepares_requests(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cfg=self.config(root,'train-predict')
            train=self.frame(); train.to_csv(root/'train.csv',index=False)
            valid=pd.DataFrame({'pep':['DDDDDDDA']*2,'ab':['CAVF/CATF','CAVF/CAWF'],'label':['1','0']})
            valid.to_csv(root/'valid.csv',index=False)
            test=pd.DataFrame({'pep':['EEEEEEEA']*2,'ab':['CAVF/CATF','CAVF/CAWF'],'label':['1','0']})
            test.to_csv(root/'input.csv',index=False)
            train[['pep','ab']].to_csv(root/'blocked.csv',index=False)
            cfg.update(train=str(root/'train.csv'),valid=str(root/'valid.csv'),negative_exclusion=str(root/'blocked.csv'),init_checkpoint=cfg['checkpoint'])
            info=baseline.prepare(cfg)
            self.assertEqual(info['training_prepared_rows'],4)
            self.assertEqual([s['stage'] for s in info['stages']],['train','predict'])
            request=json.loads((root/'out/predict_request.json').read_text())
            self.assertEqual(request['checkpoint'],str(root/'checkpoints/best.pt'))
            self.assertNotIn('label',pd.read_csv(root/'out/data/input.csv'))
            valid.loc[0,'pep']=train.pep.iloc[0]; valid.to_csv(root/'valid.csv',index=False)
            with self.assertRaises(ValueError): baseline.plan(cfg)

    def test_run_restores_row_order_and_outputs_only_final_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cfg=self.config(root)
            cfg['run_dir']=str(root/'runs')
            def mock_run(argv,**kwargs):
                request=json.loads(Path(argv[-1]).read_text())
                pd.DataFrame({'row_id':[1,0],'score':[.8,.2],'threshold':[.5,.5]}).to_csv(request['output'],index=False)
                return SimpleNamespace(returncode=0)
            with patch('baseline.subprocess.run',side_effect=mock_run): baseline.run(cfg)
            p=pd.read_csv(root/'out/predictions.csv')
            self.assertEqual(p.score.tolist(),[.2,.8])
            self.assertEqual(p.label.tolist(),[1,1])
            self.assertEqual(p.pep.tolist(),self.frame().pep.tolist())
            self.assertEqual({f.name for f in (root/'out').iterdir()},{'predictions.csv','metrics.json'})
            self.assertNotIn('input_sha256',json.dumps(baseline.plan(cfg)))
            self.assertNotIn('repo_commit',json.dumps(baseline.plan(cfg)))

    def test_evaluate_new_labeled_predictions_and_reject_unlabeled(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p=root/'scores.csv'
            pd.DataFrame({'peptide':['AAAAAAAC']*2,'label':[0,1],'score':[.1,.9],'threshold':[.5,.5]}).to_csv(p,index=False)
            result=baseline.evaluate(p,root/'metrics.json',peptide_column='peptide')
            self.assertEqual(result['overall']['auroc'],1.)
            self.assertEqual(result['overall']['macro_eligible_peptides'],1)
            pd.DataFrame({'peptide':['AAAAAAAC'],'score':[.1]}).to_csv(p,index=False)
            with self.assertRaises(ValueError): baseline.evaluate(p,root/'missing.json',peptide_column='peptide')

    def test_relative_paths_resolve_from_config_not_cwd(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            config=root/'config.json'
            config.write_text(json.dumps({'model':'tcrlm','repo':str(self.repo),'input':'input.csv','checkpoint':'weights.bin','output':'out'}))
            cfg=baseline.load_config(config)
            self.assertEqual(cfg['input'],str(root/'input.csv'))
            self.assertEqual(cfg['output'],str(root/'out'))


if __name__=='__main__': unittest.main()
