import unittest
import pandas as pd
from baselines.adapters import get_adapter

class TulipTests(unittest.TestCase):
    def test_paired_cdr3_mhc_and_native_unknowns_preserved(self):
        frame=pd.DataFrame(dict(pep=['AAAA','CCCC'],ab=['WRONG/WRONG']*2,cdr3_ab=['CA#?/CASOF','/CASSF'],**{'hla.allele':['A*02:01','B*07:02']},label=[1,0]))
        out=get_adapter('tulip-tcr').convert(frame,{})
        self.assertEqual(out.alpha.tolist(),['CA#?','<MIS>'])
        self.assertEqual(out.beta.tolist(),['CASOF','CASSF'])
        self.assertEqual(out.mhc.tolist(),['HLA-A*02:01','HLA-B*07:02'])
        self.assertNotIn('label',out)

    def test_positive_only_and_validation_labels_retained(self):
        frame=pd.DataFrame(dict(pep=['AAAA','AAAA'],ab=['CAF/CASF','CAF/CATF'],label=[1,0]))
        adapter=get_adapter('tulip-tcr');converted=adapter.convert(frame,{})
        train,valid=adapter.prepare_training(frame,frame,converted,converted,None,42)
        self.assertEqual(train.label.tolist(),[1])
        self.assertEqual(valid.label.tolist(),[1,0])
        self.assertFalse(adapter.uses_negative_exclusion)

    def test_missing_both_chains_rejected(self):
        with self.assertRaises(ValueError):get_adapter('tulip-tcr').convert(pd.DataFrame(dict(pep=['AAAA'],ab=['/'])),{})

if __name__=='__main__':unittest.main()
