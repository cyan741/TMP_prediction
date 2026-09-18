"""Full-corpus split constraints and fixed negative group isolation."""
import unittest
import pandas as pd
from build_full_unseen_benchmark import choose_split, fixed_evaluation
from core_engine.trainer.fixed_splits import cdr3_group


class FullUnseenTests(unittest.TestCase):
    def test_whole_groups_constraints_and_reproducibility(self):
        counts=pd.DataFrame({'pep':list('ABCDEFGH'),'source_positive_rows':[100,1,2,3,4,5,6,7],'unique_positive_pairs':[100,1,2,3,4,5,6,7]})
        a=choose_split(counts,42,2,10,.5)
        self.assertEqual(a,choose_split(counts,42,2,10,.5))
        v,t,_=a
        self.assertFalse(v&t)
        self.assertNotIn('A',v|t)
        self.assertEqual(len(v),2)
        for selected in [v,t]:self.assertLessEqual(counts.loc[counts.pep.isin(selected),'source_positive_rows'].sum(),10)

    def test_fixed_negatives_preserve_exact_ratio_and_exclude_endpoint_variants(self):
        donors=pd.DataFrame({'ab':['CASSF/CATW','CGGGF/CGGTW','CHHHF/CHHTW','CJJJF/CJJTW'],'cdr3_ab':['CASSF/CATW','CGGGF/CGGTW','CHHHF/CHHTW','CJJJF/CJJTW']})
        positive=pd.DataFrame({'pep':['PEPTIDEA']*2,'hla':['X'*34]*2,'hla.allele':['A']*2,'ab':['ASS/AT','COTHERF/COTHERW'],'cdr3_ab':['ASS/AT','COTHERF/COTHERW'],'beta':['AT','COTHERW'],'label':[1,1]})
        known={'PEPTIDEA':set(positive.cdr3_ab.map(cdr3_group))}
        f,r=fixed_evaluation(positive,donors,known,1042)
        negatives=f.loc[f.label==0]
        self.assertEqual(len(negatives),2)
        self.assertEqual(r['duplicate_rows_removed'],0)
        self.assertEqual(negatives.ab.map(cdr3_group).nunique(),2)
        self.assertFalse(set(negatives.ab.map(cdr3_group))&known['PEPTIDEA'])
        pd.testing.assert_frame_equal(f,fixed_evaluation(positive,donors,known,1042)[0])


if __name__=='__main__':unittest.main()
