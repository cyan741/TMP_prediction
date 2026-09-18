import unittest
import pandas as pd
from build_stratified_unseen_benchmark import allocate_peptides


class StratifiedUnseenTests(unittest.TestCase):
    def test_exact_counts_disjoint_assignment_and_frequency_coverage(self):
        counts=[1,2,4,8,20,30]
        f=pd.DataFrame({'pep':[f'P{i:04}' for i in range(600)],'unique_positive_pairs':[counts[i%6] for i in range(600)]})
        result,quotas=allocate_peptides(f,90,42)
        self.assertEqual(result.split.value_counts().to_dict(),{'train':420,'valid':90,'test':90})
        for b,q in quotas.items():
            g=result.loc[result.frequency_bin==b]
            self.assertEqual((g.split=='test').sum(),q)
            self.assertEqual((g.split=='valid').sum(),q)
        again,_=allocate_peptides(f.sample(frac=1,random_state=9),90,42)
        pd.testing.assert_frame_equal(result,again)
        other,_=allocate_peptides(f,90,43)
        self.assertFalse(result.split.equals(other.split))

    def test_largest_remainder_hits_exact_total(self):
        f=pd.DataFrame({'pep':[f'P{i:04}' for i in range(3337)],'unique_positive_pairs':[(i%40)+1 for i in range(3337)]})
        result,quota=allocate_peptides(f,500,42)
        self.assertEqual(sum(quota.values()),500)
        self.assertEqual(result.split.value_counts().to_dict(),{'train':2337,'valid':500,'test':500})

    def test_invalid_evaluation_size_rejected(self):
        f=pd.DataFrame({'pep':['A','B'],'unique_positive_pairs':[1,1]})
        with self.assertRaises(ValueError):
            allocate_peptides(f,1)
