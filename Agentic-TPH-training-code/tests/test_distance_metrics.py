"""Metric behavior relevant to small unseen-peptide distance strata."""
import unittest
import pandas as pd
from report_distance_metrics import metrics


class DistanceMetricTests(unittest.TestCase):
    def test_single_label_auc_is_undefined(self):
        result = metrics(pd.DataFrame({'pep':['A','A'], 'label':[1,1], 'score':[0.2,0.8]}))
        self.assertIsNone(result['auroc'])
        self.assertIsNone(result['auprc'])
        self.assertEqual(result['macro_eligible_peptides'], 0)

    def test_macro_peptide_weight_is_equal(self):
        frame = pd.DataFrame({'pep':['A']*2+['B']*4, 'label':[0,1,0,0,1,1], 'score':[0.1,0.9,0.8,0.9,0.1,0.2], 'threshold':[0.5]*6})
        result = metrics(frame)
        self.assertEqual(result['macro_peptide_auroc'], 0.5)
        self.assertEqual(result['macro_eligible_peptides'], 2)
        self.assertNotEqual(result['auroc'], result['macro_peptide_auroc'])

    def test_varying_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics(pd.DataFrame({'pep':['A','A'], 'label':[0,1], 'score':[0.1,0.9], 'threshold':[0.3,0.7]}))


if __name__ == '__main__':
    unittest.main()
