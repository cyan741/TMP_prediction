import unittest

import pandas as pd
from core_engine.trainer.fixed_splits import (
    apply_validation,
    cdr3_group,
    deduplicate_evaluation,
    potential_pair_keys,
    select_validation,
)
from pan_cnn import negatives


class SplitTests(unittest.TestCase):
    def test_endpoints_and_retention(self):
        f = pd.DataFrame(
            {
                "pep": ["PEPTIDEA"] * 4 + ["SINGLE"],
                "hla.allele": ["A"] * 5,
                "ab": [
                    "CASSF/CATW",
                    "ASS/AT",
                    "CGGGF/CGGTW",
                    "CHHHF/CHHTW",
                    "CONEF/CTWOW",
                ],
                "label": [1] * 5,
            }
        )
        val, _ = select_validation(
            f, peers={"f": f}, grouping="pair_stratified", fraction=0.4
        )
        train, _ = apply_validation(f, val, grouping="pair_stratified")
        self.assertFalse(
            set(potential_pair_keys(train)) & set(potential_pair_keys(val))
        )
        self.assertEqual(set(train.pep), set(f.pep))
        self.assertEqual(cdr3_group("CASSF/CATW"), cdr3_group("ASS/AT"))

    def test_negative_exclusions_apply_across_mhc_alleles(self):
        f = pd.DataFrame(
            {
                "domain": ["human_I"] * 2,
                "pep": ["PEPTIDE"] * 2,
                "hla.allele": ["human|I|A"] * 2,
                "tcr_alpha_cdr3": ["CASSF", "CGGGF"],
                "tcr_beta_cdr3": ["CATW", "CGGTW"],
                "ab": ["CASSF/CATW", "CGGGF/CGGTW"],
                "pair_key": ["one", "two"],
                "label": [1, 1],
            }
        )
        blocked = {
            (pep, "DIFFERENT_MHC", group) for pep, _, group in potential_pair_keys(f)
        }
        with self.assertRaises(ValueError):
            negatives(f, f, blocked, 42)

    def test_conflicting_evaluation_labels_rejected(self):
        f = pd.DataFrame(
            {
                "pep": ["PEPTIDE"] * 2,
                "hla.allele": ["A"] * 2,
                "ab": ["CASSF/CATW", "ASS/AT"],
                "label": [1, 0],
            }
        )
        with self.assertRaises(ValueError):
            deduplicate_evaluation(f)


if __name__ == "__main__":
    unittest.main()
