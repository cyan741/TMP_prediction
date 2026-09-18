"""Regression checks for configuration overrides and unlabeled prediction output."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from cli_utils import load_prepare_config, prepare_overrides
from inference import read_inputs, write_predictions
from train import build_parser


class InterfaceTests(unittest.TestCase):
    def test_explicit_split_override_preserves_other_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"validation_grouping": "pair_stratified", "split_seed": 43})
            )
            args = build_parser().parse_args(
                ["prepare", "--config", str(path), "--validation-grouping", "peptide"]
            )
            config = load_prepare_config(path, prepare_overrides(args))
            self.assertEqual(config["validation_grouping"], "peptide")
            self.assertEqual(config["split_seed"], 43)

    def test_unlabeled_predictions_keep_order_and_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            target = Path(directory) / "predictions.csv"
            source.write_text("id,pep\nsecond, GILGFVFTL \nfirst,NLVPMVATV\n")
            frame = read_inputs(source, ["pep"])
            write_predictions(frame, [0.2, 0.5], 0.5, target)
            result = pd.read_csv(target)
            self.assertEqual(result.id.tolist(), ["second", "first"])
            self.assertEqual(result.pep.tolist(), frame.pep.tolist())
            self.assertEqual(result.predicted_label.tolist(), [0, 1])
            self.assertNotIn("label", result)
            with self.assertRaises(FileExistsError):
                write_predictions(frame, [0.2, 0.5], 0.5, target)


if __name__ == "__main__":
    unittest.main()
