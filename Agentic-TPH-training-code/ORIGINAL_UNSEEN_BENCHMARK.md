# Unseen-peptide data regenerated with original repository functions

The current regenerated workspace is `workspaces/unseen_full_seed42_original/hitph`.

```bash
python regenerate_original_unseen.py
```

This preserves the whole-corpus seed-42 peptide assignment already selected by `build_full_unseen_benchmark.py`: 2,337 training peptides, 500 validation peptides and 500 test peptides. It retains the raw-source-positive 10% cap and validation/test count balance. The regeneration checks the source SHA256 and records the assignment SHA256. Peptide assignment is the previously defined constrained random procedure; evaluation data handling and asset writing use original repository functions directly.

- `deduplicate_evaluation(identity(...))` groups by peptide, MHC identity and endpoint-equivalent alpha/beta CDR3s, keeping the first original row.
- `model_frame()` restores the retained original model `ab`; endpoint-stripped keys are not written as model inputs.
- `fixed_negatives()` generates one random training-TCR mismatch per evaluation positive, seeds 1042 for validation and 2042 for test. Known source positives are screened across MHC using exact sequences and endpoint groups.
- Combined positive/negative evaluation records are deduplicated using the original function. Repeated negative groups are removed without replenishment, so final label counts can differ slightly from 1:1.
- `save_workspace()` writes train/valid/test CSVs, the sorted unique training `tcr2candidates_pools.npy`, and its original `negative_blacklist_corpus.csv`. That runtime blacklist includes training positives and all fixed evaluation records (both labels), expanded to endpoint-equivalent training donor sequences. It is not replaced by a custom positives-only runtime file.
- A separate `positive_blacklist_train_valid.csv` records all exact original train+valid positive `pep,ab` pairs, including source sequence variants merged in evaluation deduplication.

Training retains original positive rows and uses the existing dynamic 1:1 per-epoch negative sampling. Validation and test are fixed. The training CLI already uses `fixed_validation=True`.

Use this workspace with the existing `train.py train --workspace`. For labeled testing, use `train.py eval` and the selected checkpoint's saved validation threshold. For distance reporting, score the fixed labeled test CSV with `train.py inference`, then run:

```bash
python report_distance_metrics.py \
  --predictions predictions/unseen_original_test.csv \
  --benchmark workspaces/unseen_full_seed42_original \
  --output predictions/unseen_original_distance_metrics.json
```

`benchmark_report.json` records counts before and after positive/combined deduplication. Distance annotations retain the previously computed exact Levenshtein distances, since the training peptide set is identical, and update the evaluation label counts. Previous workspaces are retained.

The two source peptide groups larger than the 10% cap remain training-only; the benchmark is still conditioned on feasible whole-peptide holdouts. No model training or model-performance measurement is performed by regeneration.

## Regenerated counts

| Split | Peptides | Original source positives | CSV positives | CSV negatives | CSV rows |
|---|---:|---:|---:|---:|---:|
| train | 2,337 | 106,918 | 106,918 | 0 | 106,918 |
| valid | 500 | 7,994 | 5,868 | 5,823 | 11,691 |
| test | 500 | 7,782 | 5,930 | 5,904 | 11,834 |

Original combined evaluation deduplication removed 45 validation negatives and 26 test negatives. No replacement negatives were drawn. Positive counts and all peptide assignments are unchanged from the preceding full-corpus benchmark.
