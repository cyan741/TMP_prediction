# Whole-corpus unseen-peptide benchmark

This human HLA-I Level III benchmark is built from all supported positive observations in `corpora/hitph260630.csv`. No low-frequency tail or edit-distance filter is used.

```bash
python build_full_unseen_benchmark.py \
  --seed 42 --peptides-per-eval 500 --cap-basis positive \
  --output workspaces/unseen_full_seed42
```

Each peptide belongs to exactly one of train, valid, test. Validation and test each contain 500 peptide identities. The source-positive observation count of each evaluation set must be at most `floor(0.1 * source_rows)`; negative records are additional. Both raw-positive counts and deduplicated positive counts must differ by at most 5% between validation and test, relative to the larger count. Training retains original duplicate observations; evaluation positives are deduplicated using the repository's peptide/MHC/endpoint-normalized paired-TCR rule.

Sampling starts from lexical peptide order and draws full uniform permutations with NumPy RandomState(seed). The first partition satisfying the constraints is accepted. A rejection consumes the complete permutation; the accepted attempt is recorded. No distance or model performance is used to select the partition. This is random peptide sampling conditioned on sample-count constraints, not an unconditional simple random sample. Peptides whose individual source counts exceed the evaluation cap must stay in training; these are explicitly listed in the report. In particular, source frequencies of 29,737 and 26,813 exceed the 12,269-row evaluation cap. Therefore the benchmark cannot assess unseen performance on these two peptide identities while preserving both full-group holdout and the requested cap. It estimates performance for feasible unseen-peptide holdouts under this training and split protocol, not unconditional performance on every possible held-out source peptide.

## Negative sampling and assets

- Training CSV contains positives only. Existing PLM training dynamically draws one negative TCR per positive per epoch.
- Validation/test CSVs contain fixed positive/negative records with exactly 1:1 labels after deduplication, seeds 1042/2042 respectively.
- The negative scheme retains each positive's peptide and MHC and replaces paired TCR from training donors. Known positive peptide/TCR endpoint groups are excluded across MHC alleles. This is the original repository's random mismatch scheme, with duplicate negative groups rejected during fixed generation to preserve exactly 1:1 after evaluation deduplication. That duplicate repair is an intentional small difference from the original independent-with-replacement fixed sampler.
- `hitph/tcr2candidates_pools.npy` is the sorted unique FINAL training `ab` pool, compatible with the training loader.
- `hitph/positive_blacklist_train_valid.csv` is the exact unique `pep,ab` union of all original source positives assigned to train and valid, before evaluation deduplication or negative generation. This preserves exact sequence variants collapsed by evaluation grouping.
- `hitph/negative_blacklist_corpus.csv` additionally expands endpoint-equivalent TCR identities to exact training donor sequences, so the original exact-string runtime sampler respects the repository's conservative grouping rule. It contains no negative observations or test peptide keys. Test generation separately screens against ALL source-positive endpoint groups, including test positives. Because peptide sets are disjoint and train/val negative sampling retains peptide, test keys cannot occur in training or validation sampling.

## Why validation is fixed

Fixed validation compares checkpoints on the same cases, stabilizing AUROC-based early stopping and MCC-based classification-threshold selection. Train negatives change each epoch to expose more mismatches. Test negatives stay fixed so experiments are comparable. A new training seed must not redraw validation/test data.

Redrawing validation negatives each epoch is reasonable when the intended selection objective is the expectation over the mismatch distribution, but one draw adds sampling noise to model improvement and checkpoint choice. A better sensitivity check is to predefine several fixed negative panels and report their average/spread using identical panels across models. Such panels do not change validation peptides and should not be used to choose a split based on test results. The main benchmark uses one fixed panel and the existing `train.py` setting `fixed_validation=True`.

## Training and reporting

Use `workspaces/unseen_full_seed42/hitph` directly as `train.py train --workspace`. This builder fixes both evaluation peptide sets jointly; rerunning the legacy `prepare` selector is unnecessary and can choose a different validation set.

`benchmark_report.json` records source SHA256, seeds, accepted draw, limits, forced-training groups and split counts. `peptide_frequencies.csv` records all peptide assignments. `peptide_distances.csv` computes exact minimum unit-cost Levenshtein distance to FINAL training peptides; validation is excluded. `distance_strata.csv` contains evaluation counts by exact distance. Annotated validation/test CSVs preserve the fixed set rows and add distance and one lexical nearest training peptide.

Score `hitph/test_data_fold0.csv` using the best trained checkpoint via existing `train.py inference`, which preserves its labels and row order. Then run:

```bash
python report_distance_metrics.py \
  --predictions predictions/unseen_full_test.csv \
  --benchmark workspaces/unseen_full_seed42 \
  --output predictions/unseen_full_distance_metrics.json
```

Report both pooled record AUROC/AUPRC and mean per-peptide AUROC/AUPRC. The latter gives equal weight to peptide identities; the former weights identities by their evaluation records. Neither removes the restriction imposed by infeasible large holdout groups. Natural distance strata are not balanced. Interpret small strata with their peptide counts and, where needed, peptide-bootstrap confidence intervals. Model scores and synthetic mismatch labels do not establish experimental binding probabilities.

## Generated seed-42 results

The accepted partition is permutation attempt 312. The entire source has 122,694 positive observations and 3,337 peptide identities.

| Split | Peptides | Original source positives | Deduplicated evaluation positives | Fixed negatives | CSV rows |
|---|---:|---:|---:|---:|---:|
| train | 2,337 | 106,918 | Not applied | 0 | 106,918 |
| valid | 500 | 7,994 | 5,868 | 5,868 | 11,736 |
| test | 500 | 7,782 | 5,930 | 5,930 | 11,860 |

Validation/test source positives represent approximately 6.52%/6.34% of the full source, respectively. They have equal peptide counts, and both raw and deduplicated positive counts differ by less than 5%.

| Minimum edit distance | Test peptides | Test positives | Test negatives |
|---|---:|---:|---:|
| 1 | 300 | 2,655 | 2,655 |
| 2 | 29 | 75 | 75 |
| 3 | 25 | 322 | 322 |
| 4 | 55 | 284 | 284 |
| 5 | 79 | 2,561 | 2,561 |
| 6 | 11 | 30 | 30 |
| 7 | 1 | 3 | 3 |

These are natural strata under the accepted count-constrained random split. The one-peptide distance-7 stratum cannot support a broad generalization claim by itself.
