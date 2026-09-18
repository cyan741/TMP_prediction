# Frequency-tail unseen-peptide benchmark

Run from the repository root. This benchmark targets human HLA-I Level III.

```bash
python build_unseen_benchmark.py
```

The source is `corpora/hitph260630.csv`. Count raw positive observations per normalized peptide, including duplicate observations. Sort ascending by frequency and then alphabetically by peptide. The candidate tail contains `floor(0.8 * number_of_peptides)` peptides; this is 80% of peptide identities, not 80% of records. Equal-frequency ties at the boundary are resolved alphabetically. Draw 100 candidates without replacement with `numpy.random.RandomState(42).choice`. All observations for those peptides are held out. Selected unsupported observations cause an error instead of silently changing the test set.

Validation uses the existing `select_validation(..., grouping='peptide', fraction=0.1, seed=42)` on the remaining supported corpus. The target is 10% of unique positive peptide/MHC/endpoint-normalized paired-TCR groups, not 10% of peptide identities or raw records. The existing seeded subset-sum algorithm chooses whole peptides near that target. Training retains eligible original observations, including duplicates. Validation and test positives are deduplicated with the existing endpoint-equivalence rule. This validation baseline is unseen peptide, but is not explicitly matched to the low-frequency test distribution. If that distribution is the intended deployment target, predefine a frequency-matched validation protocol separately; do not choose it using test model performance.

Generate one random mismatched TCR negative per unique evaluation positive, using only FINAL training TCR donors and excluding all known corpus positives across MHC alleles. Validation uses negative seed 1042; test uses 2042. Evaluation pairs are deduplicated after negative generation, so label balance is measured rather than assumed. Negatives are synthetic mismatches, not experimentally confirmed nonbinders.

Outputs are written to `workspaces/unseen_tail80_seed42/`. The `hitph/` subdirectory is compatible with `train.py train`, `eval`, and `inference`. `prepare_config.json` allows existing `train.py prepare` to reproduce the split using the newly fixed test CSV. Source frequencies, selected peptides, RNG, source SHA256, deduplication and split counts are recorded in `benchmark_report.json` and `peptide_frequencies.csv`.

`peptide_distances.csv` contains exact minimum Levenshtein distance to FINAL training peptides (validation is excluded), one lexical nearest match and nearest-match tie counts. Unit insertion, deletion and substitution costs are used. `distance_strata.csv` reports counts for test and validation separately; `test_annotated.csv` and `valid_annotated.csv` carry the corresponding distances per record. Peptide grouping permits near-neighbor peptides across splits; cluster grouping would change this benchmark and its low-distance strata.

After training, use the best checkpoint and its saved validation threshold to score the fixed labeled test CSV through `train.py inference`, preserving labels and row order. Then run:

```bash
python report_distance_metrics.py \
  --predictions predictions/unseen_test.csv \
  --benchmark workspaces/unseen_tail80_seed42 \
  --output predictions/unseen_distance_metrics.json
```

This reports overall and exact-distance AUROC/AUPRC, plus equally weighted mean per-peptide AUROC/AUPRC. If a frozen threshold is present, it also reports F1 and MCC. Groups without both labels have null AUC values. The script requires all fixed-test rows in their original order and checks identities/labels before joining distances. The prediction set must not select a threshold. Counts alone do not establish a performance trend; small strata, peptide frequencies, lengths, MHC and synthetic negatives can affect interpretation. Peptide-bootstrap confidence intervals are a suitable follow-up for uncertainty, especially for small strata.

Additional analysis dependency: `rapidfuzz`. This does not change the training runtime dependencies.

## Seed-42 generated counts

The source has 122,694 positive rows and 3,337 peptides. The low-frequency tail has 2,669 peptides, with a maximum source frequency of 8. The 100 selected test peptides have 248 source rows and 196 positive evaluation groups after deduplication. The fixed test contains 196 positive and 196 negative rows. Final training has 109,843 positive rows over 2,389 peptides; validation has 9,561 positives and 9,541 negatives over 848 peptides. Training negatives remain dynamic in the existing PLM training loop.

| Minimum edit distance | Test peptides | Positive rows | Negative rows | Total rows |
|---|---:|---:|---:|---:|
| 1 | 66 | 136 | 136 | 272 |
| 2 | 4 | 6 | 6 | 12 |
| 3 | 1 | 4 | 4 | 8 |
| 4 | 16 | 27 | 27 | 54 |
| 5 | 12 | 20 | 20 | 40 |
| 6 | 1 | 3 | 3 | 6 |

This is unseen peptide, with many near neighbors retained in training. It is not an unseen-cluster benchmark. Actual model AUC values require a trained checkpoint and predictions; no model performance is inferred from these counts.
