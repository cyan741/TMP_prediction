# Whole-corpus unseen-peptide benchmark

Run from `/root/Agentic-TPH-training-code`:

```bash
python build_stratified_unseen_benchmark.py
```

The source is `corpora/hitph260630.csv`. All positive rows must be supported by the human HLA-I Level III input contract; the builder rejects unsupported rows rather than silently changing this full-corpus benchmark.

Peptide frequencies count unique peptide/MHC/endpoint-normalized paired-TCR positive groups, using the existing evaluation deduplication rule. Peptides are stratified into counts `1`, `2`, `3–5`, `6–10`, `11–25`, and `>25`. Each evaluation set gets 500 peptide identities. Per-bin quotas are apportioned proportionally with largest remainders, with identical validation/test quotas. Remainder ties follow the listed bin order. The builder sorts peptides alphabetically, then shuffles each bin in listed order using one NumPy RandomState(42); the first quota is test, the second validation, and the rest training. No high-frequency peptide is forced into training, and edit distance does not select splits.

A peptide's entire source observations are assigned to one split. Train/validation/test contain 2,337/500/500 peptides. Training retains source observations, including duplicates; validation/test positives are deduplicated. Validation and test receive one proposed random mismatch negative per unique positive, drawn from FINAL training TCR donors and excluding all known corpus positives across MHC alleles. Negative seeds are 1042/2042. Evaluation deduplication can reduce the resulting negative count. Training negatives remain dynamic in the existing PLM training loop. Synthetic negatives do not represent experimentally confirmed nonbinding observations.

Outputs are in `workspaces/unseen_stratified_seed42/`:

- `hitph/`: ready-to-use train/validation/test CSVs, TCR donor pool, negative blacklist and workspace report.
- `train_peptides.csv`, `valid_peptides.csv`, `test_peptides.csv`: fixed identity lists.
- `peptide_frequencies.csv`, `frequency_strata.csv`: source and unique pair counts, allocation and strata audit.
- `benchmark_report.json`: source SHA256, RNG protocol, split counts, per-bin quotas, deduplication and distances.
- `build_config.json`: invocation parameters; repeat the builder with a NEW `--output` directory to reproduce.
- `peptide_distances.csv`: validation/test minimum exact Levenshtein distance to FINAL training peptides only, nearest lexical match and tie counts.
- `distance_strata.csv`: natural distance distributions; no distance balancing.
- `valid_annotated.csv`, `test_annotated.csv`: fixed evaluation records with distances.

Do not rerun existing `train.py prepare` on the new test and expect the same validation: its record-fraction validation selection differs from this explicitly fixed 500-peptide validation. Use the produced workspace directly.

```bash
python train.py train \
  --workspace workspaces/unseen_stratified_seed42/hitph \
  --asset-root assets --seed 42 \
  --output outputs/unseen_stratified_s42
```

After training, find `best_checkpoint` in `training_result.json` and keep its adjacent selected-threshold file. Generate predictions on the FIXED labeled test CSV, preserving labels and row order:

```bash
python train.py inference \
  --input workspaces/unseen_stratified_seed42/hitph/test_data_fold0.csv \
  --checkpoint /path/to/best_checkpoint.pt \
  --asset-root assets \
  --output predictions/unseen_stratified_test.csv

python report_distance_metrics.py \
  --predictions predictions/unseen_stratified_test.csv \
  --benchmark workspaces/unseen_stratified_seed42 \
  --output predictions/unseen_stratified_distance_metrics.json
```

The metrics script reports overall and natural distance-layer AUROC/AUPRC, row-pooled and equally weighted per-peptide means. F1/MCC use the frozen validation threshold when present. Single-label groups have undefined AUC. This one seed is one benchmark estimate; confidence intervals and further predeclared split seeds can quantify uncertainty. Identical peptide quotas per frequency bin do not guarantee identical record counts, especially for extreme-frequency peptides.

Additional analysis dependency: `rapidfuzz`. Old frequency-tail benchmark outputs remain available separately.

## Generated seed-42 results

| Split | Peptides | Positive records | Negative records | Total records | Assigned source positive rows |
|---|---:|---:|---:|---:|---:|
| Train | 2,337 | 71,203 | 0 | 71,203 | 71,203 |
| Validation | 500 | 21,310 | 19,091 | 40,401 | 35,755 |
| Test | 500 | 14,480 | 13,434 | 27,914 | 15,736 |

Per-evaluation peptide quotas in frequency bins `1`, `2`, `3–5`, `6–10`, `11–25`, `>25` are 212, 79, 101, 46, 43, 19. Both sets have identical per-bin peptide counts. Record counts differ because whole high-frequency peptides are assigned randomly: SLLMWITQV is in training, KLGGALQAK in validation, and NLVPMVATV in test. Negatives are fewer than positives after deduplicating independent with-replacement donor draws.

| Minimum edit distance | Test peptides | Positives | Negatives | Total records |
|---|---:|---:|---:|---:|
| 1 | 304 | 11,825 | 10,794 | 22,619 |
| 2 | 37 | 107 | 107 | 214 |
| 3 | 9 | 13 | 13 | 26 |
| 4 | 52 | 306 | 305 | 611 |
| 5 | 79 | 2,144 | 2,130 | 4,274 |
| 6 | 17 | 61 | 61 | 122 |
| 7 | 1 | 4 | 4 | 8 |
| 8 | 1 | 20 | 20 | 40 |

These counts are descriptive and do not imply model performance. Refer to `verification_report.json` for actual-data integration checks. There are no trained model predictions in this construction run.
