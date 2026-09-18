# ATM-TCR baseline integration

ATM-TCR encodes a peptide and one beta-chain CDR3 independently using a shared
trainable amino-acid embedding, two separate multi-head self-attention modules,
flattening and concatenation, then a 2048 → 1024 → 1 dense classifier with
BatchNorm, dropout, SiLU and sigmoid. Default embedding dimension is 25, with
5 attention heads, peptide width 22, TCR width 20 and middle padding.

`baselines/workers/atm_tcr.py` imports the actual upstream `attention.Net`.
It compiles the original vocabulary, tokenizer, embedding loader and `pad`
method from `data_loader.py` without executing its obsolete torchtext imports.
The upstream `blosum=None` default reads BLOSUM45 for matrix dimensions, then
initializes a random trainable embedding; it does **not** initialize embedding
weights from the BLOSUM values. Token IDs, including pad=24, are preserved.
Sequences exceeding the configured widths are rejected, not truncated.
The original tokenizer accepts B/Z/X and maps other residue letters to `*`;
the one training beta sequence containing O is retained under this rule.

## Original repository execution logic

`data_io_tf.py` parses peptide/TCR/label triples; `data_loader.py` builds
cross-validation splits, tokenizes residues and prepares torchtext batches.
`main.py` builds `attention.Net`, uses Adam and binary cross-entropy, and evaluates
the held-out fold each epoch. Its original early stopping checks the average
change in held-out loss over a ten-epoch window after a minimum of 30 epochs;
it saves the final model rather than selecting the best validation AUROC model.
Test mode also requires a training input to construct loaders and a separate
independent test input. `utils.py` computes metrics and writes sequence/score rows.

The adapter bypasses `main.py` to preserve the supplied workspace splits,
restore original row IDs, and select the best validation checkpoint. Both
attention branches are self-attention, with no cross-attention, positional
embedding or padding attention mask. The upstream embedding has 26 rows and
`padding_idx=25`, while its vocabulary's padding token is 24; this behavior is
preserved for checkpoint compatibility and for the newly trained baseline.

## Environment

Both ATM-TCR and tcrLM share `/root/miniconda3/envs/baseline-models`
(Python 3.8, PyTorch 1.10 CUDA 11.3). The runner uses the lightweight
`/root/miniconda3/envs/baseline-host` Python 3.10 environment. Installation
wheels are cached under `/root/miniconda3/pkgs/baseline-wheels`.

## Direct checkpoint prediction

Official checkpoint: https://github.com/Lee-CBG/ATM-TCR/blob/main/models/original.ckpt

The local source snapshot omitted `models/`; the official checkpoint is downloaded
into `../baseline_runs/repos/ATM-TCR/models/original.ckpt`. Provenance and SHA256
are recorded separately. IMMREP2025 TSV uses `peptide` and `tcrb_cdr3`; no label,
alpha-chain or HLA information is passed to the model. Raw upstream checkpoints
use its default threshold 0.5, independently of prediction labels.

```bash
/root/TMP_prediction/baseline_runs/configs/ATM-TCR/run.sh immrep25
```

Outputs: `../baseline_runs/predictions/ATM-TCR/immrep25/`, including
only `predictions.csv` and `metrics.json`. Model weights and logs are in
`../baseline_runs/checkpoints/ATM-TCR/` and `../baseline_runs/logs/ATM-TCR/`.
Predictions retain every original field and the original row order, appending
`score`, `predicted_label`, `threshold`.

## Training and fixed unseen-peptide test

```bash
/root/TMP_prediction/baseline_runs/configs/ATM-TCR/run.sh train
```

Uses `workspaces/hitph0630_unseen_seed42/hitph/` exactly: 106918 training
positive rows, 11691 validation rows and 11834 test rows. Original paired CDR3
identities are projected to beta CDR3. No splits are redrawn, no evaluation rows
are removed, and model initialization is from scratch. All beta CDR3s fit the
original 20-residue limit. Peptides are verified disjoint across the three splits.

Each epoch draws one random mismatched beta CDR3 per training positive from the
training beta pool, with seed `negative_seed + epoch`. Known positives and all
held-out pairs are projected from the workspace exclusion table; terminal
C/F/W notation variants are excluded together, across MHC alleles. Full beta
sequences remain model inputs. Training observations and evaluation labels stay
unchanged. Conflicting labels after single-chain projection are reported for
validation/test; training contradictions are rejected. The actual validation
and test data have 15 and 2 conflicting projected groups respectively; all
evaluation rows and labels are retained.

Adam/BCE and ATM-TCR architecture defaults are retained. As requested, experiment
selection follows the host protocol: best validation AUROC, patience 4, maximum
100 epochs, and maximum validation MCC over 199 interior threshold candidates.
Threshold comparisons use `>=` consistently with exported predictions. Unlike
upstream main.py, the fixed test set is never used for epoch selection and the
best checkpoint is restored before testing. The last singleton training batch
is merged into its predecessor for BatchNorm, retaining all rows.

Outputs: `../baseline_runs/predictions/ATM-TCR/hitph0630_unseen_seed42/`:
only `predictions.csv` and distance-stratified `metrics.json`. Checkpoint
`best.pt` is under `../baseline_runs/checkpoints/ATM-TCR/hitph0630_unseen_seed42/`;
history and logs are under `../baseline_runs/logs/ATM-TCR/hitph0630_unseen_seed42/`.
This checkpoint is a newly trained ATM-TCR baseline, not the official model.

## Generic CLI

From this project's root, using the host Python:

```bash
python baseline.py plan --config ../baseline_runs/configs/ATM-TCR/immrep25_predict.json
python baseline.py run --config ../baseline_runs/configs/ATM-TCR/hitph0630_unseen_seed42_train_predict.json
```

`plan` checks inputs without running the model. `run` refuses to overwrite
finished predictions and automatically produces final metrics. Intermediate
requests and converted inputs are temporary and removed after execution.
