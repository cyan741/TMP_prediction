# Agentic-TPH training and inference

This package prepares datasets, trains T-cell receptor (TCR) interaction models, and predicts scores for new TCR–peptide–MHC records. It contains source code, example configurations and dependency specifications. Datasets, pretrained weights, trained checkpoints and credentials must be supplied separately.

A record contains a peptide, a major histocompatibility complex (MHC) representation, and an ordered pair of TCR alpha/beta chains. HLA denotes human MHC. CDR3 is the complementarity-determining region 3 of a TCR chain. A *pseudo-sequence* is a fixed selection of MHC residues, not a complete MHC protein. A *workspace* is the directory written by `prepare`, containing fixed data splits and the model input settings.

Run all commands below from this package's root directory. Relative paths, including paths inside configuration files, are resolved from the current working directory. Source provenance is recorded in `SOURCE.txt`.

## Supported models

| Entry point / configuration | Model inputs |
|---|---|
| `train.py`, `configs/levelIII.json` | Human HLA-I: paired CDR3s, peptide and a 34-residue MHC pseudo-sequence |
| `train.py`, `configs/levelIV.json` | Human HLA-I: paired TCR variable domains, peptide and a 34-residue MHC pseudo-sequence; original CDR3s remain the identities used for splitting |
| `train.py`, `configs/pan_esm2.json` | Protein language model (PLM) for MHC-I/II, with explicit MHC class and a supplied 34-residue pseudo-sequence |
| `pan_cnn.py`, `configs/pan_cnn.json` | Convolutional neural network (CNN) using complete MHC chains and paired CDR3s across human/nonhuman MHC-I/II domains |

The CNN supports four single-domain models, six two-domain combinations and one four-domain model. Dataset growth experiments train each dataset version from the base model; this package does not implement warm-start updates or replay. The pan-MHC PLM uses this package's standalone training loop. Numerical equivalence with the historical Hi-TpH injection-based entry point has not been established; report newly trained results with their actual configuration.

## Installation and local model assets

Use Linux and Python 3.11. The reference runtime uses PyTorch 2.5.1+cu124, transformers 4.36.2, NumPy 1.24.3, pandas 2.3.3 and scikit-learn 1.3.0. The dependency files reflect that runtime; a fresh installation has not been verified.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-torch.txt
python -m pip install -r requirements.txt
```

The CUDA dependency example requires a compatible driver. The CNN is trained from scratch and needs no pretrained assets. PLMs require local model weights, configuration and tokenizer files. TAPE additionally requires `tape-proteins`.

For example, place an already downloaded ESM2-150M model and tokenizer in `assets/models/esm2-150M`, then register the directory:

```bash
python configure_assets.py --asset-root ./assets \
  --model-id esm2-150M --relative-root models/esm2-150M
```

This writes the local asset index, `assets/manifest.json`. It does not download or copy weights. Training and inference load assets locally.

## Dataset splitting and negative examples

All preparation entry points default to `validation_grouping=pair_stratified`, targeting 10% positive validation records. Within each peptide, they select paired-TCR/MHC groups for validation while retaining at least one training group wherever the reference and compared datasets permit. Peptides with only one group remain in training. These constraints can reduce the achieved validation fraction; `workspace_report.json` records the actual counts.

A split group is identified by peptide, MHC identity and ordered alpha/beta CDR3s. For grouping only, each CDR3 has leading runs of C and trailing runs of F/W removed to a stable form. This conservatively keeps possible terminal-notation variants together. Model inputs retain the full supplied sequences, with whitespace trimmed and letters converted to uppercase. Level IV uses original CDR3 identities for splitting, not reconstructed variable domains.

| Grouping option | Validation behavior |
|---|---|
| `pair_stratified` | Preserve training coverage per peptide while keeping paired-TCR/MHC groups separate |
| `peptide` | Hold out whole peptides; training and validation do not share peptides |
| `cluster` | Hold out single-linkage peptide clusters formed using edit distance at most 2 |

`test_grouping` independently controls which training records are excluded by the supplied fixed test file. Use `peptide` for unseen-peptide evaluation or `cluster` for unseen-cluster evaluation. Preparation never redraws the test set. Every mode also checks that paired-TCR groups do not cross train/validation/test boundaries. A comparison uses one fixed validation set across datasets and training seeds.

Validation and test records are deduplicated by group; contradictory labels within a group raise an error. Splitting preserves eligible training rows. Source converters may apply their own documented capability filters and deduplication before splitting.

Negative examples are randomly mismatched training/evaluation records, not experimentally established nonbinders. The exclusion table covers known positives and held-out pairs. It excludes a peptide/TCR group across MHC alleles; CNN TCR donors additionally stay within the same domain. Generated negative labels must not be treated as experimental observations.

## Training input files

### PLM inputs

Canonical CSV columns are:

```text
pep,hla,hla.allele,ab,label
```

`ab` contains alpha/beta chains separated by `/`, in that order. `hla` contains the 34-residue pseudo-sequence, and `hla.allele` identifies MHC for splitting. Level IV additionally requires `cdr3_ab` or `ab_cdr3`, containing original paired CDR3s; `ab` itself contains variable domains. Extra columns are retained. Training sources must contain explicit positives; the fixed test file must contain 0/1 labels.

The human configurations also accept TPH_250630 and TPH_260630 `TPH-HiTPHplus-level-III.csv` exports. An unlabeled source must explicitly set `positive_source=true` and must actually be a positive-observation export. Level III reads `alpha.cdr3/beta.cdr3`; Level IV reads `alpha.vseq.reconstructed/beta.vseq.reconstructed` and retains original CDR3s for splitting. Peptides use a recorded `minimal_epitope` when available, otherwise `antigen.epitope`; no minimal epitope is inferred.

To convert historical Hi-TpH positive records:

```bash
python prepare_inputs.py hitph --root /path/to/Hi-TpH \
  --level III --output ./data/hitph_levelIII_positive.csv
```

Use `--level IV` and a different output path for variable-domain inputs. The converter reads the source without modifying it.

Pan-MHC PLM inputs use the canonical columns and an additional `mhc_class` column containing `I` or `II`. For a file containing only one class, the input specification may provide `mhc_class` instead. Supply resolved pseudo-sequences; full MHC chains cannot be placed in the `hla` field.

### CNN inputs

Both positive source and fixed test CSVs require:

```text
peptide,allele_key,tcr_alpha_cdr3,tcr_beta_cdr3,mhc_alpha_seq,mhc_beta_seq,species,mhc_class,domain,label
```

`species` is `human`, `mouse` or `other`; `mhc_class` is `I` or `II`; `domain` is `human_I`, `human_II`, `nonhuman_I` or `nonhuman_II`. MHC-II requires both MHC chains; MHC-I leaves `mhc_beta_seq` empty. `allele_key` must identify the full allele/chain combination. Split identities also include species and MHC class to distinguish names across species.

You may use an existing resolved `positive_records.csv`, or convert the TPH source using local reference sequences:

```bash
python prepare_inputs.py pan-source \
  --data-root ./data/TPH_260630 \
  --hla-fasta ./references/hla_protein.fasta \
  --mouse-fasta ./references/mouse_mhc.fasta \
  --ipd-mhc-fasta ./references/MHC_prot.fasta \
  --output ./data/pan_source
```

The output contains `positive_records.csv`, `exclusions.csv` and resolved MHC sequence records. Reference FASTAs are not included. Nonhuman MHC-II coverage depends on what the source parser can resolve from the supplied references.

## Prepare and train

Edit the example JSON to provide input paths, the fixed test file, output location and reference corpus. `corpora` defines the datasets to compare; `reference` names the dataset used to select validation groups. For a single dataset, keep one `corpora` entry and use its name as `reference`. Dataset aliases such as `hitph`, `tph25` and `tph26` only identify entries and generated subdirectories; supply your own input files. Source ablations require explicit source combinations in the configuration.

```bash
python train.py prepare --config configs/levelIII.json
python train.py train --workspace ./workspaces/levelIII/tph26 \
  --asset-root ./assets --seed 42 --output ./outputs/levelIII_tph26_s42
```

For Level IV, use `configs/levelIV.json` and `./workspaces/levelIV/tph26`. For pan-MHC PLM, use `configs/pan_esm2.json` and `./workspaces/pan_esm2/pan`.

Two-GPU PLM training:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 train.py train \
  --workspace ./workspaces/levelIII/tph26 --asset-root ./assets \
  --seed 42 --output ./outputs/levelIII_tph26_s42
```

The default Level III/IV settings give 96 positive examples per optimizer step on two GPUs. On one GPU, this becomes 48; to keep 96, set `--gradient-accumulation-steps 4` for Level III or `12` for Level IV. Dynamic negatives double the records processed. Record hardware and process count with results.

CNN training:

```bash
python pan_cnn.py prepare --config configs/pan_cnn.json
python pan_cnn.py train --workspace ./workspaces/pan_cnn \
  --runs all_four --output ./outputs/pan_cnn_s42
```

`--runs all_four` trains the four-domain model. Omit it to train all 11 combinations, or use `--runs human_I human_I+human_II` for selected models. Each model evaluates every test domain listed in the workspace, regardless of the training combination. Each workspace domain requires both positive and negative validation/test examples.

### Preparation parameters

Both `train.py prepare` and `pan_cnn.py prepare` require `--config`. Explicit options below override the matching JSON fields; omitted options leave the configuration unchanged. `prepare.py --config ...` is equivalent to `train.py prepare`.

| Option | JSON field | Behavior when absent from JSON |
|---|---|---|
| `--output` | `output` | Must be provided by configuration or command line |
| `--validation-grouping` | `validation_grouping` | `pair_stratified` |
| `--test-grouping` | `test_grouping` | Task-dependent for PLM; `pair_stratified` for CNN; examples set it explicitly |
| `--validation-fraction` | `validation_fraction` | `0.1` |
| `--split-seed` | `split_seed` | `42` |
| `--validation-negative-seed` | `validation_negative_seed` | `1042` |
| `--training-negative-seed` (CNN only) | `training_negative_seed` | `42` |

For example, change validation grouping and write a new workspace:

```bash
python train.py prepare --config configs/levelIII.json \
  --validation-grouping peptide --output ./workspaces/levelIII-peptide
```

Historical comparisons used split/validation-negative seeds 42/1042 with the Hi-TpH reference, and 43/1043 for the source-combination experiments whose configured reference was named `DLE`. These are experiment settings, not requirements for a new dataset. Training seeds used for repeated experiments were 42, 1337 and 2024.

### PLM training parameters: `python train.py train`

| Option | Required / default | Meaning |
|---|---|---|
| `--workspace` | Required | Prepared single-dataset directory containing `workspace_report.json` |
| `--asset-root` | Required | Local model asset root containing `manifest.json` |
| `--output` | Required | Directory for checkpoints, thresholds, epoch records and `training_result.json` |
| `--model-id` | `esm2-150M` | Model name registered in the asset manifest |
| `--epochs` | `100` | Maximum training epochs |
| `--learning-rate` | `8e-5` | Adam learning rate |
| `--batch-size` | III=24; IV=8; pan-ESM2=32 | Positive examples per GPU microbatch, before adding negatives |
| `--gradient-accumulation-steps` | III=2; IV=6; pan-ESM2=1 | Microbatches per optimizer step |
| `--patience` | III/IV=4; pan-ESM2=5 | Epochs without validation AUROC improvement before stopping |
| `--seed` | `42` | Initialization and runtime seed |
| `--negative-seed` | Follows `--seed` | Dynamic negative-sampling seed |
| `--device` | `cuda` | Device, such as `cuda:0` or `cpu` |
| `--precision` | `bf16` | `bf16`, `fp32` or `tf32` |
| `--num-workers` | `0` | DataLoader subprocesses |

Sequence widths come from the workspace. `training_result.json` identifies `best_checkpoint`; keep this checkpoint and its adjacent `.selected_threshold.json` file together.

### CNN training parameters: `python pan_cnn.py train`

| Option | Required / default | Meaning |
|---|---|---|
| `--workspace` | Required | Prepared directory with train, validation and domain-specific test CSVs |
| `--output` | Required | New directory; models are saved as `models/<combination>/checkpoint.pt`, with `summary.json` at the root |
| `--runs` | `all` | One or more single/pair domain combinations, `all_four`, or `all` |
| `--epochs` | `8` | Maximum training epochs |
| `--patience` | `2` | Epochs without mean validation AUPRC improvement across domains |
| `--batch-size` | `512` | Total positive and negative records per batch |
| `--learning-rate` | `8e-4` | AdamW learning rate |
| `--seed` | `42` | Model initialization and per-epoch balanced sampling seed |
| `--device` | `cuda:0` | Compute device |

CNN negatives are generated once during preparation using `training_negative_seed`. Changing the training seed does not regenerate negatives or redraw validation data. Each epoch balances domain-by-label cells by downsampling to the smallest cell.

### Model and optimization settings

| Setting | Human Level III | Human Level IV | Pan-MHC PLM | Pan-MHC CNN |
|---|---|---|---|---|
| Default model | ESM2-150M + 3-layer MLP | Same | Same | Chain-sequence CNN |
| Weight updates | Full fine-tuning | Full fine-tuning | Full fine-tuning | From scratch |
| Optimizer | Adam | Adam | Adam | AdamW; weight decay 1e-4 |
| Precision | bf16 | bf16 | bf16 | fp32 |
| Peptide length | 8–13 | 8–13 | I: 8–13; II: at most 30 | At most 30 |
| Per-chain TCR length | CDR3 at most 28 | Variable domain at most 127 | CDR3 at most 19 | CDR3 at most 40 |
| MHC representation | 34-residue pseudo-sequence | Same | Same | Fixed alpha 400 + beta 300 slots |
| Training negatives | Dynamic 1:1 | Dynamic 1:1 | Dynamic 1:1 | Fixed 1:1, within domain |
| Validation negatives | Fixed 1:1 when generated | Same | Same | Same |
| Checkpoint selection | Validation AUROC | Same | Same | Mean validation AUPRC across domains |
| Decision threshold | Maximum validation MCC | Same | Same | Maximum validation F1 |

The PLM head uses 256→64→2 dimensions. The CNN uses embedding dimension 24, sequence dimension 48, hidden dimension 192, dropout 0.15 and gradient clipping at 5.0. Historical human Level III/IV experiments used two A800 GPUs. PLM and CNN input representations, optimizers and selection metrics remain distinct even when split rules are shared. Unsupported training rows are counted and excluded; unsupported fixed test rows cause an error rather than silent truncation or removal.

## Inference without labels

Inference requires a checkpoint produced by a supported training entry point, a new input CSV and a new output path. PLMs additionally require local model/tokenizer assets. No training workspace or `label` column is required. An existing `label` column is preserved but never used for scoring or threshold selection. Load only trusted PyTorch checkpoints.

### PLM prediction

Required columns are `pep,hla,ab`. `ab` is alpha/beta CDR3 for Level III and pan-MHC PLM, or alpha/beta variable domains for Level IV. Prediction does not require allele or split-identity columns. Peptide and TCR sequences use standard amino-acid letters; MHC pseudo-sequences also allow `X` for unknown residue slots. Overlength inputs are rejected. Use sequences, species and MHC classes within the training scope of the selected model.

```bash
python train.py inference \
  --input ./data/new_pairs.csv \
  --checkpoint ./outputs/levelIII_tph26_s42/esm2-150M_fold0_epoch1.pt \
  --asset-root ./assets --output ./predictions/levelIII.csv
```

The epoch-1 filename illustrates the naming convention; use the `best_checkpoint` reported by your training run. Model type, sequence widths and architecture settings are restored from the checkpoint.

### CNN prediction

Required columns are:

```text
peptide,mhc_alpha_seq,mhc_beta_seq,tcr_alpha_cdr3,tcr_beta_cdr3,species,mhc_class
```

`species` is `human/mouse/other`; `mhc_class` is `I/II`. Keep the beta MHC field empty for class I and supply both MHC chains for class II. Neither `allele_key`, `domain` nor `label` is required. Sequence length limits are listed above.

```bash
python pan_cnn.py inference \
  --input ./data/new_pan_pairs.csv \
  --checkpoint ./outputs/pan_cnn_s42/models/all_four/checkpoint.pt \
  --output ./predictions/pan_cnn.csv
```

### Inference parameters and outputs

| Option | Required / default | Meaning |
|---|---|---|
| `--input` | Required | Input CSV |
| `--checkpoint` | Required | Trained model checkpoint |
| `--output` | Required | New prediction CSV; an existing file is not overwritten |
| `--asset-root` | Required for PLM; unused by CNN | Local base model and tokenizer assets |
| `--batch-size` | PLM=32; CNN=512 | Records per prediction batch |
| `--device` | `cuda` | Compute device, such as `cuda:0` or `cpu` |
| `--precision` | PLM=`bf16`; CNN uses fp32 | PLM supports `bf16/fp32/tf32` |
| `--threshold` | Saved validation threshold | Optional explicit classification threshold in [0, 1] |

Outputs preserve original fields and row order, adding `score`, `predicted_label` and `threshold`. A score at least the threshold receives predicted label 1. Inputs must not already contain these output column names. Scores are model outputs, not experimentally verified binding probabilities.

PLMs read the adjacent `.selected_threshold.json`; CNN checkpoints store the threshold internally. If no saved threshold is available, supply `--threshold` explicitly. The prediction set never determines the threshold.

## Labeled evaluation and code layout

PLM evaluation requires `--workspace`, `--asset-root` and `--checkpoint`:

```bash
python train.py eval --workspace ./workspaces/levelIII/tph26 \
  --asset-root ./assets \
  --checkpoint ./outputs/levelIII_tph26_s42/esm2-150M_fold0_epoch1.pt
```

It evaluates the fixed labeled test set using the saved validation threshold. Model, batch, seed, precision, device and DataLoader options are shared with the PLM training parser. CNN training automatically reloads its best model and evaluates the workspace test domains. Unlabeled inference only produces predictions.

Root scripts define command-line interfaces; `inference.py` implements batch prediction; `core_engine/trainer/` contains splitting, encoding, models and training loops. Each entry point supports `--help`. Focused regression tests cover split leakage, conflicting labels, negative-example exclusions and prediction row correspondence:

```bash
python -m unittest discover -s tests -v
```

## External baseline models

`baseline.py` adds isolated external-repository adapters, currently tcrLM, for direct checkpoint inference and training followed by inference. Planning/data preparation do not launch models; execution requires the explicit `run` command. See [EXTERNAL_BASELINES.md](EXTERNAL_BASELINES.md) for source review, input contracts, configuration examples, prepared requests, compatibility differences and adapter extension instructions.

```bash
python baseline.py plan --config configs/baselines/tcrlm_predict.json
python baseline.py plan --config configs/baselines/tcrlm_train_predict.json
```
