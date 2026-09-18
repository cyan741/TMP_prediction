# External baselines

`baseline.py` connects ATM-TCR, tcrLM and TULIP-TCR to the existing data splits. It projects peptide and beta CDR3, imports the original model definitions, scores every test row, and preserves all original columns and their order.

```bash
/root/TMP_prediction/baseline_runs/run.sh /root/TMP_prediction/baseline_runs/configs/tcrLM/immrep25.json
/root/TMP_prediction/baseline_runs/run.sh /root/TMP_prediction/baseline_runs/configs/tcrLM/hitph0630_unseen_seed42.json
/root/TMP_prediction/baseline_runs/configs/ATM-TCR/run.sh immrep25
/root/TMP_prediction/baseline_runs/configs/ATM-TCR/run.sh train
```

Existing results are protected against accidental overwrite. `baseline.py plan --config ...` checks inputs without running the model.

## Files and environment

- `/root/miniconda3/envs/baseline-models`: shared Python 3.8, PyTorch 1.10 CUDA 11.3, NumPy, pandas, scikit-learn and einops. All three models use this environment; TULIP requires transformers 4.32.1.
- `/root/miniconda3/envs/baseline-host`: lightweight Python 3.10 environment for the runner and benchmark metrics; no second PyTorch installation.
- `/root/miniconda3/pkgs/baseline-wheels`: reusable installation cache.
- `baseline_runs/predictions/<model>/<dataset>/`: only `predictions.csv` and `metrics.json`.
- `baseline_runs/logs/<model>/<dataset>/`: execution logs and epoch history.
- `baseline_runs/checkpoints/<model>/<dataset>/best.pt`: selected trained weights.
- `baseline_runs/checkpoints/<model>/pretrained/`: official weights; original repository paths are symlinks.
- Converted inputs and worker requests live in a temporary directory and are removed when the run finishes. No input/repository fingerprints are generated.

## Protocol

IMMREP25 directly loads the official finetuned checkpoint, with threshold 0.5. The fixed benchmark uses `workspaces/hitph0630_unseen_seed42/hitph`, without redrawing the split. Training, validation and test peptides must be disjoint. Test labels are used only for final metrics.

ATM-TCR and tcrLM use peptide and beta CDR3, ignoring MHC and alpha. Duplicate projected inputs and contradictory evaluation labels remain in the test set. Final output adds `score`, `predicted_label` and one frozen `threshold`.

ATM-TCR uses the original attention network and tokenizer/padding. Scratch training uses Adam, fresh screened 1:1 negatives each epoch, and validation AUROC selection with patience 4. Unsupported residue letters follow the upstream tokenizer's `*` mapping.

tcrLM uses two 48-layer FLASH encoders (512 dimensions), concatenates their 34-token representations and trains the two-output linear classifier. Both encoders are initialized from the official pretrained weights and frozen, matching upstream fine-tuning. Their deterministic representations are cached in memory once. Training uses fixed screened 1:1 negatives, Adam at 1e-4, classifier batch size 2048, encoder/prediction batch size 64, up to 35 epochs and patience 8. Training-only occurrence-weighted feature means and standard deviations condition the linear head. The new head starts with zero weights/bias. Export folds standardization into the original classifier weights; prediction requires no separate normalizer. CUDA matmul uses full FP32 precision (TF32 disabled). One training beta sequence containing `O` is excluded because the official vocabulary contains only 20 standard residues; all validation/test rows are kept.

Checkpoint selection uses validation AUROC. Classification thresholds maximize validation MCC and are saved in the checkpoint before test prediction. Fixed-test metrics also report performance by nearest training-peptide edit distance.

`tcrLM` defaults to `mask_width=20` to preserve the original checkpoint's executable masking behavior despite 34-token padding. `mask_width=34` is an optional corrected variant and changes model behavior.

Official tcrLM weights are available in the repository history before their removal from the current branch: https://github.com/hliulab/tcrLM/tree/850135f05c9f001b4a07e8228028ba0b19aec8e7/pretrained_model . Physical weights are under `baseline_runs/checkpoints/tcrLM/pretrained/`; repository paths are compatibility symlinks.

## TULIP-TCR

```bash
/root/TMP_prediction/baseline_runs/run.sh /root/TMP_prediction/baseline_runs/configs/TULIP-TCR/immrep25.json
/root/TMP_prediction/baseline_runs/run.sh /root/TMP_prediction/baseline_runs/configs/TULIP-TCR/hitph0630_unseen_seed42.json
```

The original `src/multiTrans.py` supplies three BERT encoders and three causal cross-attention decoders, with MHC embeddings and learned missing-chain representations. Each branch has two layers and hidden size 128. Training reconstructs observed alpha, beta and peptide sequences and adds the original 15% masked-language objective. Binding labels are not part of this generative objective. Only positive training rows are used, without fabricated negatives.

IMMREP25 loads the official checkpoint. The fixed unseen split trains from scratch with the official architecture, avoiding unknown overlap with official pretraining data. All 106,918 positives are retained: O has a native token; two alpha chains containing #/? use upstream UNK encoding. MHC aliases are normalized to the official vocabulary; unknown alleles use its missing token.

CUDA is mandatory. Training uses AdamW at 1e-4, batch 1024, mixed precision, gradient clipping 1, up to 30 epochs and patience 4. All token arrays reside on the GPU. Loss is the original summed LM + MLM objective divided by the number of predicted tokens. Checkpoints maximize validation macro per-peptide AUROC; the classification threshold maximizes validation MCC and is frozen before testing. `train_gpu.csv` and `predict_gpu.csv` record nvidia-smi utilization and memory every five seconds in the log directory.

Prediction follows the upstream summed peptide conditional log-likelihood (including EOS), exported as its exponential to preserve ranking while satisfying the common score contract. This is a sequence probability, not a calibrated binding probability. Scores are primarily comparable within the same peptide; macro per-peptide AUROC/AUPRC are primary, pooled metrics supplementary. The official checkpoint has no learned binding threshold, so direct IMMREP25 prediction uses the default 0.5; its binary F1/MCC are not informative. Test labels are used only for metrics.
