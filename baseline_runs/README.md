# Baseline runs

ATM-TCR, tcrLM and TULIP-TCR share `/root/miniconda3/envs/baseline-models`. The runner uses `/root/miniconda3/envs/baseline-host`. Installation wheels are cached under `/root/miniconda3/pkgs/baseline-wheels`.

Run one configuration:

```bash
./run.sh /root/TMP_prediction/baseline_runs/configs/tcrLM/immrep25.json
./run.sh /root/TMP_prediction/baseline_runs/configs/tcrLM/hitph0630_unseen_seed42.json
```

Results: `predictions/<model>/<dataset>/predictions.csv` and `metrics.json`.
Logs and epoch history: `logs/<model>/<dataset>/`.
Trained weights: `checkpoints/<model>/<dataset>/best.pt`.
Official weights: `checkpoints/<model>/pretrained/` (repository paths are compatibility symlinks).

The tcrLM retraining uses classifier batch 2048, learning rate 1e-4 and training-only feature standardization. Standardization is folded into exported linear weights, so inference uses the original model architecture.
See `../Agentic-TPH-training-code/EXTERNAL_BASELINES.md` for input and training rules.

TULIP-TCR configurations are in `configs/TULIP-TCR/`. It uses both TCR chains and MHC, positive-only generative training, mandatory CUDA and batch 1024. Official peptide likelihoods are intended for comparison within each peptide; use macro per-peptide metrics.
