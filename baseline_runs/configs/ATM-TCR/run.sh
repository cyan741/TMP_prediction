#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
base=/root/TMP_prediction
host="/root/miniconda3/envs/baseline-host/bin/python"
configs="$base/baseline_runs/configs/ATM-TCR"
cd "$base/Agentic-TPH-training-code"
case "${1:-}" in
  immrep25)
    "$host" baseline.py run --config "$configs/immrep25_predict.json"
    ;;
  train)
    "$host" baseline.py run --config "$configs/hitph0630_unseen_seed42_train_predict.json"
    ;;
  *) echo 'Usage: run.sh {immrep25|train}' >&2; exit 2 ;;
esac
