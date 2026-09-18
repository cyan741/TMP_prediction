#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /root/TMP_prediction/Agentic-TPH-training-code
exec /root/miniconda3/envs/baseline-host/bin/python baseline.py run --config "$1"
