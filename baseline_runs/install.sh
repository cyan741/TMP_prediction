#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
base=/root/TMP_prediction
runtime="/root/miniconda3/envs/baseline-models"
host="/root/miniconda3/envs/baseline-host"
repo="$base/baseline_runs/repos/ATM-TCR"
if [[ ! -x "$runtime/bin/python" ]]; then
  /root/miniconda3/bin/conda create -y -p "$runtime" python=3.8 pip
fi
wheel="/root/miniconda3/pkgs/baseline-wheels/torch-1.10.0+cu113-cp38-cp38-linux_x86_64.whl"
if "$runtime/bin/python" -c "import torch" 2>/dev/null; then
  :
elif [[ -f "$wheel" ]]; then
  printf '%s  %s\n' cccddc32b8941bd03ede29ff0a1cce2f2b51113a5ee23bb8b979316ac2114183 "$wheel" | sha256sum -c -
  "$runtime/bin/python" -m pip install "$wheel"
else
  "$runtime/bin/python" -m pip install "https://download-r2.pytorch.org/whl/cu113/torch-1.10.0%2Bcu113-cp38-cp38-linux_x86_64.whl#sha256=cccddc32b8941bd03ede29ff0a1cce2f2b51113a5ee23bb8b979316ac2114183"
fi
"$runtime/bin/python" -m pip install --find-links "/root/miniconda3/pkgs/baseline-wheels" -r "$repo/requirements.txt" pandas==1.3.5 rapidfuzz==3.9.7 einops==0.6.1 transformers==4.32.1
if [[ ! -x "$host/bin/python" ]]; then
  /root/miniconda3/bin/python -m venv --system-site-packages "$host"
fi
"$host/bin/python" -m pip install pandas==1.5.3 scikit-learn==1.3.0 rapidfuzz==3.9.7
"$runtime/bin/python" -m pip check
