"""Finish the verified runtime installation, then execute the two real workflows."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

BASE=Path('/root/TMP_prediction/baseline_runs')
CONFIG=BASE/'configs/ATM-TCR'
DOWNLOAD=BASE/'downloads'
WHEEL=DOWNLOAD/'torch-1.10.0+cu113-cp38-cp38-linux_x86_64.whl'
PYTHON=BASE/'envs/atm-tcr/bin/python'
def status(stage):
    (CONFIG/'run_status.json').write_text(json.dumps(dict(stage=stage,pid=__import__('os').getpid()),indent=2)+'\n')

def main():
    status('waiting_cuda_download')
    EXPECTED='cccddc32b8941bd03ede29ff0a1cce2f2b51113a5ee23bb8b979316ac2114183'
    while not (DOWNLOAD/'torch-download.log').read_text().rstrip().endswith(str(WHEEL)):
        time.sleep(10)
    status('installing_runtime')
    print('CUDA wheel download complete; checking official SHA256',flush=True)
    h=hashlib.sha256()
    with WHEEL.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=EXPECTED:raise ValueError('Official torch wheel SHA256 mismatch')
    (CONFIG/'torch_artifact_verified.json').write_text(json.dumps(dict(path=str(WHEEL),sha256=h.hexdigest(),official_index='https://download.pytorch.org/whl/cu113/torch/'),indent=2)+'\n')
    subprocess.run([str(PYTHON),'-m','pip','install','--no-index','--no-build-isolation','--find-links',str(DOWNLOAD),str(WHEEL),'torchtext==0.11.0'],check=True)
    subprocess.run([str(PYTHON),'-m','pip','check'],check=True)
    for name,python in [('runtime',PYTHON),('host',BASE/'envs/baseline-host/bin/python')]:
        with (CONFIG/(name+'-freeze.txt')).open('w') as f:subprocess.run([str(python),'-m','pip','freeze'],stdout=f,check=True)
    probe="import json,torch;print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)));x=torch.randn(32,32,device='cuda');print(float((x@x).mean()))"
    env=dict(__import__('os').environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    result=subprocess.run([str(PYTHON),'-c',probe],capture_output=True,text=True,check=True,env=env)
    (CONFIG/'gpu_probe.txt').write_text(result.stdout+result.stderr);print(result.stdout,flush=True)
    status('predicting_immrep25')
    print('Running official IMMREP2025 checkpoint prediction',flush=True)
    subprocess.run([str(CONFIG/'run.sh'),'immrep25'],check=True,env=env)
    status('training_hitph0630_unseen_seed42')
    print('Starting fixed-workspace ATM-TCR training followed by test prediction',flush=True)
    subprocess.run([str(CONFIG/'run.sh'),'train'],check=True,env=env)
    status('complete')
    print('Both ATM-TCR workflows completed',flush=True)

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed')
        raise
