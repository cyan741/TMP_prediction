"""Install the shared environment and run both ATM-TCR configurations."""
import subprocess
from pathlib import Path

config=Path(__file__).resolve().parent
if __name__=='__main__':
    subprocess.run(['bash',str(config/'install.sh')],check=True)
    for task in ['immrep25','train']:
        subprocess.run(['bash',str(config/'run.sh'),task],check=True)
