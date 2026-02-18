import argparse
import subprocess
from config.consts import TRACKS
from models.gbdt.features.lesions import LESIONS

SUBSET = None

for ablation in LESIONS.keys():
    print(f"\n=== Running gbdt ensemble on ablation {ablation} ===")
    cmd = [
        "python", "-m", "main",
        "gbdt",
        "--track", "all",   
        #"--train-with-dev",
        #"--subset", str(SUBSET),
        "--lesion", ablation,
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running gbdt on ablation {ablation}: {e}")