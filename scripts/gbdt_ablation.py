import argparse
import subprocess
from config.consts import TRACKS
from models.gbdt.features.lesions import EVAL_LESIONS

SUBSET = None

for ablation in EVAL_LESIONS:
    print(f"\n=== Running gbdt ensemble on ablation {ablation} ===")
    cmd = [
        "python", "-m", "main",
        "gbdt",
        "--track", "all",
        "--lesion", ablation,
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running gbdt on ablation {ablation}: {e}")