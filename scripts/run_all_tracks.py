import argparse
import subprocess
from config.consts import TRACKS


p = argparse.ArgumentParser()
p.add_argument("model_name", 
               choices=["lr", "gbdt", "dkt", "bert_dkt", "lmkt", "qg"])
p.add_argument("-i","--item-level", choices=["token", "exercise"], default="token")
p.add_argument("-e","--epochs", type=int, default=5)
p.add_argument("--eval_every", type=int, default=1)
args, next_args = p.parse_known_args()

MODEL = args.model_name

for track in TRACKS:
    print(f"\n=== Running {MODEL} on track {track} with train_with_dev={True} and subset={None} ===")
    cmd = [
        "python", "-m", "main",
        MODEL,
        "--track", track,
        "--train-with-dev",
        "--epochs", str(args.epochs),
        "--eval-every", str(args.eval_every),
        "--item-level", args.item_level
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {MODEL} on track {track}: {e}")