import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate logit-space ensemble given 2 files.")
    p.add_argument("file_a", type=Path)
    p.add_argument("file_b", type=Path)
    p.add_argument("--alpha", type=float, default=0.5, help="Weight for file_b logits.")
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()

def read_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "label" in df.columns, f"Missing 'label' column in {path}"
    assert "prob" in df.columns, f"Missing 'prob' column in {path}"
    return df

def align(a: pd.DataFrame, b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Align by user_id and tok_id if possible
    if "user_id" in a.columns and "user_id" in b.columns and "tok_id" in a.columns and "tok_id" in b.columns:
        merged = a[["user_id", "tok_id", "label", "prob"]].merge(
            b[["user_id", "tok_id", "label", "prob"]],
            on=["user_id", "tok_id"],
            suffixes=("_a", "_b"),
            validate="one_to_one" # ensure unique
        )

        assert len(merged) == len(a) and len(merged) == len(b)

        y_a = merged["label_a"].to_numpy(dtype=np.int8)
        y_b = merged["label_b"].to_numpy(dtype=np.int8)
        p_a = merged["prob_a"].to_numpy(dtype=np.float64)
        p_b = merged["prob_b"].to_numpy(dtype=np.float64)
    else:
        assert len(a) == len(b), f"Length mismatch: file_a={len(a)} file_b={len(b)}"

        y_a = a["label"].to_numpy(dtype=np.int8)
        y_b = b["label"].to_numpy(dtype=np.int8)
        p_a = a["prob"].to_numpy(dtype=np.float64)
        p_b = b["prob"].to_numpy(dtype=np.float64)

    if not np.array_equal(y_a, y_b):
        raise ValueError("Prediction files disagree on labels.")
    
    return y_a, p_a, p_b


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def main() -> None:
    args = parse_args()

    assert 0 <= args.alpha <= 1, "--alpha must be between 0 and 1."

    y, p_a, p_b = align(read_preds(args.file_a), read_preds(args.file_b))
    logits = (1 - args.alpha) * logit(p_a) + args.alpha * logit(p_b)
    probs = sigmoid(logits)
    preds = probs >= args.threshold

    print(f"rows={len(y)} alpha={args.alpha:.4f} threshold={args.threshold:.4f}")
    print(f"auc={roc_auc_score(y, probs):.5f}")
    print(f"accuracy={accuracy_score(y, preds):.5f}")
    print(f"f1={f1_score(y, preds):.5f}")


if __name__ == "__main__":
    main()
