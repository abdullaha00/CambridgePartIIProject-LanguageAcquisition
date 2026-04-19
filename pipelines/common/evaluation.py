from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from pandas import col
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pipelines.common.checkpointing import ckpt_dir, run_key_from_record, save_name

import logging
logger = logging.getLogger(__name__)

def eval_directory(rec) -> Path:
    out_dir = ckpt_dir(run_key_from_record(rec)) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def save_binary_eval_predictions(
    rec,
    y_true,
    probs,
    extra_cols: dict[str, Iterable] | None = None,
    pred_labels: Iterable | None = None,
) -> Path:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)

    if y.shape[0] != p.shape[0]:
        raise ValueError("Prediction inputs must have matching lengths.")

    if pred_labels is None:
        pred = (p >= 0.5).astype(np.int8)
    else:
        pred = np.asarray(pred_labels, dtype=np.int8)
        if len(pred) != len(y):
            raise ValueError(f"Length mismatch: {len(pred)} != {len(y)} (pred_labels)")

    frame = pd.DataFrame(
        {
            "label": y,
            "prob": p,
            "pred": pred,
        }
    )

    # used for seen/unseen etc.
    if extra_cols:
        for col, values in extra_cols.items():
            arr = np.asarray(values)
            if len(arr) != len(frame):
                raise ValueError(f"Length mismatch: {len(arr)} != {len(frame)} (extra cols)")
            
            frame[col] = arr

    save_path = eval_directory(rec) / save_name(
        run_key_from_record(rec),
        suffix="eval_predictions.csv",
        auc=rec.auc,
        epoch=rec.epochs,
    )

    frame.to_csv(save_path, index=False)
    return save_path

def binary_metrics_score(y_true, probs, threshold: float = 0.5) -> dict[str, float]:

    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    pred = (p >= threshold).astype(np.int8)

    out = {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred),
    }
    out["auc"] = roc_auc_score(y, p) if np.unique(y).size >= 2 else float("nan")
    return out

def bootstrap(y_true, probs, n_bootstrap: int = 1000, ci: float = 0.95) -> dict[str, float]:
    y = np.asarray(y_true)
    p = np.asarray(probs)

    assert y.shape[0] == p.shape[0], "Length mismatch between y_true and probs"

    score = binary_metrics_score(y, p)

    n_samples = len(y)
    all_metrics: dict[str, list[float]] = {key: [] for key in score.keys()}

    for _ in range(n_bootstrap):
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        
        if len(np.unique(y[idxs])) < 2:
            logger.warning("Bootstrap sample with only one class present, skipping")
            continue

        metric = binary_metrics_score(y[idxs], p[idxs])
        for key, value in metric.items():
            all_metrics[key].append(value)

    out = score

    for key, vals in all_metrics.items():
        if len(vals) == 0:
            logger.warning(f"No valid bootstrap samples for metric {key}.")
            out[f"{key}_ci_lower"] = float("nan")
            out[f"{key}_ci_upper"] = float("nan")
        else:
            out[f"{key}_ci_lower"] = np.quantile(vals, (1 - ci) / 2)
            out[f"{key}_ci_upper"] = np.quantile(vals, (1 + ci) / 2)
    
    return out

def pair_bootstrap_auc(y_true, probs_a, probs_b, n_bootstrap: int = 1000, ci: float = 0.95) -> dict[str, float]:
    y = np.asarray(y_true)
    p_a = np.asarray(probs_a)
    p_b = np.asarray(probs_b)

    assert y.shape[0] == p_a.shape[0] == p_b.shape[0], "Length mismatch between inputs"

    base_auc_a = roc_auc_score(y, p_a) if np.unique(y).size >= 2 else float("nan")
    base_auc_b = roc_auc_score(y, p_b) if np.unique(y).size >= 2 else float("nan")
    base_diff = base_auc_a - base_auc_b

    diffs = []
    n_samples = len(y)

    for _ in range(n_bootstrap):
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y[idxs])) < 2:
            logger.warning("Bootstrap sample with only one class present, skipping")
            continue

        auc_a = roc_auc_score(y[idxs], p_a[idxs]) if np.unique(y[idxs]).size >= 2 else float("nan")
        auc_b = roc_auc_score(y[idxs], p_b[idxs]) if np.unique(y[idxs]).size >= 2 else float("nan")

        diffs.append(auc_a - auc_b)

    out = {
        "auc_a": base_auc_a,
        "auc_b": base_auc_b,
        "delta_auc": base_diff,
    }

    if len(diffs) == 0:
        logger.warning("No valid bootstrap samples for AUC difference.")
        out["delta_auc_ci_lower"] = float("nan")
        out["delta_auc_ci_upper"] = float("nan")
    else:
        out["delta_auc_ci_lower"] = np.quantile(diffs, (1 - ci) / 2)
        out["delta_auc_ci_upper"] = np.quantile(diffs, (1 + ci) / 2)

    return out
