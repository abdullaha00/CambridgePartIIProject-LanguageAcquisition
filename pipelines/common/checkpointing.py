
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
from db.log_db import MetricRecord
from typing import Any

@dataclass(frozen=True)
class RunKey:
    model: str
    track: str
    subset: int | None
    train_with_dev: bool
    variant: str | None
    tag: str | None = None

def run_key_from_record(record: MetricRecord) -> RunKey:
    return RunKey(
        model=record.model,
        track=record.track,
        subset=record.subset,
        train_with_dev=record.train_with_dev,
        variant=record.variant,
        tag=record.tag,
    )

def save_name(key: RunKey, suffix: str = ".ckpt", auc: float | None = None, epoch: int | None = None) -> str:
    subset_str = "full" if key.subset is None else f"subset{key.subset}"
    twd_str = "twd1" if key.train_with_dev else "twd0"
    variant_str = f"_{key.variant}" if key.variant is not None else ""
    tag_str = f"_{key.tag}" if key.tag else ""
    epoch_str = f"_ep{epoch}" if epoch is not None else ""
    auc_str = f"_auc{auc:.4f}" if auc is not None else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{subset_str}_{twd_str}_{variant_str}{tag_str}{epoch_str}{auc_str}_{ts}{suffix}"

def ckpt_dir(key: RunKey) -> Path:
    p = Path("model_outputs") / key.model / key.track
    return p

def save_torch(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer | None,
    rec: MetricRecord,
    extra: dict[str, Any] | None = None
) -> Path:
    
    key = run_key_from_record(rec)
    dir_path = ckpt_dir(key)
    dir_path.mkdir(parents=True, exist_ok=True)
    save_path = dir_path / save_name(key, auc=rec.auc, epoch=rec.epochs)

    output = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict() if opt is not None else None,
        "epoch": rec.epochs,
        "metrics": {
            "auc": rec.auc,
            "accuracy": rec.acc,
            "f1": rec.f1
        },
        "rng_state": {
            "torch": torch.random.get_rng_state(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    if extra is not None:
        assert not set(extra).issubset(output.keys()), "Extra keys must not overwrite existing keys in output"
        output.update(extra)
    
    torch.save(output, save_path)
    return save_path


def load_torch_ckpt(path: str | Path) -> dict:
    return torch.load(path, weights_only=False)

def save_lgbm(model, rec: MetricRecord) -> Path:
    
    assert hasattr(model, "booster_"), "Model must be fitted before saving"
    
    key = run_key_from_record(rec)
    dir_path = ckpt_dir(key)
    dir_path.mkdir(parents=True, exist_ok=True)

    save_path = dir_path / save_name(key, ".txt", auc=rec.auc, epoch=rec.epochs)
    save_path_mets = dir_path / save_name(key, "_metrics.txt", auc=rec.auc, epoch=rec.epochs)

    with open(save_path_mets, "w") as f:
        f.write(f"tag: {rec.tag}\nauc: {rec.auc}\naccuracy: {rec.acc}\nf1: {rec.f1}\n")
    model.booster_.save_model(save_path)
    return save_path

