
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import torch
from db.log_db import MetricRecord


@dataclass(frozen=True)
class RunKey:
    model: str
    track: str
    subset: int | None
    train_with_dev: bool
    variant: str | None

def run_key_from_record(record: MetricRecord) -> RunKey:
    return RunKey(
        model=record.model,
        track=record.track,
        subset=record.subset,
        train_with_dev=record.train_with_dev,
        variant=record.variant
    )

def save_name(key: RunKey, suffix: str = ".ckpt") -> str:
    subset_str = "full" if key.subset is None else f"subset{key.subset}"
    twd_s = "1" if key.train_with_dev else "0"
    variant_str = key.variant if key.variant is not None else "default"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{key.model}_{key.track}_{subset_str}_{twd_s}_{variant_str}_{ts}{suffix}"

def ckpt_dir(key: RunKey) -> Path:
    subset_str = "full" if key.subset is None else f"subset{key.subset}"
    p = Path("model_outputs") / key.model / key.track
    return p

def save_torch(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer | None,
    rec: MetricRecord,
) -> None:
    
    key = run_key_from_record(rec)
    dir_path = ckpt_dir(key)
    dir_path.mkdir(parents=True, exist_ok=True)
    save_path = dir_path / save_name(key)

    output = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict() if opt is not None else None,
        "epoch": rec.epochs,
        "metrics": {
            "auc": rec.auc,
            "accuracy": rec.acc,
            "f1": rec.f1
        }
    }
    
    torch.save(output, save_path)

def save_lgbm(model, rec: MetricRecord) -> None:
    
    assert hasattr(model, "booster_"), "Model must be fitted before saving"
    
    key = run_key_from_record(rec)
    dir_path = ckpt_dir(key)
    dir_path.mkdir(parents=True, exist_ok=True)
    save_path = dir_path / save_name(key, ".txt")
    save_path_mets = dir_path / save_name(key, "_metrics.txt")

    with open(save_path_mets, "w") as f:
        f.write(f"auc: {rec.auc}\naccuracy: {rec.acc}\nf1: {rec.f1}\n")
    model.booster_.save_model(save_path)

