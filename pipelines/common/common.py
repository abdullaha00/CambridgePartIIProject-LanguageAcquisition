from db.log_db import MetricRecord
from dataclasses import dataclass

@dataclass
class PipelineResult:
    models: list[str]
    metrics: list[MetricRecord]
    
def mk_record(
    model_name: str,
    track: str,
    subset: int | None,
    train_with_dev: bool,
    metrics: dict[str, float],
    variant: str,
    epochs: int | None = None,
) -> MetricRecord:
    return MetricRecord(
        model=model_name,
        track=track,
        subset=subset,
        train_with_dev=train_with_dev,
        variant=variant,
        epochs=epochs,
        auc=float(metrics.get("auc", float("nan"))),
        acc=float(metrics.get("accuracy", float("nan"))),
        f1=float(metrics.get("f1", float("nan"))),
    )