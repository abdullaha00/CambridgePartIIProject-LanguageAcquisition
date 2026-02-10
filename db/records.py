
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class MetricRecord:
    model: str
    track: str
    subset: Optional[int]
    train_with_dev: bool
    variant: Optional[str]
    auc: float
    acc: float
    f1: float



