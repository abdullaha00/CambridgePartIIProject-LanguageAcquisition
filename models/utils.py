import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging

import torch

logger = logging.getLogger(__name__)


def compute_metrics(preds: np.ndarray, targets: np.ndarray):
    if len(np.unique(targets)) < 2:
        logger.warning("Warning: only one class present in y_true. AUC is not defined in this case.")
        return float('nan')  # AUC is not defined in this case

    auc = roc_auc_score(targets, preds)
    
    preds = (preds >= 0.5)
    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)

    return {
        "auc": auc,
        "accuracy": accuracy,
        "f1": f1,
    }

    