from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PAD_ID = 0
UNK_ID = 1
NA_VALUE = "<NA>"
TOKEN_COL = "tok"

def string_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(NA_VALUE)

def build_vocab(series: pd.Series) -> dict[str, int]:
    tokens = string_series(series).unique()
    token_vocab = {tok: idx + 2 for idx, tok in enumerate(tokens)} # 2 reserved for PAD and UNK
    return token_vocab

@dataclass(frozen=True)
class FAVocabs:
    token_vocab: dict[str, int]
    feature_vocabs: dict[str, dict[str, int]]
    feature_cols: tuple[str, ...] 
    numeric_feature_cols: tuple[str, ...]

    @property
    def vocab_size(self) -> int:
        return len(self.token_vocab) + 2
    
    @property
    def feature_vocab_sizes(self) -> dict[str, int]:
        return {ft: len(vocab) + 2 for ft, vocab in self.feature_vocabs.items()}
    
@dataclass(frozen=True)
class FASequenceInstance:
    user_id: object
    ex_key: object
    token_ids: np.ndarray
    feature_ids: dict[str, np.ndarray]
    numeric_features: dict[str, np.ndarray]
    labels: np.ndarray
    tok_ids: np.ndarray
    target_pos: np.ndarray

def build_encoded_sequences(
    df: pd.DataFrame,
    vocabs: FAVocabs
) -> list[FASequenceInstance]:

    sequences: list[FASequenceInstance] = []

    for ex_key, group in df.groupby("ex_key", sort=False):
        token_ids = string_series(group[TOKEN_COL]).map(vocabs.token_vocab).fillna(UNK_ID).astype(np.int64).to_numpy()
        feature_ids = {
            feat: string_series(group[feat]).map(feat_vocab).fillna(UNK_ID).astype(np.int64).to_numpy()
            for feat, feat_vocab in vocabs.feature_vocabs.items()
        }
        numeric_features = {
            col: group[col].fillna(0).astype(np.float32).to_numpy()
            for col in vocabs.numeric_feature_cols
        }

        sequences.append(
            FASequenceInstance(
                user_id=group["user_id"].iloc[0],
                ex_key=ex_key,
                token_ids=token_ids,
                feature_ids=feature_ids,
                numeric_features=numeric_features,
                labels=group["label"].astype(np.int64).to_numpy(),
                tok_ids=group["tok_id"].to_numpy(),
                target_pos=np.arange(len(group), dtype=np.int64),
            )
        )

    return sequences

# ===== DATALOADERS

def fab_collate_fn(batch: list[FASequenceInstance]) -> dict[str, torch.Tensor]:
    B = len(batch)
    T_max = max(len(seq.token_ids) for seq in batch)

    feat_cols = batch[0].feature_ids.keys()
    numeric_cols = list(batch[0].numeric_features.keys())
    numeric_dim = len(numeric_cols)

    mask = torch.zeros((B, T_max), dtype=torch.bool)
    numeric_feats = torch.zeros((B, T_max, numeric_dim), dtype=torch.float)
    feature_ids = {feat: torch.full((B, T_max), PAD_ID, dtype=torch.long) for feat in feat_cols}
    token_ids = torch.full((B, T_max), PAD_ID, dtype=torch.long)
    labels = torch.zeros((B, T_max), dtype=torch.long)
    
    # metadata for saving preds 
    user_ids, ex_keys, tok_ids, target_pos = [], [], [], []

    for i, seq in enumerate(batch):
        T = len(seq.token_ids)
        token_ids[i, :T] = torch.from_numpy(seq.token_ids)
        labels[i, :T] = torch.from_numpy(seq.labels)
        mask[i, :T] = 1
        for j, col in enumerate(numeric_cols):
            numeric_feats[i, :T, j] = torch.from_numpy(seq.numeric_features[col])

        for feat in feat_cols:
            feature_ids[feat][i, :T] = torch.from_numpy(seq.feature_ids[feat])

        user_ids.append(seq.user_id)
        ex_keys.append(seq.ex_key)
        tok_ids.append(seq.tok_ids)
        target_pos.append(seq.target_pos)

    return {
        "token_ids": token_ids,
        "feature_ids": feature_ids,
        "numeric_features": numeric_feats,
        "labels": labels,
        "mask": mask,
        "user_ids": user_ids,
        "ex_keys": ex_keys,
        "tok_ids": tok_ids,
        "target_pos": target_pos
    }
