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
    tokens = string_series(series).drop_duplicates()
    return {str(tok): idx + 2 for idx, tok in enumerate(tokens)} # 2 reserved for PAD and UNK

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

class FASequenceDataset(Dataset):
    def __init__(
        self,
        user_ids: np.ndarray,
        ex_keys: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        token_ids: np.ndarray,
        feature_ids: dict[str, np.ndarray],
        numeric_features: dict[str, np.ndarray],
        labels: np.ndarray,
        tok_ids: np.ndarray,
    ):
        self.user_ids = user_ids
        self.ex_keys = ex_keys
        self.starts = starts
        self.ends = ends
        self.token_ids = token_ids
        self.feature_ids = feature_ids
        self.numeric_features = numeric_features
        self.labels = labels
        self.tok_ids = tok_ids

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> FASequenceInstance:
        start = self.starts[idx]
        end = self.ends[idx]
        return FASequenceInstance(
            user_id=self.user_ids[start],
            ex_key=self.ex_keys[start],
            token_ids=self.token_ids[start:end],
            feature_ids={feat: ids[start:end] for feat, ids in self.feature_ids.items()},
            numeric_features={col: vals[start:end] for col, vals in self.numeric_features.items()},
            labels=self.labels[start:end],
            tok_ids=self.tok_ids[start:end],
            target_pos=np.arange(end - start, dtype=np.int64),
        )

def build_encoded_sequences(
    df: pd.DataFrame,
    vocabs: FAVocabs
) -> FASequenceDataset:

    ex_keys = df["ex_key"].to_numpy()
    starts = np.where(np.r_[True, ex_keys[1:] != ex_keys[:-1]])[0].astype(np.int64)
    ends = np.r_[starts[1:], len(df)].astype(np.int64)

    token_ids = string_series(df[TOKEN_COL]).map(vocabs.token_vocab).fillna(UNK_ID).astype(np.int64).to_numpy()
    feature_ids = {
        feat: string_series(df[feat]).map(feat_vocab).fillna(UNK_ID).astype(np.int64).to_numpy()
        for feat, feat_vocab in vocabs.feature_vocabs.items()
    }
    numeric_features = {
        col: df[col].fillna(0).astype(np.float32).to_numpy()
        for col in vocabs.numeric_feature_cols
    }
    labels = df["label"].astype(np.int64).to_numpy()
    tok_ids = df["tok_id"].to_numpy()
    user_ids = df["user_id"].to_numpy()

    return FASequenceDataset(user_ids, ex_keys, starts, ends, token_ids, feature_ids, numeric_features, labels, tok_ids)

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
