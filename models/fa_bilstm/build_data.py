import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from models.fa_bilstm.data import FAVocabs
from torch.utils.data import DataLoader

from config.consts import TRACKS
from data_processing.data_parquet import load_train_and_eval_df

logger = logging.getLogger(__name__)

# use tuples for immutability 
EXERCISE_FEATURES = ("user_id", "format", "session", "client", "countries")
FEATURE_SETS = {
    "token-only": (),
    "exercise": EXERCISE_FEATURES,
    "pos": EXERCISE_FEATURES + ("pos",),
    "dep": EXERCISE_FEATURES + ("dep",),
    "posdep": EXERCISE_FEATURES + ("pos", "dep"),
    "all": EXERCISE_FEATURES + ("pos", "dep"),
}
    
@dataclass(frozen=True)
class FAData:
    train_dl: DataLoader
    eval_dl: DataLoader
    vocabs: FAVocabs

def build_fab_dataloaders(
    track: str,
    variant: str,
    subset: int | None,
    train_with_dev: bool,
    feature_set: str,
    batch_size: int,
    shuffle_train: bool = True,
    user_history_features: bool = False,
    global_history_features: bool = False,
    position_features: bool = False,
    numeric_metadata: bool = False,
    normalise_numeric_features: bool = False,
) -> FAData:
    
    assert feature_set in FEATURE_SETS
    feature_cols = FEATURE_SETS[feature_set]

    logger.info(f"FA_dataloaders: Loading data for track {track} with features {feature_cols}")

    # === load dfs
    df_train, df_eval = load_train_and_eval_df(track, subset, train_with_dev)
    df_train["ex_key"] = df_train["tok_id"].astype(str).slice(0, 10)
    df_eval["ex_key"] = df_eval["tok_id"].astype(str).slice(0, 10)
    assert set(feature_cols).issubset(set(df_train.columns))
    assert set(feature_cols).issubset(set(df_eval.columns))

    numeric_feat_cols = []
    if position_features:
        add_position_features(df_train)
    if user_history_features:
        add_user_history_features(df_train, df_eval)
    if global_history_features:
        add_global_history_features(df_train, df_eval)
    if numeric_metadata:
        add_numeric_metadata_features(df_train, df_eval)
    if normalise_numeric_features:
        numeric_feat_cols = normalise_numeric_features(df_train, df_eval)
    # ==== FAVocabs

    vocabs = FAVocabs(
        token_vocab=build_vocab(df_train["tok"]),
        feature_vocabs={feat: build_vocab(df_train[feat]) for feat in feature_cols},
        feature_cols=feature_cols,
        numeric_feature_cols=numeric_feat_cols
    )

    train_encoded = build_encoded_sequences(df_train, vocabs)
    eval_encoded = build_encoded_sequences(df_eval, vocabs)

    train_dl = DataLoader(train_encoded, batch_size=batch_size, shuffle=shuffle_train, collate_fn=fab_collate_fn)
    eval_dl = DataLoader(eval_encoded, batch_size=batch_size, shuffle=False)

    logger.info(f"FA_dataloaders: Built dataloaders with {len(train_dl)} train batches and {len(eval_dl)} eval batches")

    return FAData(train_dl=train_dl, eval_dl=eval_dl, vocabs=vocabs)


    
        






