import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from config.consts import TRACKS
from data_processing.data_parquet import load_train_and_eval_df
from models.fa_bilstm.data import (
    NA_VALUE,
    TOKEN_COL,
    FAVocabs,
    build_encoded_sequences,
    build_vocab,
    fab_collate_fn,
    string_series,
)
from models.fa_bilstm.features import (
    add_global_history_num_features,
    add_metadata_num_features,
    add_positional_num_features,
    add_user_history_num_features,
    normalise_numeric_features as normalise_numeric_feature_frames,
)

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

    @property
    def numeric_feature_dim(self) -> int:
        return len(self.vocabs.numeric_feature_cols)


def resolve_feature_set(feature_set: str) -> tuple[str, ...]:
    if feature_set in FEATURE_SETS:
        return tuple(FEATURE_SETS[feature_set])

    cols = tuple(x.strip() for x in feature_set.split(","))
    logger.info(f"Resolving custom feature set {feature_set} to columns {cols}")
    
    out = []
    for col in cols:
        if col == "country":
            out.append("countries")
        elif col == "deprel":
            out.append("deprel")
        else:
            out.append(col)
    return tuple(out)
 
def normalise_frame(df: pd.DataFrame, track_name: str, feature_cols: Iterable[str], prefix_ids: bool) -> pd.DataFrame:
    df = df.copy()
    assert {TOKEN_COL, "user_id", "ex_key"}.issubset(df.columns), f"df is missing required columns: {df.columns}"

    df["ex_key"] = df["tok_id"].str.slice(0, 10)
    if "dep" not in df.columns:
        df["dep"] = df["deprel"] if "deprel" in df.columns else pd.NA
    if "countries" not in df.columns and "country" in df.columns:
        df["countries"] = df["country"]
    if prefix_ids:
        df["track"] = track_name
        for col in ("user_id", TOKEN_COL):
            df[col] = track_name + "_" + df[col].astype("string")

    df[TOKEN_COL] = string_series(df[TOKEN_COL])
    for col in feature_cols:
        if col not in df.columns:
            logger.warning(f"Feature column {col} is missing; filling with {NA_VALUE}")
            df[col] = pd.NA
        else:
            df[col] = string_series(df[col])

    assert not labels.isna().any(), "Labels has na values"

    return df


def load_dfs(track: str, variant: str, train_with_dev: bool, subset: int | None, feature_cols: tuple[str, ...]):
    if track != "all":
        df_train, df_eval = load_train_and_eval_df(track, variant, train_with_dev, subset=subset)
        return (
            normalise_frame(df_train, track, feature_cols, prefix_ids=False),
            normalise_frame(df_eval, track, feature_cols, prefix_ids=False),
        )

    train_xs = []
    eval_xs = []
    all_features = tuple(dict.fromkeys(feature_cols + ("track",)))
    for track_name in TRACKS:
        df_train, df_eval = load_train_and_eval_df(track_name, variant, train_with_dev, subset=subset)
        train_xs.append(normalise_frame(df_train, track_name, all_features, prefix_ids=True))
        eval_xs.append(normalise_frame(df_eval, track_name, all_features, prefix_ids=True))
    return pd.concat(train_xs, ignore_index=True), pd.concat(eval_xs, ignore_index=True)

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
    
    feature_cols = resolve_feature_set(feature_set)

    logger.info(f"FA_dataloaders: Loading data for track {track} with features {feature_cols}")

    # === load dfs
    df_train, df_eval = load_dfs(track, variant, train_with_dev, subset, feature_cols)
    if track == "all": # mark track in feature col for all track
        feature_cols = tuple(dict.fromkeys(feature_cols + ("track",)))

    numeric_feat_cols = []
    if position_features:
        df_train, df_eval, cols = add_positional_num_features(df_train, df_eval)
        numeric_feat_cols.extend(cols)
    if user_history_features:
        df_train, df_eval, cols = add_user_history_num_features(df_train, df_eval)
        numeric_feat_cols.extend(cols)
    if global_history_features:
        df_train, df_eval, cols = add_global_history_num_features(df_train, df_eval)
        numeric_feat_cols.extend(cols)
    if numeric_metadata:
        df_train, df_eval, cols = add_metadata_num_features(df_train, df_eval)
        numeric_feat_cols.extend(cols)
    numeric_feat_cols = tuple(dict.fromkeys(numeric_feat_cols))
    if normalise_numeric_features:
        df_train, df_eval = normalise_numeric_feature_frames(df_train, df_eval, numeric_feat_cols)
    # ==== FAVocabs

    vocabs = FAVocabs(
        token_vocab=build_vocab(df_train[TOKEN_COL]),
        feature_vocabs={feat: build_vocab(df_train[feat]) for feat in feature_cols},
        feature_cols=feature_cols,
        numeric_feature_cols=numeric_feat_cols
    )

    train_encoded = build_encoded_sequences(df_train, vocabs)
    eval_encoded = build_encoded_sequences(df_eval, vocabs)

    train_dl = DataLoader(train_encoded, batch_size=batch_size, shuffle=shuffle_train, collate_fn=fab_collate_fn)
    eval_dl = DataLoader(eval_encoded, batch_size=batch_size, shuffle=False, collate_fn=fab_collate_fn)

    logger.info(f"FA_dataloaders: Built dataloaders with {len(train_dl)} train batches and {len(eval_dl)} eval batches")

    return FAData(train_dl=train_dl, eval_dl=eval_dl, vocabs=vocabs)




