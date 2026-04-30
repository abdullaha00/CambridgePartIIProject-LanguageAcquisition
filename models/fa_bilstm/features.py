from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from .data import TOKEN_COL

logger = logging.getLogger(__name__)

USER_HISTORY_ALPHA = 10.0
GLOBAL_HISTORY_ALPHA = 20.0

POS_NUM_COLS = [
    "pos_ex_len_log",
    "pos_index_log",
    "pos_from_end_log",
    "pos_rel",
    "pos_is_first",
    "pos_is_last",
    "tok_len_log",
    "tok_is_title",
    "tok_is_upper",
    "tok_has_digit",
    "tok_has_apostrophe",
    "tok_has_punct"
]

META_SOURCE_COLS = ("days", "time", "rt", "ex_inst_idx")
META_CYCLIC_COLS = ("meta_days_frac_sin", "meta_days_frac_cos")


def metadata_invalid_mask(source_col: str, vals: pd.Series) -> pd.Series:
    if source_col == "time":
        # time<0 occurs in the dataset, we treat as invalid 
        return vals < 0
    if source_col in META_SOURCE_COLS:
        return vals < 0
    return pd.Series(False, index=vals.index)


def key(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.Series(zip(left, right), index=left.index)


def add_positional_num_features(df_train: pd.DataFrame, df_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:

    for df in (df_train, df_eval):

        # === 
        g = df.groupby("ex_key", sort=False)[TOKEN_COL]
        pos_in_ex = (g.cumcount() + 1).astype(np.float32)
        ex_len = g.transform("size").astype(np.float32)
        assert ex_len.min() > 0, "Invalid ex"
        assert pos_in_ex.min() >= 1, "Invalid position"
        # ===

        # == EXERCISE POS FEATS

        # np default is float64, switch to 32 to save memory
        df["pos_ex_len_log"] = np.log(ex_len).astype(np.float32)
        df["pos_index_log"] = np.log(pos_in_ex).astype(np.float32)
        df["pos_from_end_log"] = np.log((ex_len - pos_in_ex + 1).clip(lower=0)).astype(np.float32)
        df["pos_rel"] = ((pos_in_ex - 1) / ((ex_len - 1)).clip(lower=1)).astype(np.float32)
        df["pos_is_first"] = (pos_in_ex == 1).astype(np.float32)
        df["pos_is_last"] = (pos_in_ex == ex_len).astype(np.float32)

        # == Tok features

        df["tok_len_log"] = np.log1p(df[TOKEN_COL].str.len().fillna(0).astype(np.float32)).astype(np.float32)
        df["tok_is_title"] = df[TOKEN_COL].str.istitle().fillna(False).astype(np.float32)
        df["tok_is_upper"] = df[TOKEN_COL].str.isupper().fillna(False).astype(np.float32)
        df["tok_has_digit"] = df[TOKEN_COL].str.contains(r"\d", regex=True, na=False).astype(np.float32)
        df["tok_has_apostrophe"] = df[TOKEN_COL].str.contains("'", regex=False, na=False).astype(np.float32)
        df["tok_has_punct"] = df[TOKEN_COL].str.contains(r"[^\w\s']", regex=True, na=False).astype(np.float32)

    return df_train, df_eval, tuple(POS_NUM_COLS)


def add_user_history_num_features(df_train: pd.DataFrame, df_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    global_train_label_mean = df_train["label"].mean()

    # we group 2 cols into (col1, col2) pairs
    groups = [
        ("user", df_train["user_id"], df_eval["user_id"]),
        ("user_token", key(df_train["user_id"], df_train[TOKEN_COL]), key(df_eval["user_id"], df_eval[TOKEN_COL])),
        ("user_format", key(df_train["user_id"], df_train["format"]), key(df_eval["user_id"], df_eval["format"])),
        ("user_country", key(df_train["user_id"], df_train["countries"]), key(df_eval["user_id"], df_eval["countries"])),
    ]

    num_cols = []
    for name, train_key, eval_key in groups:
        #===== TRAIN ROWS: we can use only previous labels here
        train_counts = df_train.groupby(train_key, sort=False).cumcount().astype(np.float32)
        train_errors = (df_train.groupby(train_key, sort=False)["label"].cumsum() - df_train["label"]).astype(np.float32)

        #===== EVAL ROWS: no labels; just copy aggregate train data
        train_stats = df_train.groupby(train_key, sort=False)["label"].agg(["size", "sum"])

        eval_counts = eval_key.map(train_stats["size"]).astype(np.float32)
        eval_errors = eval_key.map(train_stats["sum"]).astype(np.float32)

        #=====
        count_col = f"hist_{name}_log_count"
        error_rate_col = f"hist_{name}_error_rate"
        num_cols.extend([count_col, error_rate_col])

        df_train[count_col] = np.log1p(train_counts).astype(np.float32)
        df_train[error_rate_col] = ((train_errors + USER_HISTORY_ALPHA * global_train_label_mean) / (train_counts + USER_HISTORY_ALPHA)).astype(np.float32)

        df_eval[count_col] = np.log1p(eval_counts).astype(np.float32)
        df_eval[error_rate_col] = ((eval_errors + USER_HISTORY_ALPHA * global_train_label_mean  ) / (eval_counts + USER_HISTORY_ALPHA)).astype(np.float32)

    return df_train, df_eval, tuple(num_cols)

def add_global_history_num_features(df_train: pd.DataFrame, df_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:

    #we group by col1, col2 and compute counts/errors
    
    global_train_label_mean = float(df_train["label"].mean())

    train_pos_in_ex = df_train.groupby("ex_key", sort=False)[TOKEN_COL].cumcount().clip(upper=12)
    eval_pos_in_ex = df_eval.groupby("ex_key", sort=False)[TOKEN_COL].cumcount().clip(upper=12)

    groups = [
        ("token", df_train[TOKEN_COL], df_eval[TOKEN_COL]),
        ("format", df_train["format"], df_eval["format"]),
        ("country", df_train["countries"], df_eval["countries"]),
        ("token_format", key(df_train[TOKEN_COL], df_train["format"]), key(df_eval[TOKEN_COL], df_eval["format"])),
        ("format_pos", key(df_train["format"], train_pos_in_ex), key(df_eval["format"], eval_pos_in_ex)),
        ("token_pos", key(df_train[TOKEN_COL], train_pos_in_ex), key(df_eval[TOKEN_COL], eval_pos_in_ex)),
    ]

    num_cols = []
    for name, train_key, eval_key in groups:
        #===== TRAIN ROWS: we can use only previous labels here
        train_counts = df_train.groupby(train_key, sort=False).cumcount().astype(np.float32)
        train_errors = (df_train.groupby(train_key, sort=False)["label"].cumsum() - df_train["label"]).astype(np.float32)

        #===== EVAL ROWS: no labels; just copy aggregate train data
        train_stats = df_train.groupby(train_key, sort=False)["label"].agg(["size", "sum"])

        eval_counts = eval_key.map(train_stats["size"]).fillna(0).astype(np.float32)
        eval_errors = eval_key.map(train_stats["sum"]).fillna(0).astype(np.float32)

        #=====
        count_col = f"hist_{name}_log_count"
        error_rate_col = f"hist_{name}_error_rate"
        num_cols.extend([count_col, error_rate_col])

        df_train[count_col] = np.log1p(train_counts).astype(np.float32)
        df_train[error_rate_col] = ((train_errors + GLOBAL_HISTORY_ALPHA * global_train_label_mean) / (train_counts + GLOBAL_HISTORY_ALPHA)).astype(np.float32)

        df_eval[count_col] = np.log1p(eval_counts).astype(np.float32)
        df_eval[error_rate_col] = ((eval_errors + GLOBAL_HISTORY_ALPHA * global_train_label_mean  ) / (eval_counts + GLOBAL_HISTORY_ALPHA)).astype(np.float32)

    return df_train, df_eval, tuple(num_cols)


def add_metadata_num_features(df_train: pd.DataFrame, df_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:

    num_cols = []
    
    for source_col in META_SOURCE_COLS:
        num_log_col = f"meta_{source_col}_log"
        num_missing_col = f"meta_{source_col}_missing"
        num_cols.extend([num_log_col, num_missing_col])

        for df in (df_train, df_eval):
            if source_col in df.columns:
                vals = pd.to_numeric(df[source_col], errors="coerce")
                missing = vals.isna() | metadata_invalid_mask(source_col, vals)
            else:
                logger.warning(f"Source column {source_col} not found in dataframe; filling with zeros and marking all as missing")
                vals = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)
                missing = pd.Series(np.ones(len(df), dtype=bool), index=df.index)
            vals = vals.mask(missing, 0)

            df[num_log_col] = np.log1p(vals).astype(np.float32)
            df[num_missing_col] = missing.astype(np.float32)

    for df in (df_train, df_eval):

        days = df["days"].fillna(0).astype(np.float32)
        day_phase = (days % 1.0).to_numpy(dtype=np.float32)

        df["meta_days_frac_sin"] = np.sin(2 * np.pi * day_phase).astype(np.float32)
        df["meta_days_frac_cos"] = np.cos(2 * np.pi * day_phase).astype(np.float32)

    num_cols.extend(META_CYCLIC_COLS)

    return df_train, df_eval, tuple(num_cols)


def normalise_numeric_features(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    num_cols: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Normalise based on train only

    for col in num_cols:
        mean = df_train[col].mean()
        std = df_train[col].std()

        if pd.isna(std) or std == 0:
            logger.warning(f"Standard deviation for column {col} is zero or na; setting to 1.0")
            std = 1.0

        df_train[col] = ((df_train[col] - mean) / std).astype(np.float32)
        df_eval[col] = ((df_eval[col] - mean) / std).astype(np.float32)

    return df_train, df_eval
